import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import chromadb
from langchain.llms import GooglePalm
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llamaapi import LlamaAPI
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.llms import GooglePalm

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from llamaapi import LlamaAPI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
import aspose.words as aw
from llamaapi import LlamaAPI

from langchain_experimental.llms import ChatLlamaAPI
from llamaapi import LlamaAPI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
import aspose.words as aw
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import requests
from bs4 import BeautifulSoup
# Replace 'Your_API_Token' with your actual API token
llama = LlamaAPI("LL-rue4yBAEM6QIKlDyRI4klfII3cWnZ7KzyR8OsH9HhuZ9L2il2p0AvOuXSvvTUBO5")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = ChatLlamaAPI(client=llama)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain



chat_history = []
output_text_path = "Output.txt"

def convert_docx_to_text(docs):
    # Load the Word document
    doc = aw.Document(docs)

    # Save the document as a text file
    doc.save(output_text_path, aw.SaveFormat.TEXT)

    # Read the text content from the saved text file
    with open(output_text_path, 'r', encoding='utf-8') as text_file:
        text_content = text_file.read()

    return text_content

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text
def load_data_from_url(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks
def get_text_chunksd(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=20)
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",    task_type="semantic_similarity")
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
def get_vector_stored(text_chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",    task_type="semantic_similarity")
    #embeddings = HuggingFaceInstructEmbeddings()
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def extract_text_from_url(url):
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        text_content = soup.get_text(separator=' ')

        return text_content
def get_conversational_chain():

    prompt_template = """
    <s>[INST] <<SYS>>
    You are a helpful AI assistant.
    Answer based on the context provided. If you cannot find the correct answer, and answer should be from context say I don't know. 
    \n when some one ask hi you just say how may i help you .
    if you dont found information found in context then it will say ,sorry i dont know about this 
    <</SYS>>
    {context}
    Question: {user_question}
    Helpful Answer: [/INST]

    Answer:
    """

    #model = ChatGoogleGenerativeAI(model="gemini-pro",
                             #emperature=0.0)
    model = ChatLlamaAPI(client=llama)


    prompt = PromptTemplate.from_template(prompt_template)
    #prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    #chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    chain = ConversationChain(
    model=model,
    prompt=prompt,
    memory=ConversationBufferMemory()
                    )
    
    return chain



def user_input(user_question):
    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    #embeddings = HuggingFaceInstructEmbeddings()
    embeddings = OpenAIEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = ConversationalRetrievalChain.from_llm(model, new_db.as_retriever(), return_source_documents=True)


    #chain = get_conversational_chain()

    
    
    response = chain(
        { "question": user_question,"chat_history": chat_history})
    chat_history.append({'user_question': user_question, 'response': response["answer"]})
    print(response) 
    st.write("Assistant: ", response["answer"])
    


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Llama 2💁")

    user_question = st.text_input("Ask a Question from the documents Files")

    if user_question:
       user_input(user_question)

       
    with st.sidebar:
        st.title("Menu:")

        st.markdown("<hr>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



        st.markdown("<hr>", unsafe_allow_html=True)        
        Docs= st.file_uploader("Upload your DOC Files and Click on the Process Document  Button")
        if st.button(" Process Document"):
            with st.spinner("Processing..."):
                data = convert_docx_to_text(Docs)
                text_chunks = get_text_chunks(data)
                get_vector_store(text_chunks)
                st.success("Done")

        st.markdown("<hr>", unsafe_allow_html=True)
        url= st.text_input("Enter Site URL")
        if st.button(" Process URL"):
            with st.spinner("Processing..."):
                data = extract_text_from_url(url)
                text_chunks = get_text_chunks(data)
                get_vector_store(text_chunks)
                st.success("Done")


       
                



if __name__ == "__main__":
    main()
