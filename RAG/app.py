import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import

# Ensure the Hugging Face API token is set
os.environ["HUGGINGFACEHUB_API_TOKEN"] = " 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"  # Replace with your actual Hugging Face token


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Option 1: Upgrade sentence-transformers (recommended)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # Option 2: Use from_pretrained if upgrade is not feasible
   # embeddings = HuggingFaceInstructEmbeddings.from_pretrained("hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title="RAG", page_icon=":books:")

    st.header("RAG :books:")
    st.text_input("Ask your Question")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get the pdf and return it as a single string
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                text_chunks = get_text_chunk(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write("Vectorstore created successfully.")

if __name__ == '__main__':
    main()
