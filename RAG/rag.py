import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def update_vectorstore(text_chunks, embeddings, existing_vectorstore=None):
    new_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    if existing_vectorstore:
        existing_vectorstore.merge(new_vectorstore)
        return existing_vectorstore
    return new_vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2, top_p=0.85)  # Initialize Google API model
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        st.session_state.previous_questions.append(user_question)

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Please upload and process your documents first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "previous_questions" not in st.session_state:
        st.session_state.previous_questions = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Initialize Google API Embeddings
                    
                    if st.session_state.vectorstore:
                        st.session_state.vectorstore = update_vectorstore(text_chunks, embeddings, st.session_state.vectorstore)
                    else:
                        # Initialize the vector store with new documents
                        st.session_state.vectorstore = update_vectorstore(text_chunks, embeddings)
                    
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                    st.session_state.chat_history = []  # Clear history on new document load
                    st.session_state.previous_questions = []  # Clear previous questions on new document load
            else:
                st.write("Please upload at least one PDF.")

        st.subheader("Previous Questions")
        if st.session_state.previous_questions:
            for question in st.session_state.previous_questions:
                st.write(f"- {question}")

if __name__ == '__main__':
    main()
