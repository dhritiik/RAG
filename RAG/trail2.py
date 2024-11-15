import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import FAISS
from fpdf import FPDF

# Function to extract text and page numbers from PDF documents
def extract_pdf_text(pdf_docs):
    pdf_text_with_page_numbers = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_text_with_page_numbers.extend(
            (page.extract_text(), page_num + 1) for page_num, page in enumerate(pdf_reader.pages) if page.extract_text()
        )
    return pdf_text_with_page_numbers

# Function to split text into chunks, including metadata
def split_text_into_chunks(text_with_page_numbers):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return [{"chunk": chunk, "page_num": page_num} for text, page_num in text_with_page_numbers for chunk in text_splitter.split_text(text)]

# Function to create a vectorstore from text chunks with metadata
def create_vectorstore(chunks_with_metadata):
    if not chunks_with_metadata:
        st.error("No text chunks found. Please check your PDF files.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [chunk['chunk'] for chunk in chunks_with_metadata]
    metadatas = [{"page_num": chunk['page_num']} for chunk in chunks_with_metadata]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

# Function to initialize a conversational chain for metadata retrieval
def initialize_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"return_metadata": True}), 
        memory=memory
    )

# Function to summarize text chunks
def summarize_chunks(text_chunks):
    summarization_prompt = PromptTemplate(template="Summarize the following text:\n\n{text}", input_variables=["text"])
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
    summary_chain = LLMChain(llm=llm, prompt=summarization_prompt)
    return summary_chain.run({"text": " ".join([chunk['chunk'] for chunk in text_chunks])})

# Function to create a downloadable PDF of the chat history
def generate_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Conversation with the Bot", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for i, message in enumerate(chat_history):
        pdf.set_text_color(0, 0, 255 if i % 2 == 0 else 255, 0, 0)
        prefix = "User: " if i % 2 == 0 else "Bot: "
        pdf.multi_cell(0, 10, f"{prefix}{message.content}")
        pdf.ln(5)

    pdf_path = "/tmp/conversation.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Function to generate questions from summary
def generate_questions_from_summary(summary_text):
    question_generation_prompt = PromptTemplate(
        template="Generate 10 important questions from each chapter of the uploaded PDF content:\n\n{text}", 
        input_variables=["text"]
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
    question_chain = LLMChain(llm=llm, prompt=question_generation_prompt)
    return question_chain.run({"text": summary_text})

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    for key in ["conversation", "chat_history", "summary", "show_summary", "questions_answers"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                pdf_text = extract_pdf_text(pdf_docs)
                if not pdf_text:
                    st.error("No valid text extracted. Please check your PDF files.")
                    return

                text_chunks = split_text_into_chunks(pdf_text)
                vectorstore = create_vectorstore(text_chunks)
                if vectorstore:
                    st.session_state.conversation = initialize_conversation_chain(vectorstore)
                    st.session_state.summary = summarize_chunks(text_chunks)
                    st.success("PDFs processed successfully!")
                    st.session_state.show_summary = False

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        
        if st.button("Download Conversation as PDF"):
            pdf_file = generate_pdf(st.session_state.chat_history)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name="conversation.pdf", mime="application/pdf")

    if st.session_state.summary and st.button("View Summary"):
        st.session_state.show_summary = not st.session_state.show_summary
    if st.session_state.show_summary:
        st.subheader("Document Summary")
        st.write(st.session_state.summary)

    if st.session_state.summary and st.button("Generate Questions from Summary"):
        st.session_state.questions_answers = generate_questions_from_summary(st.session_state.summary)
    if st.session_state.questions_answers:
        st.subheader("Generated Questions")
        st.write(st.session_state.questions_answers)

if __name__ == '__main__':
    main()
