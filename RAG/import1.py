import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings

# Replace with your Hugging Face API token
login("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")

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Function to initialize the conversation chain using Meta LLaMA
def get_conversation_chain(vectorstore):
    # Load Meta LLaMA model and tokenizer from Hugging Face
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Example model name, adjust based on your needs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set up a text generation pipeline
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=100,
        temperature=0.7, 
        top_p=0.9
    )

    # Wrap the pipeline with HuggingFacePipeline to make it LangChain-compatible
    llm = HuggingFacePipeline(pipeline=pipe)

    # Use LangChain's ConversationalRetrievalChain
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Use the wrapped model here
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to update or initialize the vectorstore with embeddings
def update_vectorstore(text_chunks, embeddings, existing_vectorstore=None):
    # Create a new FAISS index from the text chunks and embeddings
    new_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    # Check if an existing vectorstore is provided
    if existing_vectorstore:
        # Retrieve vectors from both the existing and new vector stores
        existing_vectors = existing_vectorstore.index.reconstruct_n(0, existing_vectorstore.index.ntotal)
        new_vectors = new_vectorstore.index.reconstruct_n(0, new_vectorstore.index.ntotal)
        
        # Combine the vectors
        combined_vectors = np.vstack((existing_vectors, new_vectors))
        
        # Create a new FAISS index with the combined vectors
        combined_index = FAISS.from_embeddings(combined_vectors, embeddings)
        return combined_index
    
    # Return the new vectorstore if no existing vectorstore is provided
    return new_vectorstore

# Function to initialize the conversation chain using Meta LLaMA
# def get_conversation_chain(vectorstore):
#     # Load Meta LLaMA model and tokenizer from Hugging Face
#     model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
#     model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

#     # Use LangChain's ConversationalRetrievalChain
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=model,  # Meta LLaMA model
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# Function to handle user input and display chat history
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

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
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

    # Handle user question
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Extract text and split into chunks
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)

                    # Initialize Hugging Face embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    # Update or initialize vectorstore
                    if st.session_state.vectorstore:
                        st.session_state.vectorstore = update_vectorstore(text_chunks, embeddings, st.session_state.vectorstore)
                    else:
                        st.session_state.vectorstore = update_vectorstore(text_chunks, embeddings)
                    
                    # Initialize conversation chain
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                    st.session_state.chat_history = []  # Clear history on new document load
                    st.session_state.previous_questions = []  # Clear previous questions on new document load
            else:
                st.write("Please upload at least one PDF.")

        # Display previous questions
        st.subheader("Previous Questions")
        if st.session_state.previous_questions:
            for question in st.session_state.previous_questions:
                st.write(f"- {question}")

if __name__ == '__main__':
    main()
