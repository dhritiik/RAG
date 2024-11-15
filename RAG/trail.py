import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import FAISS

# Step 1: Extract text and page numbers from PDF documents
def get_pdf_text(pdf_docs):
    pdf_text_with_page_numbers = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pdf_text_with_page_numbers.append((page_text, page_num))
            else:
                print(f"No text found on page: {page_num}")
    return pdf_text_with_page_numbers

# Step 2: Split text into chunks and manually track page ranges for each chunk
def get_text_chunks_with_page_ranges(text_with_page_numbers):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks_with_metadata = []
    concatenated_text = ""
    start_page = None

    # Manually track page ranges
    for text, page_num in text_with_page_numbers:
        if start_page is None:
            start_page = page_num
        concatenated_text += text

        # Split the concatenated text into chunks
        chunks = text_splitter.split_text(concatenated_text)

        for chunk in chunks:
            # Add metadata with page range
            end_page = page_num  # The last page of the current chunk
            chunks_with_metadata.append({
                "chunk": chunk,
                "start_page": start_page,
                "end_page": end_page
            })
            concatenated_text = ""  # Reset concatenated text after splitting

        # Set new start_page for next chunk
        start_page = page_num

    print(f"Number of chunks created: {len(chunks_with_metadata)}")
    return chunks_with_metadata

# Step 3: Create a vectorstore from text chunks with page ranges as metadata
def get_vectorstore(chunks_with_metadata):
    if not chunks_with_metadata:
        raise ValueError("The text chunks list is empty. Make sure to provide valid text data.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    texts = [chunk['chunk'] for chunk in chunks_with_metadata]
    metadatas = [{"start_page": chunk['start_page'], "end_page": chunk['end_page']} for chunk in chunks_with_metadata]

    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vectorstore

# Step 4: Create a conversational chain that retrieves metadata including page ranges
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"return_metadata": True}),
        memory=memory
    )
    return conversation_chain

# Step 5: Handle user input and display answers with page ranges
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # Bot response
            if 'source_documents' in response and response['source_documents']:
                sources = response['source_documents']
                # Display all retrieved sources with page ranges
                source_info = []
                for source in sources:
                    start_page = source.metadata.get('start_page', 'Unknown')
                    end_page = source.metadata.get('end_page', 'Unknown')
                    source_info.append(f"Pages {start_page}-{end_page}")
                
                # Format the bot response with the page range(s) included
                bot_response_with_page = f"{message.content}\n\n_Source(s): {', '.join(source_info)}_"
            else:
                bot_response_with_page = message.content

            st.write(bot_template.replace("{{MSG}}", bot_response_with_page), unsafe_allow_html=True)

# Main app function
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF document.")
                    return
                raw_text_with_page_numbers = get_pdf_text(pdf_docs)
                chunks_with_metadata = get_text_chunks_with_page_ranges(raw_text_with_page_numbers)
                if not chunks_with_metadata:
                    st.error("No valid text chunks created. Please check your PDF files.")
                    return

                # Get the vectorstore and conversation chain
                vectorstore = get_vectorstore(chunks_with_metadata)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("PDF processed successfully!")

if __name__ == '__main__':
    main()
