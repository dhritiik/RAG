from django.shortcuts import render



# Create your views here.
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import numpy as np

# Hugging Face API token
login("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")

# Helper functions

def get_pdf_text(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into smaller chunks."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def update_vectorstore(text_chunks, embeddings, existing_vectorstore=None):
    new_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    if existing_vectorstore:
        existing_vectors = existing_vectorstore.index.reconstruct_n(0, existing_vectorstore.index.ntotal)
        new_vectors = new_vectorstore.index.reconstruct_n(0, new_vectorstore.index.ntotal)
        combined_vectors = np.vstack((existing_vectors, new_vectors))
        combined_index = FAISS.from_embeddings(combined_vectors, embeddings)
        return combined_index
    return new_vectorstore

def get_conversation_chain(vectorstore):
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.7, top_p=0.9)
    
    from langchain_community.llms import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Views
def index(request):
    return render(request, 'index.html')

def process_pdfs(request):
    if request.method == 'POST' and request.FILES.getlist('pdfs'):
        pdfs = request.FILES.getlist('pdfs')
        raw_text = get_pdf_text(pdfs)
        text_chunks = get_text_chunks(raw_text)
        
        # Initialize embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = update_vectorstore(text_chunks, embeddings)
        
        # Save conversation chain in session
        conversation_chain = get_conversation_chain(vectorstore)
        request.session['conversation_chain'] = conversation_chain
        
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'failed'})

def ask_question(request):
    if request.method == 'POST':
        user_question = request.POST.get('question')
        conversation_chain = request.session.get('conversation_chain')
        
        if conversation_chain:
            response = conversation_chain({'question': user_question})
            chat_history = response['chat_history']
            return JsonResponse({'chat_history': chat_history})
    return JsonResponse({'status': 'failed'})
