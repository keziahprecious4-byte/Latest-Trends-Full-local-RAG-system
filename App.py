import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_core.prompts import PromptTemplate

try:
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
except ImportError:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import DocumentCompressorPipeline

st.header("Ask me anything")

@st.cache_resource
def load_embedding_model():
    embedding_model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name = embedding_model_name,
        encode_kwargs = encode_kwargs
    )
    return embedding_model

def load_retriever1():
    embedding_model = load_embedding_model()
    vector_store = Chroma(persist_directory = "./vectorDb", embedding = embedding_model)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 6})
    return retriever

def load_retriever2():
    embedding_model = load_embedding_model()
    vector_store = Chroma(persist_directory="./vectorDB", embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 6})
    return retriever

def load_final_retriever():
    retriever1 = load_retriever1()
    retriever2 = load_retriever1()
    ensemble_retriever = EnsembleRetriever(
        retrievers = [retriever1, retriever2], weights = [0.5, 0.5]
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2], weights=[0.5, 0.5]
    )
    reordering = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[reordering]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=ensemble_retriever
    )
    return compression_retriever

def load_prompt():
    template = """
###Intructions:
You are an assistant for question-answering taks. \
Use the following pieces of retrieved context and Conversation to answer the latest questions in Conversation. \
Conversation have the latest question and may also have chat history. \
if you dont know the answer, just say you dont know, DO NOT provide incorrect answer. \

### Context:
{context}

### Conversation:
{question}

### Responce:
"""
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt

user_query = st.chat_input("Ask me anything")