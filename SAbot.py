import os
import openai
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

class sabot:
    def __init__(self):
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
        
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.encoder = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L12-v2', 
            model_kwargs={'device': "cpu", 'use_auth_token': self.HUGGINGFACE_API_KEY}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2",
            use_auth_token=self.HUGGINGFACE_API_KEY
        )
        self.vector_db = None  # Initialize vector_db as None

    def generate(self, question: str, context: str = None, temperature: float = 0.2, max_tokens: int = 400):
        if context:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Using the information contained in the context, give a detailed answer to the question. Context: {context}. Question: {question}"}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Give a detailed answer to the following question. Question: {question}"}
            ]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", #gpt-4o changed
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
        )

        text = response.choices[0].message.content.strip()
        truncated_text = self.truncate_to_nearest_sentence(text, max_tokens - 50)

        if context is None:
            additional_info = "Note: The question was out of the scope of the provided document."
            return f"{truncated_text}\n\n{additional_info}"
        else:
            return truncated_text

    def truncate_to_nearest_sentence(self, text: str, max_tokens: int) -> str:
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text

        truncated_text = " ".join(tokens[:max_tokens])
        last_sentence_end = max(truncated_text.rfind('.'), truncated_text.rfind('!'), truncated_text.rfind('?'))

        if last_sentence_end != -1:
            return truncated_text[:last_sentence_end + 1]
        else:
            return truncated_text

    def embed_query(self, query: str):
        return self.encoder.embed_query(query)

    def cosine_similarity(self, q, z):
        return np.dot(q, z) / (np.linalg.norm(q) * np.linalg.norm(z))

    def load_documents(self, pdf_paths):
        loaders = [PyPDFLoader(path) for path in pdf_paths]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=256,
            chunk_overlap=32,
            strip_whitespace=True
        )
        return text_splitter.split_documents(pages)

    def create_vector_db(self, docs):
        self.vector_db = FAISS.from_documents(docs, self.encoder, distance_strategy=DistanceStrategy.COSINE)

    def similarity_search(self, query, k=5):
        if self.vector_db:
            results = self.vector_db.similarity_search(query, k=k)
            return results if results else None
        return None
