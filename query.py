import os
import sys
from os.path import join, dirname

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from vector_index_loader import IndexBuilder

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

OPENAI_KEY = os.environ.get('OPENAI_API_KEY')

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class QueryEngine(metaclass=Singleton):

    index = IndexBuilder()
    chroma_client = index.create_embeddings()
    # embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding_function = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
    print("ALL Embeeding added")
    chroma_db = Chroma(
        client=chroma_client,
        collection_name='financial_reports',
        embedding_function=embedding_function,
    )

        #Use three sentences maximum and keep the answer as concise as possible.

    def prompt(self):
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
        return PromptTemplate.from_template(template)
    
    def get_retriver(self):
        retriever=self.chroma_db.as_retriever(search_kwargs={"k": 5})
        return retriever
    
    def create_llm_and_chain(self):
        
        llm = ChatOpenAI(
            model_name='gpt-4-0613',
            )
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=self.get_retriver(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt()}
            )
        return chain

    def query_engine(self, query):
        mychain=self.create_llm_and_chain()
        response = mychain({'query': query})
        return response['result']
