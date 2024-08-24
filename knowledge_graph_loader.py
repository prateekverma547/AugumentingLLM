from ast import literal_eval
import pandas as pd
from datasets import load_dataset
import os
from dotenv import load_dotenv
from os.path import join, dirname

from transformers import pipeline
from tqdm import tqdm
from data_preprocessor import DataPreprocessor
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

HOST = os.environ.get('HOST')
PORT = os.environ.get('PORT')
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
NEO4J_URI=os.environ.get('NEO4J_URI')
NEO4J_USERNAME=os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD=os.environ.get('NEO4J_PASSWORD')

class KGLoader:
    def __init__(self):
        print("KG loaded")
        self.preprocess_data = DataPreprocessor()
        self.data = self.preprocess_data.preprocess_data()
        self.loader = DataFrameLoader(self.data, page_content_column="sentence")
        self.SentenceDocuments = self.loader.load()

    def split_texts(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=30,
            length_function=len,
        )
        split_doc = text_splitter.split_documents(self.SentenceDocuments)
        return split_doc
    
    def create_knowledge_graph(self):
        
        graph = Neo4jGraph()
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        llm_transformer = LLMGraphTransformer(llm=llm)
        print("Hi")
        split_doc= self.split_texts()
        print("Hi1")
        graph_documents = llm_transformer.convert_to_graph_documents(split_doc)
        print("Hi2")
        print(f"Nodes:{graph_documents[0].nodes}")
        print(f"Relationships:{graph_documents[0].relationships}")
        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

newGrpah=KGLoader()
newGrpah.create_knowledge_graph()