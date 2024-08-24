import os
from os.path import join, dirname

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader

from data_preprocessor import DataPreprocessor
from sentence_transformers import SentenceTransformer


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

HOST = os.environ.get('HOST')
PORT = os.environ.get('PORT')
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')


class IndexBuilder:
    def __init__(self):
        print("index loaded")
        self.preprocess_data = DataPreprocessor()
        self.data = self.preprocess_data.preprocess_data()
        self.loader = DataFrameLoader(self.data, page_content_column="sentence")
        self.documents = self.loader.load()

    def split_texts(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=430,
            chunk_overlap=20,
            length_function=len,
        )
        split_doc = text_splitter.split_documents(self.documents)
        return split_doc

    def create_embeddings(self):
        chroma_client = chromadb.PersistentClient(path='chroma')
        # openai_embeddings = embedding_functions.OpenAIEmbeddingFunction(
        #     api_key=OPENAI_KEY,
        #     model_name="text-embedding-ada-002"
        # )
        # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        #     api_key="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",
        #     model_name="sentence-transformers/multi-qa-distilbert-cos-v1"
        # )
        emb_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

        print("Creating or loading collection")
        # chroma_collection = chroma_client.get_or_create_collection(metadata={"hnsw:space": "cosine"},
        #                                                            name="financial_reports",
        #                                                            embedding_function=huggingface_ef)
        chroma_collection = chroma_client.get_or_create_collection(metadata={"hnsw:space": "cosine"},
                                                                   name="financial_reports")
        
        print("Collection item count",chroma_collection.count())

        if chroma_collection.count()==0:
            print("I am inside")
            for index, doc in enumerate(self.split_texts()):
                documents = []
                metadatas = []
                ids = []
                emb=[]
                documents.append(doc.page_content)
                query_embedding = emb_model.encode(doc.page_content)
                metadata = doc.metadata.copy()
                metadata['filingDate'] = metadata['filingDate'].strftime('%Y-%m-%d')
                # print(type(query_embedding))
                # print(query_embedding)
                metadatas.append(metadata)
                emb.append(query_embedding.tolist())
                unique_id = f"{metadata['docID']}_idx{index}"

                ids.append(unique_id)
                print("Embedding Added: ",index)
                chroma_collection.add(documents=documents, metadatas=metadatas, ids=ids,embeddings=emb)
        return chroma_client
