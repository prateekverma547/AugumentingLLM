from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.graphs.neo4j_graph import Neo4jGraph
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from os.path import join, dirname

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

HOST = os.environ.get('HOST')
PORT = os.environ.get('PORT')
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
NEO4J_URI=os.environ.get('NEO4J_URI')
NEO4J_USERNAME=os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD=os.environ.get('NEO4J_PASSWORD')

class KnowledgeGraphQuery:
    def __init__(self, uri, username, password, openai_api_key):
        # Initialize Neo4j graph connection and vector index
        self.graph = Neo4jGraph(url=uri, username=username, password=password)
        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(openai_api_key=openai_api_key),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        # Initialize the LLM model
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key)


    
    def entity_extraction(self, text: str) -> List[str]:
      class Entities(BaseModel):
          names: List[str] = Field(..., description="All the person, organization, or business entities in the text")

      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are extracting organization and person entities from the text."),
          ("human", "Use the given format to extract information from the following input: {question}")
      ])
      entity_chain = prompt | self.llm.with_structured_output(Entities)
      return entity_chain.invoke({"question": text}).names
    


    def query_structured_data(self, entity_text):
        entities = self.entity_extraction(entity_text)
        print("line28",entities)
        results = []
        for entity in entities:
            query = """
            MATCH (e)-[r]->(related)
            WHERE e.id CONTAINS $entity
            RETURN e.id + ' - ' + type(r) + ' -> ' + related.id AS output
            LIMIT 50
            """
            result = self.graph.query(query, {"entity": entity})
            results.extend([res['output'] for res in result])
        return "\n".join(results)
    

    
    def search_unstructured_data(self, question):
        # Search for unstructured data related to the question
        documents = self.vector_index.similarity_search(question)
        return "\n\n".join([doc.page_content for doc in documents])
    
    


    # def generate_answer(self, question, context):
    #     # Generate an answer using the LLM model based on the provided context
    #     prompt = f"Given the following context, answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    #     return self.llm(prompt)
    
    def generate_answer(self, question, structured_data, unstructured_data):
        # Combine structured and unstructured data
        context = f"Structured data:\n{structured_data}\n\nUnstructured data:\n{unstructured_data}"
        
        # Generate an answer using the LLM model based on the provided context
        prompt = f"Given the following context, answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm(prompt)
    

    # def run(self, question):
    #     unstructured_data = self.search_unstructured_data(question)
    #     context = f"Unstructured data:\n{unstructured_data}"
    #     return self.generate_answer(question, context)
    
    def run(self, question):
        # Dynamically extract entities from the question
        entities = self.extract_entities(question)
        
        # Query structured data from the knowledge graph
        structured_data = self.query_structured_data(entities)
        
        # Search unstructured data using vector search
        unstructured_data = self.search_unstructured_data(question)
        
        # Generate the final answer using both structured and unstructured data
        return self.generate_answer(question, structured_data, unstructured_data)
    

# Example usage:
if __name__ == "__main__":
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    openai_api_key = os.environ['OPENAI_API_KEY']

    kg_query = KnowledgeGraphQuery(uri, username, password, openai_api_key)
    question = "Who are the principal customers for the company's products and services in the Aviation Services segment?"
    answer = kg_query.run(question)
    print("Answer:", answer.content)