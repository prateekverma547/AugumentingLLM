from langchain_community.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_preprocessor import DataPreprocessor
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from query import QueryEngine
from langchain.evaluation.qa import QAEvalChain
import csv
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from knowledge_graph_query import KnowledgeGraphQuery


preprocess_data = DataPreprocessor()
data = preprocess_data.preprocess_data()
loader = DataFrameLoader(data, page_content_column="sentence")
documents = loader.load()
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

all_doc=[]


llm = ChatOpenAI(
        model_name='gpt-3.5-turbo-16k',
        )

def split_texts():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=430,
        chunk_overlap=20,
        length_function=len,
    )
    split_doc = text_splitter.split_documents(documents)
    print("0")
    return split_doc


# Testing data Q and A generation on x documents
def test_data_generation(x):
    all_doc=split_texts()
    chain = QAGenerationChain.from_llm(llm)
    finaldoc=[]
    for i in all_doc[0:x]:
        finaldoc.append(i.page_content)
    # print(finaldoc)
    qa_data=chain.batch(finaldoc, config={"max_concurrency": 5})
    qa_dataset = [l['questions'][0] for l in qa_data]
    return qa_dataset

def genrating_predition_on_test(qa_dataset):

    # Loading chain which we are using for our Q&A BOT
    QE=QueryEngine()
    chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=QE.get_retriver(),
            chain_type_kwargs={"prompt": QE.prompt()},
            input_key="question"
        )
    predictions = chain.apply(qa_dataset)
    return predictions

def genrating_predition_on_test_neo4j(qa_dataset):
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    openai_api_key = os.environ['OPENAI_API_KEY']

    kg_query = KnowledgeGraphQuery(uri, username, password, openai_api_key)

    predictions = []
    for qa in qa_dataset:
        question = qa['question']
        answer = kg_query.run(question)
        predictions.append({
            'question': question,
            'answer': qa['answer'],  # Original answer for comparison later
            'result': answer.content  # Predicted answer from Neo4j
        })
    
    return predictions


def evaluation_of_prediction(qa_dataset,predictions):

    qaeval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = qaeval_chain.evaluate(qa_dataset, 
                                     predictions, 
                                     question_key="question",
                                     answer_key="answer",
                                     prediction_key="result")
    return graded_outputs



# Load a zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Function to determine complexity using zero-shot classification
def determine_complexity(question):
    labels = ["Simple", "Moderate", "Complex"]
    result = classifier(question, labels)
    return result["labels"][0] 

# Function to categorize questions using zero-shot classification
def categorize_question(question):
    labels = ["Contracts & Agreements", "Operations & Services", "General","Financial Performace"]
    result = classifier(question, labels)
    return result["labels"][0]  


# Function to compare generated answer with original answer and determine relevance
def determine_answer_relevance(original_answer, generated_answer):
    # Generate embeddings for both answers
    original_embedding = model.encode([original_answer])
    generated_embedding = model.encode([generated_answer])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(original_embedding, generated_embedding)[0][0]

    # Determine relevance based on similarity score thresholds
    if similarity_score > 0.8:
        return "High Relevance"
    elif similarity_score > 0.5:
        return "Medium Relevance"
    else:
        return "Low Relevance"

# Updated function to process results with additional columns
def proces_result(test_result, prediction):
    merged_dataset = []
    for item1, item2 in zip(prediction, test_result):
        question = item1['question']
        original_answer = item1['answer']
        generated_answer = item1['result']
        
        complexity = determine_complexity(question)
        category = categorize_question(question)
        relevance = determine_answer_relevance(original_answer, generated_answer)
        
        merged_item = {
            'Question': question,
            'Answer': original_answer,
            'Prediction': generated_answer,
            'Outcome': 1 if item2['results'] == 'CORRECT' else 0,
            'Complexity': complexity,
            'Category': category,
            'Answer Relevance': relevance
        }
        merged_dataset.append(merged_item)
    
    save_to_csv(merged_dataset, "testOutcome.csv")


def save_to_csv(data, filename):
    # Dynamically get all field names from the first item in the data list
    if not data:
        print("No data to save.")
        return

    # Get the keys of the first dictionary as the fieldnames
    fieldnames = data[0].keys()

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in data:
            writer.writerow(item)


def score_showcase(filename):
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        merged_dataset = list(reader)

        total_positive = sum(1 for item in merged_dataset if item['Outcome'] == '1')
        total_negative = sum(1 for item in merged_dataset if item['Outcome'] == '0')
        total_entries = len(merged_dataset)
        positive_percentage = (total_positive / total_entries) * 100
        negative_percentage = (total_negative / total_entries) * 100

        print("Total Positive Outcomes:", total_positive)
        print("Total Negative Outcomes:", total_negative)
        print("Positive Outcome Percentage: {:.2f}%".format(positive_percentage))
        print("Negative Outcome Percentage: {:.2f}%".format(negative_percentage))



###############################

#Lets test the model on 100 data set
filename = 'testOutcome.csv'
if os.path.exists(filename):
    score_showcase(filename)
else:
    print("File does not exist.")
    datasetcount=3
    qa_data=test_data_generation(datasetcount)
   
    #Predication for Vector DB
    # predicted_data=genrating_predition_on_test(qa_data)

    #Predication for Node4j DB
    predicted_data=genrating_predition_on_test_neo4j(qa_data)

   
    result=evaluation_of_prediction(qa_data,predicted_data)
   
    proces_result(result,predicted_data)
    # score_showcase(filename)
