from ast import literal_eval
import pandas as pd
from datasets import load_dataset
import os
from transformers import pipeline
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self):
        print("data pro loaded")
        self.data = load_dataset("JanosAudran/financial-reports-sec", "small_full", split="train",trust_remote_code=True)
        self.data = self.data.to_pandas()
        self.classified_data_file = "classified_data.csv"
    
    def filter_data(self):
        self.data['filingDate'] = pd.to_datetime(self.data['filingDate'], errors='coerce')
        self.filtered_data = self.data[(self.data['cik'] == '0000001750') & (self.data['filingDate'].dt.year.isin([2019]))].copy()

    def classify_batch(self,batch):
        candidate_labels = [
        "revenue and profit", 
        "expenses and costs", 
        "investments and funding", 
        "market and competition", 
        "risks and compliance", 
        "operations and strategy",
        "company information",
        "general statements (irrelevant)"
        ]
        classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)  # Use CPU
        results = classifier(batch, candidate_labels)
        return [result['labels'][0] for result in results]
    
    def classify_data(self):
        # Check if the classified data is already saved
        if os.path.exists(self.classified_data_file):
            print("Loading previously classified data...")
            self.filtered_data = pd.read_csv(self.classified_data_file)
        else:
            print("Classifying data...")
            self.filter_data()
            batch_size = 16
            classifications = []
            for i in tqdm(range(0, len(self.filtered_data), batch_size)):
                batch_sentences = self.filtered_data['sentence'].iloc[i:i + batch_size].tolist()
                classifications.extend(self.classify_batch(batch_sentences))
            self.filtered_data['classification'] = classifications
            # Save the classified data to a file
            self.filtered_data.to_csv(self.classified_data_file, index=False)
        
        #self.relevant_sentences = self.filtered_data[self.filtered_data['classification'] != "general statements (irrelevant)"]
        self.relevant_sentences = self.filtered_data[self.filtered_data['classification'] == "investments and funding"]
        label_counts = self.filtered_data['classification'].value_counts()
        print(label_counts)
        

    def parse_sentiment(self, label_dict):
        if isinstance(label_dict, str):
            label_dict = literal_eval(label_dict)
        return {
            '1_day_sentiment': 'positive' if label_dict['1d'] == 1 else 'negative',
            '5_day_sentiment': 'positive' if label_dict['5d'] == 1 else 'negative',
            '30_day_sentiment': 'positive' if label_dict['30d'] == 1 else 'negative',
        }

    def preprocess_data(self):
        self.classify_data()
        # self.data['filingDate'] = pd.to_datetime(self.data['filingDate'], errors='coerce')
        # self.data = self.data[(self.data['cik'] == '0000001750') & (self.data['filingDate'].dt.year.isin([2019, 2020]))]
        expanded_sentiments = self.relevant_sentences['labels'].apply(self.parse_sentiment).apply(pd.Series)
        documents_df = pd.concat([self.relevant_sentences.drop(columns=['labels']), expanded_sentiments], axis=1)
        return documents_df.groupby('docID').agg({
            'sentence': ' '.join,
            'filingDate': 'first',
            'stateOfIncorporation': 'first',
            '1_day_sentiment': 'first',
            '5_day_sentiment': 'first',
            '30_day_sentiment': 'first'
        }).reset_index()
    
