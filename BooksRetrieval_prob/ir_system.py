#!/usr/bin/env python3

import os
import re
import math
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class IRSystem:
    def __init__(self, documents_path, covers_path):
        self.documents_path = documents_path
        self.covers_path = covers_path
        self.documents = {}
        self.inv_index = {}
        self.doc_lengths = {}
        self.idf = {}
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.average_doc_length = 0

    def read_documents(self):
        for filename in os.listdir(self.documents_path):
            if filename.endswith(".txt"):
                with open(os.path.join(self.documents_path, filename), 'r', encoding='utf-8') as file:
                    self.documents[filename] = file.read()

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def preprocess(self, text):
        tokens = self.tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopwords]
        return tokens

    def build_inverted_index(self):
        total_length = 0
        for doc_id, content in self.documents.items():
            terms = self.preprocess(content)
            self.doc_lengths[doc_id] = len(terms)
            total_length += len(terms)
            term_freq = Counter(terms)
            
            for term, freq in term_freq.items():
                if term not in self.inv_index:
                    self.inv_index[term] = {}
                self.inv_index[term][doc_id] = freq
        
        self.average_doc_length = total_length / len(self.documents)

    def compute_idf(self):
        N = len(self.documents)
        for term, doc_dict in self.inv_index.items():
            self.idf[term] = math.log((N - len(doc_dict) + 0.5) / (len(doc_dict) + 0.5) + 1)

    def compute_bm25_score(self, term, doc_id, k1=1.5, b=0.75):
        if term not in self.inv_index or doc_id not in self.inv_index[term]:
            return 0
        tf = self.inv_index[term][doc_id]
        doc_length = self.doc_lengths[doc_id]
        numerator = self.idf[term] * tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_length / self.average_doc_length)
        return numerator / denominator

    def search(self, query):
        query_terms = self.preprocess(query)
        scores = {doc_id: 0 for doc_id in self.documents}
        
        for term in query_terms:
            for doc_id in self.documents:
                scores[doc_id] += self.compute_bm25_score(term, doc_id)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def get_book_cover(self, doc_id):
        book_name = doc_id.replace('.txt', '')
        for ext in ['.jpg', '.jpeg', '.png', '.gif']:
            cover_path = os.path.join(self.covers_path, f"{book_name}{ext}")
            if os.path.exists(cover_path):
                return cover_path
        return os.path.join(self.covers_path, 'default_cover.jpg')

def evaluate_system(ir_system, queries, query_relevance):
    results = {}
    for query in queries:
        print(f"\nQuery: {query}")
        search_results = ir_system.search(query)[:5]
        print("Top 5 results:")
        for doc_id, score in search_results:
            print(f"{doc_id}: {score:.4f}")
        
        relevant_docs = query_relevance.get(query, [search_results[0][0]] if search_results else [])
        
        precision = precision_at_k(search_results, relevant_docs)
        recall = recall_at_k(search_results, relevant_docs)
        ap = average_precision(search_results, relevant_docs)
        ndcg = ndcg_at_k(search_results, relevant_docs)
        
        print(f"Precision@5: {precision:.4f}")
        print(f"Recall@5: {recall:.4f}")
        print(f"Average Precision: {ap:.4f}")
        print(f"NDCG@5: {ndcg:.4f}")
        
        results[query] = {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'ndcg': ndcg
        }
    
    map_score = mean_average_precision(results)
    print(f"\nMean Average Precision: {map_score:.4f}")
    
    return results, map_score

def precision_at_k(results, relevant_docs, k=5):
    relevant_retrieved = sum(1 for doc_id, _ in results[:k] if doc_id in relevant_docs)
    return relevant_retrieved / k if k > 0 else 0

def recall_at_k(results, relevant_docs, k=5):
    relevant_retrieved = sum(1 for doc_id, _ in results[:k] if doc_id in relevant_docs)
    return relevant_retrieved / len(relevant_docs) if relevant_docs else 0

def average_precision(results, relevant_docs):
    precision_sum = 0
    relevant_count = 0
    
    for i, (doc_id, _) in enumerate(results):
        if doc_id in relevant_docs:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / len(relevant_docs) if relevant_docs else 0

def ndcg_at_k(results, relevant_docs, k=5):
    dcg = sum(1 / math.log2(i + 2) for i, (doc_id, _) in enumerate(results[:k]) if doc_id in relevant_docs)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_docs), k)))
    return dcg / idcg if idcg > 0 else 0

def mean_average_precision(results):
    return sum(result['ap'] for result in results.values()) / len(results)

def main():
    documents_path = "/home/iv_jay/Desktop/BooksRetrieval_prob/documents"
    covers_path = "/home/iv_jay/Desktop/BooksRetrieval_prob/covers"
    
    print(f"Documents path: {documents_path}")
    print(f"Covers path: {covers_path}")
    print(f"Documents directory exists: {os.path.exists(documents_path)}")
    print(f"Covers directory exists: {os.path.exists(covers_path)}")
    
    if os.path.exists(documents_path):
        print(f"Number of documents: {len([f for f in os.listdir(documents_path) if f.endswith('.txt')])}")
    if os.path.exists(covers_path):
        print(f"Number of covers: {len(os.listdir(covers_path))}")
    
    ir_system = IRSystem(documents_path, covers_path)
    ir_system.read_documents()
    ir_system.build_inverted_index()
    ir_system.compute_idf()

    queries = [
        "Innovation", "Mysteries", "Haunted", "Ethics", "Wizardry",
        "Marriage", "Prophecy", "Survival", "Entrepreneur", "Agriculture"
    ]

    # For demonstration, we'll use a simple relevance judgement
    # In a real scenario, this should be based on human annotations
    query_relevance = {query: [ir_system.search(query)[0][0]] for query in queries}

    results, map_score = evaluate_system(ir_system, queries, query_relevance)

if __name__ == "__main__":
    main()