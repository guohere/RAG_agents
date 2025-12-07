import torch
import numpy as np
from rank_bm25 import BM25Okapi
import config

class FewShotEngine:
    def __init__(self, train_dataset, embedder, n_index=5000):
        """
        Indexes the training data so we can find similar examples.
        n_index: How many training samples to load (Higher = Better matches, but slower init)
        """
        print(f"Indexing {n_index} training examples for Few-Shot...")
        
        # Select the first N examples from the training split
        self.data = train_dataset.select(range(n_index))
        self.embedder = embedder
        
        # Prepare Data for Indexing
        self.corpus_text = [item['question'] for item in self.data]
        
        # 1. Dense Index (Semantic Search - Vectors)
        # This runs on CPU (via models.py settings) to save GPU VRAM
        print("Building Vector Index...")
        self.embeddings = self.embedder.encode(self.corpus_text, convert_to_tensor=True)
        
        # 2. Sparse Index (Keyword Search - BM25)
        print("Building Keyword Index...")
        tokenized_corpus = [doc.split(" ") for doc in self.corpus_text]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query, k=2, alpha=0.5):
        """
        Finds k similar questions using Hybrid Search.
        alpha: Weighting factor (0.5 = 50% Vector, 50% Keyword)
        """
        # --- A. Semantic Search ---
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        # Calculate Cosine Similarity
        sem_scores = self.embedder.similarity(query_emb, self.embeddings)[0]
        
        # --- B. Keyword Search ---
        tokenized_query = query.split(" ")
        bm25_scores = torch.tensor(self.bm25.get_scores(tokenized_query))
        
        # --- C. Normalization & Fusion ---
        # We must normalize scores to 0-1 range to combine them fairly
        sem_norm = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-9)
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
        
        # Combine
        hybrid_scores = (alpha * sem_norm) + ((1-alpha) * bm25_norm)
        
        # Get Top K indices
        top_indices = torch.topk(hybrid_scores, k=k).indices
        
        # --- D. Format Output ---
        output = ""
        for idx in top_indices:
            ex = self.data[int(idx)]
            opts = "\n".join([f"{key}: {val}" for key, val in ex['options'].items()])
            
            # This string teaches the model the EXACT format we want
            output += f"""
            [Example Question]
            {ex['question']}
            Options:
            {opts}
            Answer: {ex['answer_idx']}
            """
            
        return output