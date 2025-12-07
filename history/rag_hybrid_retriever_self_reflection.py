import torch
import pandas as pd
import wikipedia
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi  # For Keyword Search
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 300
OUTPUT_FILE = "medqa_ultimate_results.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_ID = "all-MiniLM-L6-v2" # Tiny & Fast
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if not torch.cuda.is_available():
    print("❌ No GPU found!")
    exit()

# ==========================================
# 2. LOAD MODELS
# ==========================================
print(f"Loading {MODEL_ID} (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.1)

print("Loading Retrievers (CPU optimized)...")
embedder = SentenceTransformer(EMBEDDING_ID, device="cpu")
reranker = CrossEncoder(RERANKER_ID, device="cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. HYBRID RETRIEVER (For Few-Shot Examples)
# ==========================================
class HybridFewShotRetriever:
    """
    Combines BM25 (Keyword) + Embeddings (Semantic) to find the best examples.
    """
    def __init__(self, train_dataset, n_index=500):
        print(f"Indexing {n_index} training examples for Hybrid Search...")
        self.data = train_dataset.select(range(n_index))
        self.corpus_text = [item['question'] for item in self.data]
        
        # 1. Sparse Index (BM25)
        tokenized_corpus = [doc.split(" ") for doc in self.corpus_text]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Dense Index (Embeddings)
        self.corpus_embeddings = embedder.encode(self.corpus_text, convert_to_tensor=True)

    def search(self, query, k=2, alpha=0.5):
        # A. Semantic Search
        query_emb = embedder.encode(query, convert_to_tensor=True)
        sem_scores = embedder.similarity(query_emb, self.corpus_embeddings)[0]
        
        # B. Keyword Search
        tokenized_query = query.split(" ")
        bm25_scores = torch.tensor(self.bm25.get_scores(tokenized_query))
        
        # Normalize scores (0 to 1) so we can add them
        sem_scores = (sem_scores - sem_scores.min()) / (sem_scores.max() - sem_scores.min() + 1e-9)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
        
        # C. Hybrid Fusion
        # alpha=0.5 means equal weight to keywords and meaning
        hybrid_scores = (alpha * sem_scores) + ((1-alpha) * bm25_scores)
        
        # Get Top K
        top_k_indices = torch.topk(hybrid_scores, k=k).indices
        
        examples_str = ""
        for idx in top_k_indices:
            ex = self.data[int(idx)]
            opts = "\n".join([f"{k}: {v}" for k, v in ex['options'].items()])
            examples_str += f"\n[Example]\nQ: {ex['question']}\nOptions:\n{opts}\nCorrect Answer: {ex['answer_idx']}\n"
            
        return examples_str

# ==========================================
# 4. EXTERNAL RETRIEVAL & RE-RANKING
# ==========================================
def smart_wikipedia_search(question, options):
    # 1. Generate Search Query (Query Expansion)
    sys_msg = [{"role": "system", "content": "Extract the main medical condition from the text."},
               {"role": "user", "content": question}]
    search_term = pipe(sys_msg, max_new_tokens=20)[0]['generated_text'][-1]['content']
    
    candidates = []
    # Search extracted term
    try:
        page = wikipedia.page(search_term)
        candidates.append(page.summary)
    except: pass
    
    # Search Options
    for opt in options.values():
        try:
            candidates.append(wikipedia.summary(opt, sentences=3))
        except: pass
        
    if not candidates: return "No info found."
    
    # 2. Re-Rank (Keep Top 3)
    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # Return top 3 chunks
    return "\n\n".join([doc for doc, score in ranked[:3]])

# ==========================================
# 5. SELF-RAG: THE "CRITIC" LOOP
# ==========================================
def self_reflective_solve(question, options, evidence, examples):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    # --- PHASE 1: GENERATE ---
    prompt = [
        {"role": "system", "content": "You are a Medical Expert. Use the Context and Examples to answer. Think step-by-step."},
        {"role": "user", "content": f"""
Examples:
{examples}

Context:
{evidence}

Question:
{question}

Options:
{options_fmt}

Instructions:
1. Reason through the symptoms.
2. Select the answer.
3. Finally, print 'Answer: [Option]'."""}
    ]
    
    response = pipe(prompt)[0]['generated_text'][-1]['content']
    
    # --- PHASE 2: SELF-CORRECTION (The "Self-RAG" part) ---
    # We ask the model to verify its own answer against the evidence
    critique_prompt = [
        {"role": "system", "content": "You are a strict Evaluator. Check if the reasoning follows the provided Context."},
        {"role": "user", "content": f"""
Context:
{evidence}

Generated Reasoning:
{response}

Does the reasoning logically follow the Context? Answer YES or NO. If NO, explain why."""}
    ]
    
    critique = pipe(critique_prompt, max_new_tokens=50)[0]['generated_text'][-1]['content']
    
    # If the model doubts itself, we append a warning to the reasoning (Simple Self-Correction)
    # In a full system, we would loop back and re-search.
    final_output = response
    if "NO" in critique.upper():
        final_output += "\n[Self-Correction Note: The model detected potential inconsistency with the evidence provided.]"
        
    return final_output

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
print("Loading Datasets...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
fs_retriever = HybridFewShotRetriever(dataset['train'], n_index=200) # Index 200 samples for speed
test_subset = dataset['test'].select(range(NUM_SAMPLES_TO_TEST))

results = []
print(f"\n🚀 Starting Hybrid Self-RAG Evaluation...")

for i, sample in tqdm(enumerate(test_subset), total=NUM_SAMPLES_TO_TEST):
    question = sample['question']
    options = sample['options']
    
    # 1. Retrieve Examples (Hybrid)
    examples = fs_retriever.search(question, k=2)
    
    # 2. Retrieve Evidence (External + ReRank)
    evidence = smart_wikipedia_search(question, options)
    
    # 3. Solve with Self-Reflection
    raw_response = self_reflective_solve(question, options, evidence, examples)
    
    # 4. Parse
    match = re.search(r"Answer:\s*([A-D])", raw_response, re.IGNORECASE)
    pred = match.group(1).upper() if match else "Unknown"
    is_correct = (pred == sample['answer_idx'])
    
    results.append({
        "id": i,
        "is_correct": is_correct,
        "prediction": pred,
        "correct": sample['answer_idx'],
        "self_rag_output": raw_response[:200] + "..."
    })

df = pd.DataFrame(results)
print(f"\nAccuracy: {df['is_correct'].mean()*100:.2f}%")
df.to_csv(OUTPUT_FILE, index=False)