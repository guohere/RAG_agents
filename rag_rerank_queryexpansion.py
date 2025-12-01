import torch
import pandas as pd
import wikipedia
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import CrossEncoder # <-- NEW: For Re-Ranking
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 300
OUTPUT_FILE = "medqa_rerank_queryexpansion_results.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Tiny, fast, high-performance re-ranker

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

print(f"Loading Re-Ranker {RERANKER_ID}...")
# This runs on GPU automatically if available, but fits easily (80MB)
reranker = CrossEncoder(RERANKER_ID)

# ==========================================
# 3. HELPER: SMART QUERY EXPANSION (NEW)
# ==========================================
def generate_search_query(question):
    """
    Uses the LLM to extract the core medical concept from the verbose question.
    Example: "45yo male with chest pain..." -> "Myocardial Infarction symptoms"
    """
    messages = [
        {"role": "system", "content": "You are a medical search assistant. Extract the single most important medical condition or symptom from the text to search for on Wikipedia."},
        {"role": "user", "content": f"Question:\n{question}\n\nSearch Query:"}
    ]
    output = pipe(messages, max_new_tokens=20)
    query = output[0]['generated_text'][-1]['content'].strip()
    return query

# ==========================================
# 4. HELPER: RE-RANKING (NEW)
# ==========================================
def rerank_evidence(question, raw_snippets, top_k=3):
    """
    Takes a list of messy Wikipedia snippets and scores them against the question.
    Returns only the top_k most relevant snippets.
    """
    if not raw_snippets:
        return []
    
    # Prepare pairs for the Cross-Encoder: (Question, Evidence)
    pairs = [[question, snippet] for snippet in raw_snippets]
    
    # Predict scores
    scores = reranker.predict(pairs)
    
    # Combine snippets with scores
    scored_snippets = list(zip(raw_snippets, scores))
    
    # Sort by score (descending) and take top_k
    scored_snippets.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the text of the top_k
    return [item[0] for item in scored_snippets[:top_k]]

# ==========================================
# 5. RETRIEVAL STRATEGIES (Updated)
# ==========================================
def retrieve_and_rerank(question, options, category):
    """
    1. Expand Query (Diagnosis extraction).
    2. Search Wikipedia (Broad Search).
    3. Re-Rank (Filter Noise).
    """
    candidates = []
    
    # --- A. EXPANSION SEARCH ---
    # Extract the core condition from the vignette (e.g., "Appendicitis")
    core_concept = generate_search_query(question)
    
    try:
        page = wikipedia.page(core_concept)
        candidates.append(f"Concept: {core_concept}\nSummary: {page.summary}")
    except:
        pass # Optimization failed, fall back to options

    # --- B. OPTION SEARCH ---
    # We still search the options because the answer must be one of them
    for opt in options.values():
        try:
            page = wikipedia.page(opt)
            # Fetch summary AND specific sections to ensure coverage
            candidates.append(f"Option: {opt}\nSummary: {page.summary}")
            
            # Fetch deeper sections if available
            section = page.section("Medical uses") or page.section("Signs and symptoms")
            if section:
                candidates.append(f"Option: {opt}\nDetails: {section[:500]}")
        except:
            continue
            
    # --- C. RE-RANKING ---
    # We might have 10+ snippets now. Filter them down to the best 3.
    # This prevents "context stuffing" where we confuse the model with too much text.
    best_snippets = rerank_evidence(question, candidates, top_k=3)
    
    return "\n\n".join(best_snippets)

# ==========================================
# 6. ROUTER & SOLVER
# ==========================================
def classify_question(question):
    prompt = [
        {"role": "system", "content": "Classify the medical question into: 'Diagnosis', 'Treatment', or 'Mechanism'. Output ONLY the word."},
        {"role": "user", "content": f"Question: {question}\n\nCategory:"}
    ]
    output = pipe(prompt, max_new_tokens=10)
    return output[0]['generated_text'][-1]['content'].strip()

def solve_question_cot(question, options, evidence):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    messages = [
        {"role": "system", "content": "You are a Medical Expert. Answer the question using the provided context."},
        {"role": "user", "content": f"""
Context:
{evidence}

Question:
{question}

Options:
{options_fmt}

Instructions:
1. Analyze the patient's symptoms against the Context.
2. Rule out options that contradict the Context.
3. Select the best matching option.

Format:
Reasoning: [Logic]
Answer: [Option Letter]"""}
    ]
    
    output = pipe(messages)
    return output[0]['generated_text'][-1]['content']

def parse_final_answer(text):
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    match = re.search(r"\b([A-D])\s*$", text)
    if match: return match.group(1).upper()
    return "Unknown"

# ==========================================
# 7. MAIN LOOP
# ==========================================
print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
subset = dataset.select(range(NUM_SAMPLES_TO_TEST))

results = []
print(f"\n🚀 Starting Advanced RAG Evaluation...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    question = sample['question']
    options = sample['options']
    correct_key = sample['answer_idx']
    
    # 1. Route
    category = classify_question(question)
    
    # 2. Advanced Retrieval (Expand -> Search -> Re-Rank)
    context = retrieve_and_rerank(question, options, category)
    if not context: context = "No relevant evidence found."

    # 3. Solve
    raw_response = solve_question_cot(question, options, context)
    
    # 4. Parse
    pred = parse_final_answer(raw_response)
    
    results.append({
        "id": i,
        "category": category,
        "is_correct": pred == correct_key,
        "prediction": pred,
        "correct_answer": correct_key,
        "evidence_used": context[:200] + "..." # Log this to verify Re-Ranking worked!
    })

df = pd.DataFrame(results)
print("\n=== RESULTS ===")
print(f"Accuracy: {df['is_correct'].mean()*100:.2f}%")
df.to_csv(OUTPUT_FILE, index=False)