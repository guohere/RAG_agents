import torch
import pandas as pd
import wikipedia
import re
import gc # <--- NEW: For memory cleanup
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 300
OUTPUT_FILE = "medqa_rerank_lowram_results.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
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

# OPTIMIZATION 1: Reduced max_tokens to 256 to save KV Cache memory
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.1)

print(f"Loading Re-Ranker {RERANKER_ID} on CPU...")
# OPTIMIZATION 2: Force Re-Ranker to CPU. Saves ~500MB VRAM.
reranker = CrossEncoder(RERANKER_ID, device="cpu")

# ==========================================
# 3. HELPERS
# ==========================================
def generate_search_query(question):
    messages = [
        {"role": "system", "content": "Extract the core medical condition or symptom from the question."},
        {"role": "user", "content": f"Question:\n{question}\n\nSearch Query:"}
    ]
    output = pipe(messages, max_new_tokens=20)
    query = output[0]['generated_text'][-1]['content'].strip()
    return query

def rerank_evidence(question, raw_snippets, top_k=3):
    if not raw_snippets: return []
    
    # This now runs on CPU, which is fine for small batches
    pairs = [[question, snippet] for snippet in raw_snippets]
    scores = reranker.predict(pairs)
    scored_snippets = list(zip(raw_snippets, scores))
    scored_snippets.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored_snippets[:top_k]]

def retrieve_and_rerank(question, options, category):
    candidates = []
    
    # Expansion Search
    core_concept = generate_search_query(question)
    try:
        page = wikipedia.page(core_concept)
        # OPTIMIZATION 3: Limit characters to 300 per snippet
        candidates.append(f"Concept: {core_concept}\nSummary: {page.summary[:300]}")
    except: pass

    # Option Search
    for opt in options.values():
        try:
            page = wikipedia.page(opt)
            content = page.section("Medical uses") or page.section("Signs and symptoms")
            if not content: content = page.summary
            candidates.append(f"Option: {opt}\nDetails: {content[:300]}")
        except: continue
            
    best_snippets = rerank_evidence(question, candidates, top_k=3)
    return "\n\n".join(best_snippets)

def classify_question(question):
    prompt = [
        {"role": "system", "content": "Classify: 'Diagnosis', 'Treatment', or 'Mechanism'. Output ONLY the word."},
        {"role": "user", "content": f"Question: {question}\n\nCategory:"}
    ]
    output = pipe(prompt, max_new_tokens=10)
    return output[0]['generated_text'][-1]['content'].strip()

def solve_question_cot(question, options, evidence):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    messages = [
        {"role": "system", "content": "Medical Expert. Answer using the context."},
        {"role": "user", "content": f"Context:\n{evidence}\n\nQuestion:\n{question}\n\nOptions:\n{options_fmt}\n\nReason step-by-step then 'Answer: [Option]'."}
    ]
    output = pipe(messages)
    return output[0]['generated_text'][-1]['content']

def parse_final_answer(text):
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    return "Unknown"

# ==========================================
# 4. MAIN LOOP WITH MEMORY MANAGEMENT
# ==========================================
print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
subset = dataset.select(range(NUM_SAMPLES_TO_TEST))

results = []
print(f"\n🚀 Starting Low-Memory Evaluation...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    try:
        question = sample['question']
        options = sample['options']
        correct_key = sample['answer_idx']
        
        category = classify_question(question)
        context = retrieve_and_rerank(question, options, category)
        if not context: context = "No relevant evidence found."

        raw_response = solve_question_cot(question, options, context)
        pred = parse_final_answer(raw_response)
        
        results.append({
            "id": i,
            "category": category,
            "is_correct": pred == correct_key,
            "prediction": pred,
            "correct_answer": correct_key
        })

        # OPTIMIZATION 4: Aggressive Memory Cleanup
        # This frees up the VRAM used by the previous question's context
        del context, raw_response, category
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error on sample {i}: {e}")
        # Even if error, clean memory to prevent chain reaction
        torch.cuda.empty_cache()
        continue

df = pd.DataFrame(results)
print("\n=== RESULTS ===")
print(f"Accuracy: {df['is_correct'].mean()*100:.2f}%")
df.to_csv(OUTPUT_FILE, index=False)