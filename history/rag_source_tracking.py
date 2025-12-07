import torch
import pandas as pd
import wikipedia
import re
import gc
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import your helper module
from fewshot import FewShotEngine 

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 30  # Running full set?
OUTPUT_FILE = "med_30Q_source_tracked.csv" # New filename
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_ID = "pritamdeka/S-PubMedBert-MS-MARCO"

if not torch.cuda.is_available():
    print("No GPU found!")
    exit()

# ==========================================
# 2. LOAD MODELS
# ==========================================
print(f"Loading {MODEL_ID} (4-bit) on GPU...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model.eval()

print(f"Loading {EMBEDDING_ID} on CPU...")
embedder = SentenceTransformer(EMBEDDING_ID, device="cpu")

print("Loading Dataset & Building Index...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
fs_engine = FewShotEngine(dataset['train'], embedder, n_index=10000)

def generate_text(messages, max_new_tokens):
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.1, 
            do_sample=True,
            top_p=0.9
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ==========================================
# 3. IMPROVED HELPERS
# ==========================================

def classify_question(question):
    prompt = [
        {"role": "system", "content": "Classify as: Diagnosis, Treatment, or Mechanism. Output ONE word."},
        {"role": "user", "content": f"Question: {question}\nCategory:"}
    ]
    return generate_text(prompt, max_new_tokens=10).strip().replace(".", "")

# --- IMPROVED: Search Query Generator ---
# Why: Previous version forced "ONE word" which is bad for complex diseases.
# Now: asks for "Key Search Phrase" allowing 2-3 words.
def generate_search_query(question, category):
    prompt = [
        {"role": "system", "content": "You are a search engine optimizer. Extract the most distinct medical concept, disease name, or unique symptom cluster from the text. Output ONLY the search phrase."},
        {"role": "user", "content": f"Question: {question}\n\nSearch Query:"}
    ]
    # Increased tokens to allow "Renal Papillary Necrosis" (3 words) instead of just "Necrosis"
    return generate_text(prompt, max_new_tokens=20).strip().replace('"', '')

# --- IMPROVED: Context Retriever (Now tracks sources!) ---
def get_best_wikipedia_context(search_term, options, question_text):
    candidates = []
    
    # 1. Search the generated term
    try:
        results = wikipedia.search(search_term, results=1)
        if results: candidates.append(results[0])
    except: pass

    # 2. ALSO search the Option A/B/C/D terms (Critical for differential diagnosis)
    # If the main search fails, these often save the day.
    for opt in list(options.values())[:2]: # Limit to first 2 options to save time/noise
        candidates.append(opt)
    
    candidates = list(set(candidates))
    final_context = []
    source_tracker = [] # <--- NEW: List to track sources
    
    for i, page_title in enumerate(candidates):
        if i >= 3: break 
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            content = page.summary
            
            # Context Filtering
            if "treatment" in question_text.lower():
                section = page.section("Treatment") or page.section("Medical uses")
                if section: content = section
            elif "mechanism" in question_text.lower():
                 section = page.section("Mechanism of action")
                 if section: content = section

            # Append Text
            snippet = f"SOURCE [{page.title}]: {content[:600]}"
            final_context.append(snippet)
            
            # Append Tracker
            source_tracker.append(f"{page.title} ({page.url})")

        except:
            continue
            
    # Return BOTH the text block AND the list of sources
    full_text = "\n\n".join(final_context)
    source_str = " | ".join(source_tracker)
    
    return full_text, source_str

# ==========================================
# 4. SOLVER (Enhanced Prompt)
# ==========================================
def solve_question_cot(question, options, evidence, examples):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    messages = [
        {"role": "system", "content": "You are a medical expert. You must answer based on the provided Context if available."},
        {"role": "user", "content": f"""
Here are examples of how to think:
{examples}

---
CURRENT TASK:

Background Context (retrieved from Wikipedia):
{evidence}

Question: 
{question}

Options:
{options_fmt}

Instructions:
1. **Source Check**: First, check if the "Background Context" contains the answer. If it does, explicitly mention: "According to Source [X]..."
2. **Analysis**: Analyze the patient's symptoms.
3. **Evaluation**: Evaluate EACH option. If you reject an option, explain why.
4. **Conclusion**: Pick the best answer.

Format:
Reasoning: [Step-by-step logic, citing sources where possible]
Answer: [A, B, C, or D]
"""}
    ]
    
    return generate_text(messages, max_new_tokens=1024)

def parse_final_answer(text):
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    if matches: return matches[-1].upper()
    return "C" 

# ==========================================
# 5. MAIN LOOP
# ==========================================
subset = dataset['test'].select(range(NUM_SAMPLES_TO_TEST))
results = []

print(f"Starting Pipeline...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        question = sample['question']
        options = sample['options']
        correct_key = sample['answer_idx']
        
        # 1. Classify
        cat = classify_question(question)
        
        # 2. Retrieve Examples
        examples = fs_engine.retrieve(question, k=2)
        
        # 3. Retrieve Context (TRACKING ADDED)
        query = generate_search_query(question, cat)
        context_text, source_refs = get_best_wikipedia_context(query, options, question)
        
        # 4. Solve
        response = solve_question_cot(question, options, context_text, examples)
        pred = parse_final_answer(response)

        # 5. Save Data (New Columns Added)
        results.append({
            "id": i,
            "category": cat,
            "is_correct": (pred == correct_key),
            "prediction": pred,
            "correct_answer": correct_key,
            "search_query": query,          # <--- NEW
            "source_references": source_refs, # <--- NEW
            "reasoning": response,
            "examples_used": len(examples)
        })
        
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error at index {i}: {e}")
        continue

df = pd.DataFrame(results)
print(f"Final Accuracy: {df['is_correct'].mean()*100:.2f}%")
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results with source tracking to {OUTPUT_FILE}")