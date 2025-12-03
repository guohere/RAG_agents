import torch
import pandas as pd
import wikipedia
import re
import gc
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer # <--- NEW
from tqdm import tqdm

# Import your helper module (Make sure fewshot.py is in the same folder)
from fewshot import FewShotEngine 

# Suppress annoying Wikipedia warnings
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 100
OUTPUT_FILE = "1000Q_v2_fewshot.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_ID = "all-MiniLM-L6-v2"

if not torch.cuda.is_available():
    print("❌ No GPU found!")
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

# --- NEW: Load Embedder on CPU to save VRAM ---
print(f"Loading {EMBEDDING_ID} on CPU...")
embedder = SentenceTransformer(EMBEDDING_ID, device="cpu")

# --- NEW: Initialize Few-Shot Engine ---
print("Loading Dataset & Building Index...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
# We index 1000 training examples. You can increase this if you have RAM.
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
# 3. HELPERS
# ==========================================
def classify_question(question):
    prompt = [
        {"role": "system", "content": "Classify as: Diagnosis, Treatment, or Mechanism. Output ONE word."},
        {"role": "user", "content": f"Question: {question}\nCategory:"}
    ]
    return generate_text(prompt, max_new_tokens=10).strip().replace(".", "")

def generate_search_query(question, category):
    prompt = [
        {"role": "system", "content": "Extract the specific disease, drug, or medical concept to search on Wikipedia."},
        {"role": "user", "content": f"Question: {question}\n\nSearch Term:"}
    ]
    return generate_text(prompt, max_new_tokens=30).strip()

def get_best_wikipedia_context(search_term, options, question_text):
    candidates = []
    try:
        results = wikipedia.search(search_term, results=1)
        if results: candidates.append(results[0])
    except: pass

    for opt in options.values():
        candidates.append(opt)
    
    candidates = list(set(candidates))
    final_context = []
    
    # Heuristic: Prioritize Option pages as they are likely the answer
    for i, page_title in enumerate(candidates):
        if i >= 3: break 
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            content = page.summary
            
            # Simple content filter
            if "treatment" in question_text.lower():
                section = page.section("Treatment") or page.section("Medical uses")
                if section: content = section
            elif "mechanism" in question_text.lower():
                 section = page.section("Mechanism of action")
                 if section: content = section

            final_context.append(f"[{page.title}]: {content[:800]}") 
        except:
            continue
            
    return "\n\n".join(final_context)

# ==========================================
# 4. SOLVER (Now with Examples!)
# ==========================================
def solve_question_cot(question, options, evidence, examples):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    # We put the examples FIRST so the model adopts the style immediately
    messages = [
        {"role": "system", "content": "You are a medical expert taking the USMLE. Follow the format of the examples provided."},
        {"role": "user", "content": f"""
Here are some similar solved examples:
{examples}

Now, solve this new question using the Context below:

Background Information:
{evidence}

Question: 
{question}

Options:
{options_fmt}

Instructions:
1. Evaluate each option against the symptoms/context.
2. Eliminate incorrect options.
3. Conclude with the correct option.

Format:
Reasoning: [Your step-by-step logic]
Answer: [Only the Letter A, B, C, or D]
"""}
    ]
    
    return generate_text(messages, max_new_tokens=512)

def parse_final_answer(text):
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    matches = re.findall(r"\b([A-D])\b", text)
    if matches: return matches[-1].upper()
    
    return "C" 

# ==========================================
# 5. LOOP
# ==========================================
subset = dataset['test'].select(range(NUM_SAMPLES_TO_TEST))
results = []

print(f"🚀 Starting Pipeline with Few-Shot...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        question = sample['question']
        options = sample['options']
        correct_key = sample['answer_idx']
        
        # 1. Classify
        cat = classify_question(question)
        
        # 2. Retrieve Examples (The "Memory")
        # Finds 2 questions from training set that look like this one
        examples = fs_engine.retrieve(question, k=2)
        
        # 3. Retrieve Context (The "Search")
        query = generate_search_query(question, cat)
        context = get_best_wikipedia_context(query, options, question)
        
        # 4. Solve
        response = solve_question_cot(question, options, context, examples)
        pred = parse_final_answer(response)

        results.append({
            "id": i,
            "category": cat,
            "is_correct": (pred == correct_key),
            "prediction": pred,
            "correct_answer": correct_key,
            "reasoning": response,
            "examples_used": len(examples)
        })
        
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error: {e}")
        continue

df = pd.DataFrame(results)
print(f"Final Accuracy: {df['is_correct'].mean()*100:.2f}%")
df.to_csv(OUTPUT_FILE, index=False)