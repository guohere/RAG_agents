import torch
import pandas as pd
import wikipedia
import re
import gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 1000
OUTPUT_FILE = "1000Q_v2.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ==========================================
# 2. LOAD MODEL (4-bit)
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model.eval()

def generate_text(messages, max_new_tokens):
    """
    Standard generation wrapper.
    """
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=0.1, # Keep low for factual accuracy
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
    # 10 tokens is plenty for one word
    return generate_text(prompt, max_new_tokens=10).strip().replace(".", "")

def generate_search_query(question, category):
    prompt = [
        {"role": "system", "content": "Extract the specific disease, drug, or medical concept to search on Wikipedia."},
        {"role": "user", "content": f"Question: {question}\n\nSearch Term:"}
    ]
    # Increased to 30 to allow for long medical names
    return generate_text(prompt, max_new_tokens=30).strip()

def get_best_wikipedia_context(search_term, options, question_text):
    candidates = []
    try:
        results = wikipedia.search(search_term, results=1)
        if results: candidates.append(results[0])
    except: pass

    # Always add the options to the search candidates
    for opt in options.values():
        candidates.append(opt)
    
    candidates = list(set(candidates))
    final_context = []
    
    print(f"   -> Searching: {candidates[:3]}...") # Log what we are looking for

    # We limit to 3 pages to save memory
    for i, page_title in enumerate(candidates):
        if i >= 3: break 
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            
            # Intelligent Section Grabbing
            content = page.summary
            if "treatment" in question_text.lower():
                section = page.section("Treatment") or page.section("Management") or page.section("Medical uses")
                if section: content = section
            elif "mechanism" in question_text.lower():
                 section = page.section("Mechanism of action") or page.section("Pathophysiology")
                 if section: content = section

            # TRUNCATION IS KEY
            # We take 800 chars per page. 3 pages * 800 chars = ~2400 chars input.
            # This leaves plenty of room for the 512 token output.
            final_context.append(f"[{page.title}]: {content[:800]}") 
        except:
            continue
            
    return "\n\n".join(final_context)

# ==========================================
# 4. SOLVER (The Critical Part)
# ==========================================
def solve_question_cot(question, options, evidence):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    # We construct a strict prompt to force the model to look at options
    messages = [
        {"role": "system", "content": "You are a medical expert taking the USMLE. You must select the correct option."},
        {"role": "user", "content": f"""
        Background Information:
        {evidence}

        Question: 
        {question}

        Options:
        {options_fmt}

        Instructions:
        1. Evaluate each option (A, B, C, D) against the symptoms/context.
        2. Eliminate incorrect options with brief reasons.
        3. Conclude with the correct option.
        4. If the exact answer isn't in the text, use your medical knowledge.
        
        Format:
        Reasoning: [Your step-by-step logic]
        Answer: [Only the Letter A, B, C, or D]
        """}
    ]
    
    # INCREASED TO 512 TO PREVENT CUT-OFF
    return generate_text(messages, max_new_tokens=512)

def parse_final_answer(text):
    # Try to find "Answer: X" at the very end
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # Fallback: Look for the last capital letter option mentioned
    matches = re.findall(r"\b([A-D])\b", text)
    if matches: return matches[-1].upper()
    
    return "C" # Blind guess if parsing fails

# ==========================================
# 5. LOOP
# ==========================================
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
subset = dataset.select(range(NUM_SAMPLES_TO_TEST))
results = []

print(f"🚀 Starting...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    try:
        # Clear VRAM before starting a new complex generation
        torch.cuda.empty_cache()
        gc.collect()

        question = sample['question']
        options = sample['options']
        correct_key = sample['answer_idx']
        
        # 1. Classify
        cat = classify_question(question)
        
        # 2. Search
        query = generate_search_query(question, cat)
        context = get_best_wikipedia_context(query, options, question)
        
        # 3. Solve (With 512 token limit)
        response = solve_question_cot(question, options, context)
        pred = parse_final_answer(response)

        results.append({
            "id": i,
            "category": cat,
            "is_correct": (pred == correct_key),
            "prediction": pred,
            "correct_answer": correct_key,
            "reasoning": response
        })
        
        # Save every 10 rows just in case it crashes
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error: {e}")
        continue

pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)