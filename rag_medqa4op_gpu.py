import torch
import pandas as pd
import wikipedia
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & GPU CHECK
# ==========================================
NUM_SAMPLES_TO_TEST = 100  # You can increase this since GPU is fast!
OUTPUT_FILE = "medqa_gpu_results.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Verify GPU is visible
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ No GPU found! This script requires an NVIDIA GPU.")
    exit()

# ==========================================
# 2. LOAD MODEL (4-bit Quantization)
# ==========================================
print(f"Loading {MODEL_ID} in 4-bit mode...")

# This config shrinks the model to fit in 8GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", # Automatically puts layers on GPU
    trust_remote_code=True
)

# Create a text-generation pipeline
# We set max_new_tokens=50 because we only need a letter + short explanation
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=100,
    temperature=0.1,
    return_full_text=False
)

# ==========================================
# 3. DATA LOADING & HELPERS
# ==========================================
print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
test_subset = dataset.select(range(NUM_SAMPLES_TO_TEST))

def get_evidence(options):
    """Searches Wikipedia for the medical topics in the options."""
    search_queries = list(options.values())
    evidence_texts = []
    
    for query in search_queries:
        try:
            # Limit to 2 sentences to save context window space
            summary = wikipedia.summary(query, sentences=2)
            evidence_texts.append(f"Fact about {query}: {summary}")
        except:
            continue 
            
    if not evidence_texts:
        return "No specific external evidence found."
    return "\n".join(evidence_texts)

def extract_letter_from_response(response_text):
    """Extracts just A, B, C, or D from the model output."""
    match = re.search(r'\b([A-D])\b', response_text)
    if match:
        return match.group(1)
    return "Unknown"

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
results = []
print(f"\n🚀 Starting GPU Evaluation on {NUM_SAMPLES_TO_TEST} samples...")

for i, sample in tqdm(enumerate(test_subset), total=NUM_SAMPLES_TO_TEST):
    
    question = sample['question']
    options = sample['options']
    correct_key = sample['answer_idx']
    
    # A. Retrieve Evidence
    context = get_evidence(options)
    
    # B. Construct Prompt (Chat Template)
    # Qwen/Llama use specific chat templates. We use the standard list format.
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    messages = [
        {"role": "system", "content": "You are a medical expert. Answer the multiple-choice question. Output ONLY the correct option letter (A, B, C, or D) followed by a brief explanation."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nOptions:\n{options_fmt}\n\nWhich is the correct option?"}
    ]
    
    # C. Generate Answer
    output = pipe(messages)
    raw_response = output[0]['generated_text']
    
    # D. Parse & Score
    predicted_key = extract_letter_from_response(raw_response)
    is_correct = (predicted_key == correct_key)
    
    results.append({
        "id": i,
        "question": question[:50] + "...",
        "correct_answer": correct_key,
        "model_prediction": predicted_key,
        "is_correct": is_correct,
        "raw_response": raw_response,
        "evidence": context[:100] + "..."
    })

# ==========================================
# 5. RESULTS
# ==========================================
df_results = pd.DataFrame(results)
accuracy = df_results['is_correct'].mean() * 100

print("\n" + "="*30)
print(f"GPU EVALUATION COMPLETE")
print("="*30)
print(f"Model: {MODEL_ID} (4-bit)")
print(f"Accuracy: {accuracy:.2f}%")
print("="*30)

df_results.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")