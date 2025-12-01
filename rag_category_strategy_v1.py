import torch
import pandas as pd
import wikipedia
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 100
OUTPUT_FILE = "medqa_adaptive_results.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

if not torch.cuda.is_available():
    print("❌ No GPU found!")
    exit()

# ==========================================
# 2. LOAD MODEL (4-bit)
# ==========================================
print(f"Loading {MODEL_ID}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")

# We use a single pipeline for both Classification and Answering
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, temperature=0.1)

# ==========================================
# 3. THE ROUTER (CLASSIFIER)
# ==========================================
def classify_question(question):
    """
    Decides if the question is about Diagnosis, Treatment, or Mechanism.
    """
    prompt = [
        {"role": "system", "content": "Classify the following medical question into one of these categories: 'Diagnosis', 'Treatment', or 'Mechanism'. Output ONLY the category name."},
        {"role": "user", "content": f"Question: {question}\n\nCategory:"}
    ]
    output = pipe(prompt, max_new_tokens=10)
    category = output[0]['generated_text'].strip().lower()
    
    if "diagnosis" in category: return "Diagnosis"
    if "treatment" in category or "step" in category: return "Treatment"
    return "Mechanism" # Default fallback

# ==========================================
# 4. ADAPTIVE RETRIEVAL STRATEGIES
# ==========================================
def get_evidence_diagnosis(options):
    """Strategy: Fetch symptoms of the diseases in the options."""
    evidence = []
    for opt in options.values():
        try:
            # Search for the disease page
            page = wikipedia.page(opt)
            # We specifically want the 'Signs and symptoms' section if possible
            content = page.content[:500] # Fallback to summary
            evidence.append(f"Disease: {opt}\nInfo: {content}")
        except:
            continue
    return "\n\n".join(evidence)

def get_evidence_treatment(question, options):
    """
    Strategy: Identify the condition from the question, then search for its treatment.
    """
    # 1. Extract the likely condition from the vignette (using LLM would be best, but we use simple heuristic here)
    # For a lab, we will search the Question keywords + "treatment"
    # A real production system would use the LLM to extract the "Probable Diagnosis" first.
    
    # Simple approach: Search for the drugs/procedures in options to see what they treat
    evidence = []
    for opt in options.values():
        try:
            summary = wikipedia.summary(opt, sentences=3)
            evidence.append(f"Intervention: {opt}\nUsed for: {summary}")
        except:
            continue
    return "\n\n".join(evidence)

def get_evidence_mechanism(options):
    """Strategy: Search for the specific fact/mechanism."""
    evidence = []
    for opt in options.values():
        try:
            summary = wikipedia.summary(opt, sentences=3)
            evidence.append(f"Fact about {opt}: {summary}")
        except:
            continue
    return "\n\n".join(evidence)

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
subset = dataset.select(range(NUM_SAMPLES_TO_TEST))

results = []
print(f"\n🚀 Starting Adaptive Evaluation...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    question = sample['question']
    options = sample['options']
    correct_key = sample['answer_idx']
    
    # --- STEP 1: ROUTING ---
    category = classify_question(question)
    
    # --- STEP 2: ADAPTIVE RETRIEVAL ---
    if category == "Diagnosis":
        context = get_evidence_diagnosis(options)
        sys_prompt = "You are a Diagnostician. Match the patient's symptoms to the correct disease."
    elif category == "Treatment":
        context = get_evidence_treatment(question, options)
        sys_prompt = "You are a Clinical Pharmacologist. Choose the appropriate management or drug."
    else:
        context = get_evidence_mechanism(options)
        sys_prompt = "You are a Biomedical Scientist. Explain the underlying mechanism or fact."
        
    if not context: context = "No Wikipedia evidence found."

    # --- STEP 3: GENERATION ---
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    messages = [
        {"role": "system", "content": sys_prompt + " Output ONLY the correct option letter (A, B, C, or D)."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nOptions:\n{options_fmt}\n\nAnswer:"}
    ]
    
    output = pipe(messages)
    raw_response = output[0]['generated_text']
    
    # Parse
    match = re.search(r'\b([A-D])\b', raw_response)
    pred = match.group(1) if match else "Unknown"
    
    results.append({
        "id": i,
        "category": category, # <-- Save the category to analyze which one we are best at!
        "is_correct": pred == correct_key,
        "prediction": pred,
        "evidence_length": len(context)
    })

# ==========================================
# 6. ANALYSIS
# ==========================================
df = pd.DataFrame(results)
print("\n=== RESULTS BY CATEGORY ===")
print(df.groupby("category")["is_correct"].mean())
print(f"\nOverall Accuracy: {df['is_correct'].mean()*100:.2f}%")

df.to_csv(OUTPUT_FILE, index=False)
