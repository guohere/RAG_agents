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
NUM_SAMPLES_TO_TEST = 200
OUTPUT_FILE = "medqa_optimized_results.csv"
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

# Increased max_tokens to 512 to allow space for "Reasoning"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.1)

# ==========================================
# 3. THE ROUTER (CLASSIFIER)
# ==========================================
def classify_question(question):
    prompt = [
        {"role": "system", "content": "Classify the following medical question into one of these categories: 'Diagnosis', 'Treatment', or 'Mechanism'. Output ONLY the category name."},
        {"role": "user", "content": f"Question: {question}\n\nCategory:"}
    ]
    output = pipe(prompt, max_new_tokens=10)
    # Extract the last message content
    raw_text = output[0]['generated_text'][-1]['content']
    category = raw_text.strip().lower()
    
    if "diagnosis" in category: return "Diagnosis"
    if "treatment" in category or "step" in category: return "Treatment"
    return "Mechanism" 

# ==========================================
# 4. OPTIMIZED RETRIEVAL STRATEGIES
# ==========================================
def get_evidence_diagnosis(options):
    """
    Strategy: Diagnosis questions usually match Symptoms -> Disease.
    We try to fetch the 'Signs and symptoms' section of the disease pages.
    """
    evidence = []
    for opt in options.values():
        try:
            page = wikipedia.page(opt)
            # Try to get specific symptom section, fallback to summary
            content = page.section("Signs and symptoms")
            if not content:
                content = page.summary
            
            # Limit length to avoid context overflow
            evidence.append(f"Disease: {opt}\nSymptoms/Info: {content[:400]}")
        except:
            continue
    return "\n\n".join(evidence)

def get_evidence_treatment(question, options):
    """
    Strategy: Treatment questions match Condition -> Drug/Therapy.
    We search for the 'Medical uses' or 'Treatment' sections of the option pages.
    """
    evidence = []
    for opt in options.values():
        try:
            page = wikipedia.page(opt)
            # Try to find treatment-related sections
            content = page.section("Medical uses")
            if not content:
                content = page.section("Treatment")
            if not content:
                content = page.summary
                
            evidence.append(f"Intervention: {opt}\nIndication/Usage: {content[:400]}")
        except:
            continue
    return "\n\n".join(evidence)

def get_evidence_mechanism(options):
    """
    Strategy: Mechanism questions are often about 'Pharmacology' or 'Mechanism of action'.
    """
    evidence = []
    for opt in options.values():
        try:
            page = wikipedia.page(opt)
            content = page.section("Mechanism of action")
            if not content:
                content = page.summary
            evidence.append(f"Fact about {opt}: {content[:400]}")
        except:
            continue
    return "\n\n".join(evidence)

# ==========================================
# 5. OPTIMIZED SOLVER (Chain of Thought)
# ==========================================
def solve_question_cot(question, options, evidence, category):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    # Dynamic System Role
    role_map = {
        "Diagnosis": "You are a Diagnostic Expert. Differential diagnosis is key.",
        "Treatment": "You are a Clinical Pharmacologist. Focus on first-line therapies.",
        "Mechanism": "You are a Biomedical Scientist. Focus on molecular pathways."
    }
    sys_role = role_map.get(category, "You are a Medical Expert.")

    # Chain-of-Thought Prompt
    messages = [
        {"role": "system", "content": f"{sys_role} Answer the USMLE question. Use the provided Context."},
        {"role": "user", "content": f"""
Context from Medical Library:
{evidence}

Question:
{question}

Options:
{options_fmt}

Instructions:
1. Analyze the patient's symptoms/history.
2. Evaluate each option against the context.
3. Eliminate incorrect options step-by-step.
4. State the final answer clearly at the end.

Format your response exactly like this:
Reasoning: [Your step-by-step logic]
Answer: [Option Letter]"""}
    ]
    
    # Generate
    output = pipe(messages)
    return output[0]['generated_text'][-1]['content']

def parse_final_answer(text):
    """
    Robust parser for Chain-of-Thought output.
    Looks for 'Answer: X' at the end of the text.
    """
    # Look for "Answer: A" pattern specifically
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: Look for just the letter if it appears at the very end
    match = re.search(r"\b([A-D])\s*$", text)
    if match:
        return match.group(1).upper()
        
    return "Unknown"

# ==========================================
# 6. MAIN PIPELINE EXECUTION
# ==========================================
print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
subset = dataset.select(range(NUM_SAMPLES_TO_TEST))

results = []
print(f"\n🚀 Starting Optimized Evaluation...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    question = sample['question']
    options = sample['options']
    correct_key = sample['answer_idx']
    
    # 1. ROUTE
    category = classify_question(question)
    
    # 2. RETRIEVE
    if category == "Diagnosis":
        context = get_evidence_diagnosis(options)
    elif category == "Treatment":
        context = get_evidence_treatment(question, options)
    else:
        context = get_evidence_mechanism(options)
        
    if not context: context = "No specific Wikipedia evidence found."

    # 3. SOLVE (CoT)
    raw_response = solve_question_cot(question, options, context, category)
    
    # 4. PARSE
    pred = parse_final_answer(raw_response)
    is_correct = (pred == correct_key)
    
    results.append({
        "id": i,
        "category": category,
        "is_correct": is_correct,
        "prediction": pred,
        "correct_answer": correct_key,
        "reasoning": raw_response[:500] + "..." # Save reasoning for review
    })

# ==========================================
# 7. ANALYSIS
# ==========================================
df = pd.DataFrame(results)
print("\n=== RESULTS BY CATEGORY ===")
print(df.groupby("category")["is_correct"].mean())
print(f"\nOverall Accuracy: {df['is_correct'].mean()*100:.2f}%")

df.to_csv(OUTPUT_FILE, index=False)