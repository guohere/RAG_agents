import pandas as pd
import wikipedia
import re
from datasets import load_dataset
from ctransformers import AutoModelForCausalLM
from tqdm import tqdm  # Progress bar

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
NUM_SAMPLES_TO_TEST = 10  # <-- Start small! CPU is slow.
OUTPUT_FILE = "medqa_results.csv"

print("Loading Llama-2-7b (CPU)...")
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF",
    model_file="llama-2-7b-chat.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=0,
    context_length=2048
)

print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

# Select only the first N examples for this test run
test_subset = dataset.select(range(NUM_SAMPLES_TO_TEST))

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_evidence(options):
    """Searches Wikipedia for the medical topics in the options."""
    search_queries = list(options.values())
    evidence_texts = []
    
    for query in search_queries:
        try:
            # Get 2 sentences summary to save context window
            summary = wikipedia.summary(query, sentences=2)
            evidence_texts.append(f"Fact about {query}: {summary}")
        except:
            continue # Skip if page not found/ambiguous
            
    if not evidence_texts:
        return "No external evidence found."
    return "\n".join(evidence_texts)

def extract_letter_from_response(response_text):
    """
    The model might say 'The correct answer is A.' or 'Option B is right.'
    We need to extract just 'A', 'B', 'C', or 'D'.
    """
    # Regex looks for a capital letter A-D alone or with punctuation
    # e.g., matches "A", "A.", "(A)", "Answer: A"
    match = re.search(r'\b([A-D])\b', response_text)
    if match:
        return match.group(1)
    return "Unknown"

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
results = []

print(f"\nStarting evaluation on {NUM_SAMPLES_TO_TEST} samples...")

# tqdm creates a progress bar in your terminal
for i, sample in tqdm(enumerate(test_subset), total=NUM_SAMPLES_TO_TEST):
    
    question = sample['question']
    options = sample['options']
    correct_key = sample['answer_idx'] # e.g., 'A'
    
    # A. Retrieve Evidence
    context = get_evidence(options)
    
    # B. Construct Prompt
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    prompt = f"""[INST] <<SYS>>
You are a medical expert taking a test. 
Read the Context and Question. Output ONLY the correct option letter (A, B, C, or D) and nothing else.
<</SYS>>

Context:
{context}

Question: 
{question}

Options:
{options_fmt}

Correct Option Letter: [/INST]"""

    # C. Generate Answer
    # temp=0.1 ensures the model is less random
    raw_response = llm(prompt, temperature=0.1)
    
    # D. Parse Answer
    predicted_key = extract_letter_from_response(raw_response)
    
    # E. Check Correctness
    is_correct = (predicted_key == correct_key)
    
    # F. Save Result
    results.append({
        "id": i,
        "question": question[:50] + "...", # Truncate for readability
        "correct_answer": correct_key,
        "model_prediction": predicted_key,
        "is_correct": is_correct,
        "raw_model_response": raw_response, # Keep this to debug!
        "evidence_found": context[:100] + "..."
    })

# ==========================================
# 4. FINAL SUMMARY & SAVING
# ==========================================
df_results = pd.DataFrame(results)

# Calculate Accuracy
accuracy = df_results['is_correct'].mean() * 100

print("\n" + "="*30)
print(f"FINAL RESULTS")
print("="*30)
print(f"Total Samples: {NUM_SAMPLES_TO_TEST}")
print(f"Accuracy: {accuracy:.2f}%")
print("="*30)

# Save to CSV
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"Detailed results saved to '{OUTPUT_FILE}'")

# Show the first few rows
print(df_results[['question', 'correct_answer', 'model_prediction', 'is_correct']].head())