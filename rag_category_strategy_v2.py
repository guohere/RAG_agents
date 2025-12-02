import torch
import pandas as pd
import wikipedia
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from tqdm import tqdm

"""
1. query expansion with question rewriting by LLM
2. retrieve article for question and 4 options
3. 
"""

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 10
OUTPUT_FILE = "smart_retrieve_query_expansion.csv"
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
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0.1)

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

def generate_search_query(question, category):
    """
    Uses the LLM to extract the core medical concept to search for.
    Why? Because the raw question is too long for a wiki search.
    """
    prompt = [
        {"role": "system", "content": "You are a search engine optimizer for medical data."},
        {"role": "user", "content": f"""
        Extract the SINGLE most important medical concept or disease from this question to search on Wikipedia.
        
        Question: {question}
        Category: {category}
        
        Output ONLY the search term. Do not output a sentence.
        Example: 
        Input: "A 45-year-old man with crushing chest pain..."
        Output: Myocardial infarction
        """}
    ]
    # Small max_tokens because we just want a keyword
    output = pipe(prompt, max_new_tokens=20)
    search_term = output[0]['generated_text'][-1]['content'].strip()
    return search_term

def get_best_wikipedia_context(search_term, options, question_text):
    """
    Searches for the main topic AND the options.
    Returns the top 5 most relevant distinct pages/sections.
    """
    candidates = []
    
    # 1. Search for the main concept (e.g., "Carpal tunnel syndrome")
    try:
        # search() returns a list of page titles
        results = wikipedia.search(search_term, results=1)
        if results:
            candidates.append(results[0])
    except:
        pass

    # 2. Add options to candidates
    for opt in options.values():
        candidates.append(opt)
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    final_context = []
    page_count = 0
    
    print(f"   -> Searching Wiki for: {candidates}")

    for page_title in candidates:
        if page_count >= 5: break # HARD LIMIT: 5 Pages
        
        try:
            # auto_suggest=False prevents it from guessing wrong pages
            page = wikipedia.page(page_title, auto_suggest=False)
            
            # CRITICAL IMPROVEMENT:
            # Don't just grab "Summary". Grab the whole text and score paragraphs.
            # (Simplifying here for speed: Grab Summary + Content)
            full_text = page.content
            
            # Simple "Rerank": specific overlap check
            # If the question mentions "side effect", look for that paragraph
            relevant_segment = page.summary
            
            # Heuristic: If question is about treatment, prioritize that section
            if "treatment" in question_text.lower() or "management" in question_text.lower():
                section = page.section("Treatment") or page.section("Management")
                if section: relevant_segment = section

            final_context.append(f"Source: {page.title}\nContent: {relevant_segment[:1000]}") # Limit chars per page
            page_count += 1
            
        except wikipedia.DisambiguationError as e:
            # Handle ambiguous terms (pick the first medical-sounding one)
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                final_context.append(f"Source: {page.title}\nContent: {page.summary[:1000]}")
                page_count += 1
            except: continue
        except:
            continue
            
    return "\n\n".join(final_context)

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
    
    sys_role = "You are a Medical Expert. specific and concise."

    messages = [
        {"role": "system", "content": sys_role},
        {"role": "user", "content": f"""
            Context:
            {evidence}

            Question:
            {question}

            Options:
            {options_fmt}

            Task:
            1. Analyze symptoms briefly.
            2. Eliminate wrong options based on context.
            3. Select the best answer.
            
            CRITICAL INSTRUCTIONS:
            - If the exact answer is missing from context, use your internal medical knowledge to make an educated guess.
            - You MUST select one option (A, B, C, or D).
            - NEVER say "None" or "Unknown".
            
            Format:
            Reasoning: [Concise logic]
            Answer: [Option Letter]
        """}
    ]
    
    output = pipe(messages)
    return output[0]['generated_text'][-1]['content']

def parse_final_answer(text, options_keys=['A', 'B', 'C', 'D']):
    """
    Robust parser that forces a choice if the model is indecisive.
    """
    # 1. Look for explicit "Answer: X"
    match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 2. Look for the last mentioned option letter in the reasoning
    # (Often the model says "Therefore, A is correct.")
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
        
    # 3. Last Resort: Random Guess (Better than "Unknown" for metrics)
    # or return the first option 'A' as a default
    return "C"

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
    
    search_query = generate_search_query(question, category)

    context = get_best_wikipedia_context(search_query, options, question)
    # 3. SOLVE (CoT)
    if not context: 
        context = "No specific Wikipedia evidence found. Use internal knowledge."

    raw_response = solve_question_cot(question, options, context, category)
    
    # 4. PARSE
    pred = parse_final_answer(raw_response)
    if pred == "Unknown":
        print(f"   ⚠️ Truncation detected for ID {i}. Attempting repair...")
        
        # We assume the reasoning is there but cut off. 
        # We force the model to just output the letter based on what it wrote so far.
        repair_messages = [
            {"role": "user", "content": f"""
            Based on the following medical reasoning, which is the correct option letter (A, B, C, or D)?
            
            Reasoning so far:
            {raw_response[-1000:]} 
            
            Output ONLY the letter.
            """}
        ]
        repair_output = pipe(repair_messages, max_new_tokens=10)[0]['generated_text'][-1]['content']
        pred = parse_final_answer(repair_output)
        
        # Append a note to reasoning so you know it was repaired
        raw_response += f"\n[SYSTEM: Output was truncated. Repair Loop Prediction: {pred}]"

    # Save result
    results.append({
        "id": i,
        "category": category,
        "is_correct": (pred == correct_key),
        "prediction": pred,
        "correct_answer": correct_key,
        "reasoning": raw_response
    })

# ==========================================
# 7. ANALYSIS
# ==========================================
df = pd.DataFrame(results)
print("\n=== RESULTS BY CATEGORY ===")
print(df.groupby("category")["is_correct"].mean())
print(f"\nOverall Accuracy: {df['is_correct'].mean()*100:.2f}%")

df.to_csv(OUTPUT_FILE, index=False)