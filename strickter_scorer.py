import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re

# ================= CONFIGURATION =================
CSV_PATH = "rag_evaluation_report.csv"  # Your input file
OUTPUT_PATH = "rag_evaluation_strict_final.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
# =================================================

class StrictJudge:
    def __init__(self, model_name):
        print(f"⚖️ Loading Strict Judge: {model_name}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            device_map="auto"
        )

    def _call_llm(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=250, 
                do_sample=False  # Deterministic (Greedy Decoding)
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1].strip()

    def evaluate_plan_strict(self, question, options, plan):
        """
        Judge if the search plan specifically targets the options.
        """
        prompt = f"""
        ### TASK
        Evaluate this Search Plan for a medical question. Categorize it into one of three strict levels.

        ### INPUT
        Question: {question}
        Options: {options}
        Plan: {plan}

        ### SCORING CRITERIA
        - SCORE 0 (Generic): Repeats keywords from the question (e.g., "Search for [Disease]"). Ignores the options completely.
        - SCORE 5 (Targeted): Mentions looking for the specific *concepts* in the options, but lacks comparison.
        - SCORE 10 (Differentiation): Explicitly targets the *difference* between the options (e.g., "Side effects of Drug A vs Drug B" or "Contraindications for Patient with Symptom X").

        ### OUTPUT FORMAT
        Score: [0, 5, or 10]
        Reason: [Explanation]
        """
        return self._call_llm(prompt)

    def evaluate_context_strict(self, question, options, context):
        """
        Judge if the retrieved text is SUFFICIENT to answer the question, not just relevant.
        """
        prompt = f"""
        ### TASK
        Evaluate if the Retrieved Text contains the *specific facts* needed to choose the correct option for this medical question.

        ### INPUT
        Question: {question}
        Options: {options}
        Retrieved Text: {context[:2500]}... (truncated)

        ### SCORING CRITERIA (Be Extremely Strict)
        - SCORE 0 (Irrelevant): Text discusses the wrong disease, wrong drug, or broad generalities (e.g., "Cancer is bad") without specifics.
        - SCORE 5 (Partial/Topical): Text discusses the correct disease/drug and is highly relevant, BUT it is missing the *exact fact* (e.g., dosage, specific gene mutation, or side effect) needed to distinguish the correct option from the distractors.
        - SCORE 10 (Sufficient): Text explicitly states the fact needed to answer (e.g., "Drug X causes Side Effect Y" or "Gene Z is mutated in this condition"). It enables a definitive answer.

        ### OUTPUT FORMAT
        Score: [0, 5, or 10]
        Reason: [Explanation]
        """
        return self._call_llm(prompt)

# ================= MAIN EXECUTION =================
def parse_score(text):
    # regex to find "Score: 10" or just "10"
    match = re.search(r"Score:\s*(\d+)", str(text))
    if match: return int(match.group(1))
    # Fallback: look for the first number in the response
    match = re.search(r"\b(0|5|10)\b", str(text))
    return int(match.group(1)) if match else 0

print(f"📂 Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Optional: Sample for testing
# df = df.head(20) 

print("🚀 Starting STRICT Evaluation...")
judge = StrictJudge(MODEL_ID)
tqdm.pandas()

# 1. Evaluate PLANS
print("... Grading Plans ...")
df['plan_eval_strict'] = df.progress_apply(
    lambda row: judge.evaluate_plan_strict(row['question'], row['options'], row['search_plan']), 
    axis=1
)

# 2. Evaluate CONTEXT RELEVANCE
print("... Grading Context ...")
df['context_eval_strict'] = df.progress_apply(
    lambda row: judge.evaluate_context_strict(row['question'], row['options'], row['retrieved_text']), 
    axis=1
)

# 3. Parse and Save
df['plan_score_strict'] = df['plan_eval_strict'].apply(parse_score)
df['context_score_strict'] = df['context_eval_strict'].apply(parse_score)

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Analysis Complete. Saved to {OUTPUT_PATH}")

# Quick Diagnostics
print("\n--- Plan Score Distribution ---")
print(df['plan_score_strict'].value_counts().sort_index())
print("\n--- Context Score Distribution ---")
print(df['context_score_strict'].value_counts().sort_index())