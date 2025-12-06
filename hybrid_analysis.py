import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ================= CONFIGURATION =================
CSV_PATH = "med_hybrid_retriever_1000q.csv"  # Your file
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH = "rag_evaluation_report.csv"
# =================================================

class RAGEvaluator:
    def __init__(self, model_name):
        print(f"⚖️ Loading Judge: {model_name}...")
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
                max_new_tokens=150, 
                temperature=0.0  # Greedy decoding for consistent grading
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1].strip()

    def evaluate_search_plan(self, question, options, plan):
        """
        Check if the plan is lazy (generic) or smart (targeted).
        """
        prompt = f"""
        ### TASK
        Rate the quality of this Search Plan for a Medical Question (1-5).
        
        ### CRITERIA
        - Score 1 (Bad): Generic keywords only (e.g. "Symptoms of Flu").
        - Score 3 (Okay): Targets the main condition but misses the nuance of the options.
        - Score 5 (Excellent): Explicitly targets the *differentiating factors* between the options (e.g. "Side effects of Drug A vs Drug B").
        
        ### INPUT
        Question: {question}
        Options: {options}
        Generated Search Plan: {plan}
        
        ### OUTPUT FORMAT
        Score: [Number]
        Reason: [Short explanation]
        """
        return self._call_llm(prompt)

    def evaluate_context_relevance(self, question, options, context):
        """
        Check if context helps solve the specific options dilemma.
        """
        prompt = f"""
        ### TASK
        Rate the utility of the Retrieved Text for answering the Question given the Options (1-5).
        
        ### INPUT
        Question: {question}
        Options: {options}
        Retrieved Text: {context[:2000]}... (truncated)
        
        ### CRITERIA
        - Score 1 (Irrelevant): Text talks about the wrong topic or wrong disease.
        - Score 3 (Partial): Defines the disease but lacks specific details (dosage, gene, mechanism) needed to pick the option.
        - Score 5 (Perfect): Contains the EXACT fact needed to prove the correct option OR disprove the distractors.
        
        ### OUTPUT FORMAT
        Score: [Number]
        Reason: [Short explanation]
        """
        return self._call_llm(prompt)

# ================= MAIN EXECUTION =================
print("📂 Loading CSV...")
df = pd.read_csv(CSV_PATH)

# Optional: Sample for testing
# df = df.head(10) 

evaluator = RAGEvaluator(MODEL_ID)

print("🚀 Starting Evaluation...")
tqdm.pandas()

# 1. Evaluate the Plan
df['plan_eval'] = df.progress_apply(
    lambda row: evaluator.evaluate_search_plan(row['question'], row['options'], row['search_plan']), 
    axis=1
)

# 2. Evaluate the Context
df['context_eval'] = df.progress_apply(
    lambda row: evaluator.evaluate_context_relevance(row['question'], row['options'], row['retrieved_text']), 
    axis=1
)

# 3. Parse Scores (Simple regex/string splitting)
def parse_score(text):
    try:
        # Looks for "Score: 5" or "5/5" patterns
        import re
        match = re.search(r"Score:\s*(\d)", text)
        return int(match.group(1)) if match else None
    except:
        return None

df['plan_score'] = df['plan_eval'].apply(parse_score)
df['context_score'] = df['context_eval'].apply(parse_score)

print(f"💾 Saving detailed report to {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)