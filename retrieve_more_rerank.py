import torch
import pandas as pd
from rank_bm25 import BM25Okapi
import wikipedia
import re
import ast
import gc
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder # <--- Added CrossEncoder
from tqdm import tqdm
import numpy as np
from sentence_transformers import util

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# ==========================================
# 1. CONFIGURATION
# ==========================================
NUM_SAMPLES_TO_TEST = 1000 
OUTPUT_FILE = "med_hierarchical_retriever_1000q.csv" 
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_ID = "pritamdeka/S-PubMedBert-MS-MARCO"
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2" # <--- New Reranker

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

print(f"Loading Bi-Encoder ({EMBEDDING_ID})...")
embedder = SentenceTransformer(EMBEDDING_ID, device="cpu")

print(f"Loading Cross-Encoder ({RERANKER_ID})...")
# We load this on GPU if possible for speed, or CPU if VRAM is tight
reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(RERANKER_ID, device=reranker_device)

print("Loading Dataset...")
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

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
# 3. HIERARCHICAL RETRIEVER LOGIC
# ==========================================

class HierarchicalRetriever:
    def __init__(self, reranker_model):
        self.reranker = reranker_model
        
    def _chunk_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            c = text[i:i+chunk_size]
            if len(c) > 50:
                chunks.append(c)
        return chunks

    def retrieve(self, question, search_queries, top_k_chunks=5):
        """
        The Funnel Strategy:
        1. Broad Search: Fetch ~15 candidate pages (Title + Summary)
        2. Page Filter: Rerank pages -> Keep Top 5
        3. Drill Down: Download Full Text -> Chunk -> Rerank Chunks -> Keep Top 5
        """
        candidates = []
        seen_titles = set()
        
        # --- PHASE 1: BROAD SEARCH ---
        # print(f"   🔎 searching: {search_queries}")
        for q in search_queries:
            try:
                # Get titles first
                results = wikipedia.search(q, results=3) # Limit to 3 per query to save time
                for title in results:
                    if title in seen_titles: continue
                    seen_titles.add(title)
                    
                    try:
                        # Fetch summary for reranking
                        page = wikipedia.page(title, auto_suggest=False)
                        candidates.append({
                            "title": title,
                            "summary": page.summary[:500], # First 500 chars for speed
                            "url": page.url,
                            "full_page_obj": page # Keep ref to avoid re-fetching
                        })
                    except (wikipedia.DisambiguationError, wikipedia.PageError):
                        continue
            except:
                continue
                
        if not candidates:
            return "No content found.", "No Sources"

        # --- PHASE 2: PAGE RERANKING ---
        # Score pairs: [Question, Title + Summary]
        page_inputs = [[question, f"{c['title']}: {c['summary']}"] for c in candidates]
        
        # Cross-Encoder scoring
        page_scores = self.reranker.predict(page_inputs)
        
        # Sort and take Top 5 Pages
        top_page_indices = np.argsort(page_scores)[::-1][:5]
        top_pages = [candidates[i] for i in top_page_indices]
        
        # --- PHASE 3: DRILL DOWN (CHUNKING) ---
        all_chunks = []
        
        for p in top_pages:
            try:
                # We already have the page object, get full content
                full_text = p['full_page_obj'].content
                chunks = self._chunk_text(full_text)
                
                for c in chunks:
                    all_chunks.append({
                        "text": c,
                        "source": p['title'],
                        "url": p['url']
                    })
            except:
                continue

        if not all_chunks:
            return "No valid chunks extracted.", "No Sources"

        # --- PHASE 4: CHUNK RERANKING ---
        # Score pairs: [Question, Chunk Text]
        # NOTE: Cross-Encoder is slower than Bi-Encoder, but for <200 chunks it's fine
        # If too slow, limit all_chunks to top 100 before scoring
        if len(all_chunks) > 100:
             all_chunks = all_chunks[:100]

        chunk_inputs = [[question, c['text']] for c in all_chunks]
        chunk_scores = self.reranker.predict(chunk_inputs)
        
        # Get Top K Chunks
        top_chunk_indices = np.argsort(chunk_scores)[::-1][:top_k_chunks]
        final_chunks = [all_chunks[i] for i in top_chunk_indices]
        
        # Format Output
        formatted_context = "\n...\n".join([c['text'] for c in final_chunks])
        unique_sources = list(set([f"{c['source']} ({c['url']})" for c in final_chunks]))
        source_str = " | ".join(unique_sources)
        
        return formatted_context, source_str

# Initialize the class
hierarchical_retriever = HierarchicalRetriever(reranker)


def classify_question(question):
    prompt = [
        {"role": "system", "content": "Classify as: Diagnosis, Treatment, or Mechanism. Output ONE word."},
        {"role": "user", "content": f"Question: {question}\nCategory:"}
    ]
    return generate_text(prompt, max_new_tokens=10).strip().replace(".", "")

# ==========================================
# 4. SOLVER & PLANNER
# ==========================================

def get_smart_search_plan(question, options, category):
    """
    Enhanced Planner: Explicitly asks for option differentiation.
    """
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    instruction_add_on = ""
    if category == "Treatment":
        instruction_add_on = "Focus on guidelines and distinguishing between the drug options."
    elif category == "Mechanism":
        instruction_add_on = "Focus on the specific mechanism differences between the options."
    
    prompt = [
        {"role": "system", "content": "You are a Medical Research Planner. Generate 5 distinct Wikipedia search queries to answer the question."},
        {"role": "user", "content": f"""
Question: {question}

Options:
{options_text}

TASK:
1. Query 1: Focus on the main medical condition/symptom.
2. Query 2: Focus strictly on the **differences** between the Options (e.g. "Lisinopril vs Losartan side effects").
3. Query 3: Focus on {instruction_add_on or "specific contraindications"}.

Output a Python list of strings.
Example: ["Kawasaki disease symptoms", "Aspirin vs Ibuprofen mechanism", "Coronary artery aneurysm complications"]

Your Output (Python List ONLY):
"""}
    ]
    
    plan_text = generate_text(prompt, max_new_tokens=150).strip()
    try:
        if "[" in plan_text:
            plan_text = plan_text[plan_text.find("["):plan_text.rfind("]")+1]
        query_list = ast.literal_eval(plan_text)
        return query_list[:5] 
    except:
        return [question[:100]]

def solve_question_cot(question, options, evidence, examples):
    options_fmt = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    messages = [
        {"role": "system", "content": "You are a medical expert. Answer based ONLY on the provided Context."},
        {"role": "user", "content": f"""
Background Context (Wikipedia):
{evidence}

Question: 
{question}

Options:
{options_fmt}

Instructions:
1. **Rule Out**: Use the context to disprove incorrect options.
2. **Support**: Find the specific fact in the context that supports the correct option.
3. **Conclusion**: Pick the best answer.

Format:
Reasoning: [Step-by-step logic]
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

print(f"Starting Hierarchical RAG Pipeline...")

for i, sample in tqdm(enumerate(subset), total=NUM_SAMPLES_TO_TEST):
    try:
        torch.cuda.empty_cache()
        gc.collect()

        question = sample['question']
        options = sample['options']
        correct_key = sample['answer_idx']
        
        # 1. Classify
        cat = classify_question(question)
        
        # 2. Plan
        search_plan = get_smart_search_plan(question, options, cat)
        
        # 3. Retrieve (The New Funnel)
        context_text, source_refs = hierarchical_retriever.retrieve(question, search_plan)
        
        # 4. Solve
        response = solve_question_cot(question, options, context_text, examples=None)
        pred = parse_final_answer(response)

        # 5. Save
        results.append({
            "id": i,
            "category": cat,
            "is_correct": (pred == correct_key),
            "prediction": pred,
            "correct_answer": correct_key,
            "search_plan": search_plan,
            "question": question,
            "options": options,
            "retrieved_text": context_text[:2000],
            "source_references": source_refs,
            "reasoning": response,
        })
        
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"Error at index {i}: {e}")
        continue

df = pd.DataFrame(results)
print(f"Final Accuracy: {df['is_correct'].mean()*100:.2f}%")
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")