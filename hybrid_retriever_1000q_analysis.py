import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ==========================================
# 1. SETUP
# ==========================================
RESULTS_FILE = "med_hybrid_retriever_1000q.csv" # Update to your actual filename
#EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO" # Fast & effective for similarity checking
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Fast & effective for similarity checking
# all-MiniLM-L6-v2
print(f"Loading Results from {RESULTS_FILE}...")
df = pd.read_csv(RESULTS_FILE)

print(f"Loading Evaluator Model ({EMBEDDING_MODEL})...")
model = SentenceTransformer(EMBEDDING_MODEL)

# ==========================================
# 2. METRIC 1: CATEGORY PERFORMANCE
# ==========================================
print("\n" + "="*40)
print("📊 1. CATEGORY BREAKDOWN")
print("="*40)
cat_stats = df.groupby("category")["is_correct"].agg(['count', 'mean']).sort_values("mean")
cat_stats['accuracy'] = (cat_stats['mean'] * 100).round(2)
print(cat_stats[['count', 'accuracy']])

# ==========================================
# 3. METRIC 2: CONTEXT SIMILARITY SCORE
# ==========================================
# We want to know: Did we actually find text related to the question?
print("\n" + "="*40)
print("📊 2. RETRIEVAL QUALITY CHECK")
print("="*40)

print("Calculating semantic similarity between Question and Retrieved Text...")

# Helper to compute similarity
def get_similarity(row):
    # We compare (Question + Options) vs (Retrieved Text)
    # This checks if the context actually talks about the concepts in the question
    query_text = f"{row['question']} {row['options']}"
    context_text = str(row['retrieved_text']) # Handle NaNs if any
    
    emb1 = model.encode(query_text, convert_to_tensor=True)
    emb2 = model.encode(context_text, convert_to_tensor=True)
    
    return util.cos_sim(emb1, emb2).item()

# Apply to all rows (might take 30s-1m for 1000 rows)
tqdm.pandas()
df['retrieval_score'] = df.progress_apply(get_similarity, axis=1)

# Define a "Good Retrieval" threshold (e.g., 0.4 cosine similarity)
# Scores below this usually mean we fetched junk.
THRESHOLD = 0.45 
df['good_context'] = df['retrieval_score'] > THRESHOLD

print("\n--- Retrieval vs Accuracy Correlation ---")
print(df.groupby('good_context')['is_correct'].mean() * 100)

# ==========================================
# 4. ERROR CLASSIFICATION
# ==========================================
print("\n" + "="*40)
print("📊 3. DIAGNOSING THE ERRORS")
print("="*40)

# Type A: Retrieval Failure (Wrong Answer + Low Similarity Score)
# The model likely hallucinated because it didn't have the facts.
retrieval_fails = df[(df['is_correct'] == False) & (df['retrieval_score'] < THRESHOLD)]

# Type B: Reasoning Failure (Wrong Answer + High Similarity Score)
# The answer WAS likely in the text, but the model picked the wrong option anyway.
reasoning_fails = df[(df['is_correct'] == False) & (df['retrieval_score'] >= THRESHOLD)]

print(f"Total Errors: {len(df[df['is_correct'] == False])}")
print(f"📉 Retrieval Failures (Bad Context): {len(retrieval_fails)} ({len(retrieval_fails)/len(df)*100:.1f}%)")
print(f"🧠 Reasoning Failures (Good Context): {len(reasoning_fails)} ({len(reasoning_fails)/len(df)*100:.1f}%)")

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='retrieval_score', hue='is_correct', bins=30, kde=True, element="step")
plt.axvline(THRESHOLD, color='red', linestyle='--', label='Relevance Threshold')
plt.title("Does Better Retrieval = Better Answers?")
plt.xlabel("Similarity Score (Question vs. Context)")
plt.savefig("retrieval_impact_analysis.png")
print("\nSaved plot to 'retrieval_impact_analysis.png'")

# ==========================================
# 6. EXPORT FAILURE EXAMPLES
# ==========================================
# Save these to look at manually! This is how you find the next bug to fix.
print("\nWriting debug reports...")

with open("debug_retrieval_failures.txt", "w", encoding='utf-8') as f:
    f.write("=== QUESTIONS WHERE WE FOUND NOTHING RELEVANT ===\n\n")
    for i, row in retrieval_fails.head(20).iterrows():
        f.write(f"ID: {row['id']} | Category: {row['category']}\n")
        f.write(f"Q: {row['question'][:100]}...\n")
        f.write(f"Search Plan: {row.get('search_plan', 'N/A')}\n")
        f.write(f"Retrieved (Score {row['retrieval_score']:.2f}): {row['retrieved_text'][:200]}...\n")
        f.write("-" * 20 + "\n")

with open("debug_reasoning_failures.txt", "w", encoding='utf-8') as f:
    f.write("=== QUESTIONS WHERE CONTEXT WAS GOOD BUT MODEL FAILED ===\n\n")
    for i, row in reasoning_fails.head(20).iterrows():
        f.write(f"ID: {row['id']} | Category: {row['category']}\n")
        f.write(f"Q: {row['question'][:100]}...\n")
        f.write(f"Correct: {row['correct_answer']} | Predicted: {row['prediction']}\n")
        f.write(f"Reasoning: {row['reasoning']}\n")
        f.write("-" * 20 + "\n")

print("Done! Check 'debug_retrieval_failures.txt' and 'debug_reasoning_failures.txt'.")