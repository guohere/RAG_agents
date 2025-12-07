import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ================= CONFIGURATION =================
CSV_PATH = "rag_evaluation_report.csv"  # Ensure this matches your saved file
# =================================================

def parse_score(val):
    """Robust parser to extract integers from LLM text responses"""
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return int(val)
    # Extract first number found in string like "Score: 4" or "4/5"
    match = re.search(r"(\d)", str(val))
    return int(match.group(1)) if match else 0

# 1. Load and Clean Data
print(f"📂 Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Ensure scores are integers (re-parsing just in case)
df['plan_score'] = df['plan_eval'].apply(parse_score)
df['context_score'] = df['context_eval'].apply(parse_score)

# Filter for Wrong Answers
wrong_df = df[df['is_correct'] == False]

print(f"📊 Loaded {len(df)} records. Found {len(wrong_df)} wrong answers.")

# ================= METRIC CALCULATIONS =================
# Define thresholds
BAD_PLAN_THRESHOLD = 3
BAD_CONTEXT_THRESHOLD = 2

# 1. Query Generator Failure: Plan is bad (Score <= 3)
bad_plans_count = len(df[df['plan_score'] <= BAD_PLAN_THRESHOLD])

# 2. Retriever Failure: Plan is Good (5), but Context is Bad (<= 2)
# We use strictly '5' for good plans to be conservative, or >=4 for lenient
retriever_fail_count = len(df[
    (df['plan_score'] >= 4) & 
    (df['context_score'] <= BAD_CONTEXT_THRESHOLD)
])

print("\n" + "="*40)
print("🚀 DIAGNOSTIC SUMMARY")
print("="*40)
print(f"📉 Weak Query Generation: {bad_plans_count} / {len(df)} ({bad_plans_count/len(df):.1%})")
print(f"   (Questions where the search plan was generic or vague)")
print(f"📉 Retrieval Gaps:        {retriever_fail_count} / {len(df)} ({retriever_fail_count/len(df):.1%})")
print(f"   (Questions where the plan was solid, but we still found nothing)")


# ================= VISUALIZATION =================
sns.set_theme(style="whitegrid")

# Create a figure with 3 subplots
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 3)

# PLOT 1: Overall Score Distributions (Histogram)
ax1 = fig.add_subplot(gs[0, 0])
# Melt data for side-by-side plotting
melted = df.melt(value_vars=['plan_score', 'context_score'], var_name='Metric', value_name='Score')
sns.histplot(
    data=melted, x='Score', hue='Metric', 
    multiple='dodge', discrete=True, shrink=.8, ax=ax1, palette='viridis'
)
ax1.set_title("Global Distribution: Plan vs. Context Scores")
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_ylabel("Count")

# PLOT 2: Pipeline Heatmap (All Data)
# Shows: Does a Good Plan -> Good Context?
ax2 = fig.add_subplot(gs[0, 1])
heatmap_data_all = pd.crosstab(df['plan_score'], df['context_score'])
sns.heatmap(heatmap_data_all, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title("Pipeline Health (All Questions)")
ax2.set_ylabel("Plan Quality (1-5)")
ax2.set_xlabel("Context Relevance (1-5)")
ax2.invert_yaxis() # Put Score 5 at top

# PLOT 3: Root Cause Heatmap (Wrong Answers Only)
# Shows: When we fail, is it the Plan or the Context?
ax3 = fig.add_subplot(gs[0, 2])
heatmap_data_wrong = pd.crosstab(wrong_df['plan_score'], wrong_df['context_score'])
sns.heatmap(heatmap_data_wrong, annot=True, fmt='d', cmap='Reds', ax=ax3, cbar=False)
ax3.set_title("Failure Analysis (Wrong Answers Only)")
ax3.set_ylabel("Plan Quality (1-5)")
ax3.set_xlabel("Context Relevance (1-5)")
ax3.invert_yaxis()

plt.tight_layout()
plt.savefig("rag_score_distribution.png")
print("\n✅ Saved plot to 'rag_score_distribution.png'")
plt.show()