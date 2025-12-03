import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load Data
FILE_PATH = "./1000Q_v2.csv" # Make sure this matches your output filename
df = pd.read_csv(FILE_PATH)

print("="*40)
print("📊 RAG SYSTEM DIAGNOSTICS")
print("="*40)

# 1. Overall Accuracy
accuracy = df['is_correct'].mean() * 100
print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")

# 2. Category Breakdown (The most actionable insight)
print("\n🔍 Accuracy by Category:")
cat_acc = df.groupby("category")["is_correct"].agg(['mean', 'count'])
cat_acc['mean'] = cat_acc['mean'] * 100
print(cat_acc)

# 3. The "Unknown" Problem
unknowns = df[df['prediction'] == 'Unknown']
unknown_rate = len(unknowns) / len(df) * 100
print(f"\n⚠️ 'Unknown' Parsing Errors: {len(unknowns)} ({unknown_rate:.1f}%)")
if len(unknowns) > 0:
    print("Example Reasoning that failed parsing:")
    print(unknowns.iloc[0]['reasoning'][:300] + "...")

# 4. Error Analysis (Confusion Matrix)
# Only look at valid predictions (A, B, C, D)
valid_df = df[df['prediction'].isin(['A', 'B', 'C', 'D'])]
labels = sorted(list(set(valid_df['correct_answer'].unique()) | set(valid_df['prediction'].unique())))

cm = confusion_matrix(valid_df['correct_answer'], valid_df['prediction'], labels=labels)

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Where is the model confused?')
plt.savefig('confusion_matrix.png')
print("\n📉 Confusion matrix saved to 'confusion_matrix.png'")

# 5. Length Analysis (Does context length hurt?)
# We assume 'reasoning' length correlates with complexity
df['reasoning_len'] = df['reasoning'].str.len()
df['is_correct_int'] = df['is_correct'].astype(int)

print("\n📏 Accuracy by Reasoning Length (Quartiles):")
df['len_quartile'] = pd.qcut(df['reasoning_len'], 4, labels=["Short", "Medium", "Long", "Very Long"])
print(df.groupby('len_quartile')['is_correct'].mean() * 100)