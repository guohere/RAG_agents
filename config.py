import os
import torch
from transformers import BitsAndBytesConfig

# --- PATHS ---
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.expanduser("~/.cache/huggingface/datasets")

# --- MODEL IDS ---
LLM_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_ID = "all-MiniLM-L6-v2"

# --- HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- QUANTIZATION ---
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)