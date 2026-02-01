from huggingface_hub import hf_hub_download
import os

# ===== CONFIG =====
REPO_ID = "bartowski/gemma-2-2b-it-abliterated-GGUF"  # exact repo
FILENAME = "gemma-2-2b-it-abliterated-Q4_K_M.gguf"    # exact file
OUTPUT_DIR = "./"
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading model...")

# Download the GGUF from Hugging Face
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=OUTPUT_DIR,
    local_dir_use_symlinks=False,
    repo_type="model"  # ensures HF treats this as a model repo
)

print(f"Done.\nModel saved at:\n{model_path}")
