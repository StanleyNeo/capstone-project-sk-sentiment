# Step 2: DistilBERT-SST2 inference on the same 25k test set.
# Pretrained, no fine-tuning — we're measuring what an off-the-shelf
# transformer actually buys us over the TF-IDF baseline.
import time
import pandas as pd
from pathlib import Path
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

RAW = Path("data/raw")
test_df = pd.read_csv(RAW / "imdb_test.csv")
texts  = test_df["text"].tolist()
y_true = test_df["label"].tolist()

print("Loading DistilBERT-SST2 (downloads ~270MB on first run)...")
t0 = time.time()
clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
    device=-1,          # CPU
)
print(f"Model loaded in {time.time() - t0:.1f}s\n")

# The model outputs {'label': 'POSITIVE'|'NEGATIVE', 'score': float}
# IMDB labels: 1 = positive, 0 = negative. Map explicitly.
LABEL_MAP = {"POSITIVE": 1, "NEGATIVE": 0}

print(f"Running inference on {len(texts)} reviews (CPU, this will take a while)...")
t0 = time.time()
# batch_size=32 gives good throughput on CPU without blowing memory
raw = clf(texts, batch_size=32, truncation=True, max_length=512)
predict_secs = time.time() - t0

y_pred  = [LABEL_MAP[r["label"]] for r in raw]
scores  = [r["score"] for r in raw]

# --- Metrics ---
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

# --- Size on disk (huggingface cache) ---
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
size_mb = sum(f.stat().st_size for f in hf_cache.rglob("*.bin")
              if "distilbert-base-uncased-finetuned-sst-2" in str(f)) / (1024 * 1024)
size_mb += sum(f.stat().st_size for f in hf_cache.rglob("*.safetensors")
               if "distilbert-base-uncased-finetuned-sst-2" in str(f)) / (1024 * 1024)

print("\n" + "=" * 60)
print("STEP 2 — DistilBERT-SST2 (pretrained, no fine-tune)")
print("=" * 60)
print(f"Test:                {len(texts)} reviews")
print()
print(f"Accuracy:            {acc:.4f}")
print(f"Precision:           {prec:.4f}")
print(f"Recall:              {rec:.4f}")
print(f"F1:                  {f1:.4f}")
print()
print(f"Predict time (25k):  {predict_secs:.1f} s  ({predict_secs/60:.1f} min)")
print(f"Model size on disk:  {size_mb:.0f} MB")
print()
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))