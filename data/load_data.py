# Step 0d: download IMDB 50k via HuggingFace, save as CSV for offline use.
from datasets import load_dataset
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

print("Downloading IMDB via HuggingFace (this fetches once, ~80MB)...")
ds = load_dataset("stanfordnlp/imdb")

train_df = pd.DataFrame(ds["train"])
test_df = pd.DataFrame(ds["test"])

train_df.to_csv(RAW / "imdb_train.csv", index=False)
test_df.to_csv(RAW / "imdb_test.csv", index=False)

print(f"\nTrain: {train_df.shape}  columns={list(train_df.columns)}")
print(f"Test:  {test_df.shape}")
print(f"\nTrain label distribution (0=neg, 1=pos):")
print(train_df["label"].value_counts(normalize=True).round(4))
print(f"\nSample review (first 200 chars):")
print(train_df.iloc[0]["text"][:200])