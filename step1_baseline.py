# Step 1: TF-IDF + LogisticRegression baseline for IMDB sentiment.
# Fast, tiny, CPU-only. This is the model DistilBERT has to beat.
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from pathlib import Path

RAW = Path("data/raw")
train_df = pd.read_csv(RAW / "imdb_train.csv")
test_df = pd.read_csv(RAW / "imdb_test.csv")

X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

# --- The pipeline: TF-IDF vectorizer + LogisticRegression ---
# ngram_range=(1,2) captures "not good" and similar bigrams — important for sentiment.
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(max_iter=1000, C=4.0, solver="liblinear")),
])

# --- Train ---
t0 = time.time()
pipe.fit(X_train, y_train)
train_secs = time.time() - t0

# --- Predict ---
t0 = time.time()
pred = pipe.predict(X_test)
predict_secs = time.time() - t0

# --- Metrics ---
acc  = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec  = recall_score(y_test, pred)
f1   = f1_score(y_test, pred)

# --- Persist ---
joblib.dump(pipe, "baseline.joblib")
size_mb = Path("baseline.joblib").stat().st_size / (1024 * 1024)

print("=" * 60)
print("STEP 1 — TF-IDF + LogisticRegression baseline")
print("=" * 60)
print(f"Train:               {len(X_train)} reviews")
print(f"Test:                {len(X_test)} reviews")
print()
print(f"Accuracy:            {acc:.4f}")
print(f"Precision:           {prec:.4f}")
print(f"Recall:              {rec:.4f}")
print(f"F1:                  {f1:.4f}")
print()
print(f"Train time:          {train_secs:.2f} s")
print(f"Predict time (25k):  {predict_secs:.2f} s")
print(f"Model file size:     {size_mb:.2f} MB")
print()
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, pred))