# IMDB Sentiment Analyzer

🔗 **Live demo:** https://capstone-project-sk-sentiment-ui.vercel.app

🔗 **API:** https://capstone-project-sk-sentiment-api.onrender.com

🔗 **API docs:** https://capstone-project-sk-sentiment-api.onrender.com/docs


**Stack:** TF-IDF + LogisticRegression (F1 0.90) · FastAPI on Render · React (Vite) on Vercel

> ⚠️ Free tier: backend spins down after 15 min of inactivity. First request may take ~50s.

---


# IMDB Sentiment Analyzer — v1.0

An end-to-end Machine Learning web application that predicts the sentiment of movie reviews. 

## Architecture
React (Vite) -> Express Gateway (Port 5000) -> FastAPI (Port 5001) -> Scikit-Learn Model

## Model Comparison & Selection

| Model | Algorithm | F1 Score | Accuracy | Disk Size | Latency (Inference) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Winner)** | TF-IDF + Logistic Regression | 0.9005 | 0.9002 | ~1 MB | < 1 ms |
| Transformer | DistilBERT | ~0.93 | ~0.92 | ~250 MB | ~50 ms |

**Justification:**
The baseline TF-IDF + Logistic Regression model gave up roughly 3-5 points of F1 for a massive reduction in disk size (1MB vs 250MB) and inference latency (sub-millisecond vs 50ms). For this use case, the baseline wins because the near-instant response time and tiny deployment footprint far outweigh the marginal accuracy gain of the transformer. The decision was made to serve the baseline in production, keeping the DistilBERT pipeline as a research artifact.

## How to Run

### 1. Start the FastAPI Service (Terminal 1)
```powershell
.\.venv\Scripts\python.exe api\main.py