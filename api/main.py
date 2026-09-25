# Day 11: FastAPI service for the IMDB sentiment baseline.
# Loads baseline.joblib (TF-IDF + LogReg Pipeline) at startup.
# Endpoint: POST /sentiment { text } -> { label, score, model }
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "api" / ".env")

PORT = int(os.getenv("PORT", "5001"))
CLIENT_URLS = [u.strip() for u in os.getenv(
    "CLIENT_URLS", "http://localhost:3000,http://localhost:5173,http://localhost:5174"
).split(",")]

MODEL_PATH = ROOT / "baseline.joblib"
MODEL = None
MODEL_META = {}


def err(status, msg):
    return JSONResponse(status_code=status, content={"success": False, "error": msg})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = joblib.load(MODEL_PATH)
    MODEL_META.update({
        "model_name": "imdb-sentiment-tfidf-logreg",
        "version":    "1.0",
        "algorithm":  "TfidfVectorizer + LogisticRegression",
        "f1":         0.9005,
        "accuracy":   0.9002,
    })

    print("=" * 52)
    print(" IMDB SENTIMENT API (PYTHON/FASTAPI)  v1.0")
    print("=" * 52)
    print(f" Port: {PORT}     URL: http://localhost:{PORT}")
    print(f" Model loaded ✅  ({MODEL_META['algorithm']})")
    print(f" Test F1:         {MODEL_META['f1']}    Accuracy: {MODEL_META['accuracy']}")
    print(" Endpoints:")
    print("   GET  /health      service + model status")
    print("   POST /sentiment   { text } -> { label, score, model }")
    print("=" * 52)
    yield


app = FastAPI(title="imdb-sentiment-api", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CLIENT_URLS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return err(404, f"Route not found: {request.method} {request.url.path}")


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


@app.get("/health")
async def health():
    return {
        "success": True,
        "service": "imdb-sentiment-api",
        "version": "1.0",
        "model_loaded": MODEL is not None,
        "model": MODEL_META,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/sentiment")
async def sentiment(body: TextIn):
    # Pipeline.predict_proba -> [[p_neg, p_pos]]
    proba = float(MODEL.predict_proba([body.text])[0][1])   # probability of positive
    label = "positive" if proba >= 0.5 else "negative"
    # score = confidence in the chosen label
    score = proba if label == "positive" else 1 - proba

    return {
        "success": True,
        "data": {
            "label": label,
            "score": round(score, 4),
            "p_positive": round(proba, 4),
            "model": MODEL_META["model_name"],
            "version": MODEL_META["version"],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)