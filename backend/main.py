import re
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text as sql_text

# ---- Load model + vectorizer once at startup ----
model = joblib.load("../ml/model.pkl")
vectorizer = joblib.load("../ml/vectorizer.pkl")

# ---- Database connection ----
DB_USER = "root"
DB_PASSWORD = "root"   # replace with your actual MySQL password
DB_HOST = "localhost"
DB_NAME = "fake_news_db"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---- Request/response schemas ----
class NewsInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label: str
    confidence: float

# ---- App setup ----
app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: NewsInput):
    cleaned = clean_text(input.text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "Real" if pred == 1 else "Fake"
    confidence = round(max(prob) * 100, 2)

    # ---- Log to MySQL ----
    with engine.connect() as conn:
        conn.execute(
            sql_text("INSERT INTO predictions (input_text, label, confidence) VALUES (:text, :label, :confidence)"),
            {"text": input.text, "label": label, "confidence": confidence}
        )
        conn.commit()

    return PredictionOutput(label=label, confidence=confidence)