from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
import joblib
from lime.lime_text import LimeTextExplainer

# On créer l'application
app = FastAPI(title="Analyseur de sentiment")

# Chargement des modèles
try:
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    lime_explainer = LimeTextExplainer(class_names=["negative", "positive"])
except Exception as e:
    raise RuntimeError(f"Failed to load model or vectorizer: {str(e)}")

# Modèle Pydantic avec validation stricte
class TweetRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=280)

    @field_validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float

class ExplanationResponse(BaseModel):
    sentiment: str
    explanation: List[Dict]
    html_explanation: str

# On définit les endpoints
@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment_with_model(input: TweetRequest):
    text = input.text
    try:
        text_vectorized = vectorizer.transform([text])
        probabilities = model.predict_proba(text_vectorized)[0]
        sentiment = "positive" if probabilities[1] > probabilities[0] else "negative"
        confidence = max(probabilities)
        return PredictionResponse(
            sentiment=sentiment,
            confidence=float(confidence),
            probability_positive=float(probabilities[1]),
            probability_negative=float(probabilities[0])
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
def explain_sentiment_with_lime(input: TweetRequest):
    text = input.text
    try:
        explanation = lime_explainer.explain_instance(
            text,
            lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=10,
            labels=[0, 1]
        )
        probabilities = model.predict_proba(vectorizer.transform([text]))[0]
        sentiment = "positive" if probabilities[1] > probabilities[0] else "negative"
        important_words = [
            {"word": word, "importance": float(importance)}
            for word, importance in explanation.as_list(label=1 if sentiment == "positive" else 0)
        ]
        html_expl = explanation.as_html()
        return ExplanationResponse(
            sentiment=sentiment,
            explanation=important_words,
            html_explanation=html_expl
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during explanation: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models": ["sentiment_model_v1", "explanation_model_v1"]}