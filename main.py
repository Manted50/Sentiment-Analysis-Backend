from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
from lime.lime_text import LimeTextExplainer

app = FastAPI(title="Sentiment Analyzer")

# Load model, vectorizer, and LIME
# model = joblib.load("sentiment_model.joblib")
# vectorizer = joblib.load("tfidf_vectorizer.joblib")
# lime_explainer = LimeTextExplainer(class_names=["negative", "positive"])

# Classes Pydantic
class TweetRequest(BaseModel):
    text: str = Field(..., max_length=280)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probability_positive: float
    probability_negative: float

class ExplanationResponse(BaseModel):
    sentiment: str
    explanation: List[Dict]  # words + importances
    html_explanation: str    # LIME HTML

# Endpoints
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
            confidence=confidence,
            probability_positive=probabilities[1],
            probability_negative=probabilities[0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
def explain_sentiment_with_lime(input: TweetRequest):
    text = input.text
    try:
        explanation = lime_explainer.explain_instance(
            text,
            model.predict_proba,
            num_features=10,
            labels=[0, 1]
        )

        probs = explanation.predict_proba
        sentiment = "positive" if probs[1] > probs[0] else "negative"
        important_words = [
            {"word": word, "importance": importance}
            for word, importance in explanation.as_list(label=1 if sentiment == "positive" else 0)
        ]

        html_expl = explanation.as_html()

        return ExplanationResponse(
            sentiment=sentiment,
            explanation=important_words,
            html_explanation=html_expl
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during explanation: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models": ["sentiment_model_v1", "explanation_model_v1"]}
