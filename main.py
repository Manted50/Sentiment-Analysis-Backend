from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
import joblib
from lime.lime_text import LimeTextExplainer

# On créer l'application
app = FastAPI(title="Analyseur de sentiment")

# Chargement des modèles
try:
    model = joblib.load("./artifacts/sentiment_model.joblib")
    vectorizer = joblib.load("./artifacts/tfidf_vectorizer.joblib")
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
        label_to_explain = 1 if sentiment == "positive" else 0

        # Récupère uniquement l'explication pour la classe prédite
        important_words = [
            {"word": word, "importance": float(importance)}
            for word, importance in explanation.as_list(label=label_to_explain)
        ]

        # Génère l'HTML pour toutes les classes, puis filtre via JS (solution robuste)
        # Ou bien, on génère un HTML personnalisé en Python
        exp = explanation.as_list(label=label_to_explain)
        html_expl = f"""
        <div style="text-align: center; margin-bottom: 10px;">
            <h3>Explication pour le sentiment {sentiment.upper()}</h3>
        </div>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="text-align: left; padding: 8px;">Mot</th>
                    <th style="text-align: right; padding: 8px;">Importance</th>
                </tr>
            </thead>
            <tbody>
        """
        for word, importance in exp:
            color = "green" if importance > 0 else "red"
            html_expl += f"""
                <tr>
                    <td style="text-align: left; padding: 8px;"><b>{word}</b></td>
                    <td style="text-align: right; padding: 8px; color: {color};">{importance:.4f}</td>
                </tr>
            """
        html_expl += """
            </tbody>
        </table>
        """

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