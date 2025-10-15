import time
import pytest
from fastapi.testclient import TestClient
from main import app

# On récupère le client de test
client = TestClient(app)

# On définit les tests
def test_predict_endpoint_valid():
    """Test the /predict endpoint with valid input."""
    test_data = {"text": "I love this product, it's amazing!"}
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    required_fields = ["sentiment", "confidence", "probability_positive", "probability_negative"]
    for field in required_fields:
        assert field in data
    assert data["sentiment"] in ["positive", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0
    assert abs(data["probability_positive"] + data["probability_negative"] - 1.0) < 0.01

def test_predict_endpoint_invalid():
    """Test the /predict endpoint with invalid input."""
    # Empty text
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    # Too long text
    long_text = "a" * 300
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422
    # Missing text
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_explain_endpoint():
    """Test the /explain endpoint with LIME."""
    test_data = {"text": "This movie is absolutely terrible, I hate it!"}
    start_time = time.time()
    response = client.post("/explain", json=test_data)
    duration = time.time() - start_time
    assert response.status_code == 200
    data = response.json()
    required_fields = ["sentiment", "explanation", "html_explanation"]
    for field in required_fields:
        assert field in data
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    assert isinstance(data["html_explanation"], str)
    assert len(data["html_explanation"]) > 100
    assert "<div" in data["html_explanation"]
    assert duration < 120  # Timeout

def test_health_endpoint():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.parametrize("text", [
    "Super !",
    "😊" * 10,
    "http://example.com test"
])
def test_explain_robustness(text):
    """Test the robustness of LIME with various texts."""
    response = client.post("/explain", json={"text": text})
    assert response.status_code in [200, 422, 400] 
    if response.status_code == 200:
        data = response.json()
        assert "html_explanation" in data
        assert isinstance(data["html_explanation"], str)
        assert len(data["html_explanation"]) > 100
        assert "<div" in data["html_explanation"]
