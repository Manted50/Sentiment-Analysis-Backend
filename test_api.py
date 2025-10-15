import time
import pytest
from fastapi.testclient import TestClient
from main import app  # Import the FastAPI app from main_api.py

# Initialize the test client
client = TestClient(app)

def test_predict_endpoint_valid():
    """Test the /predict endpoint with valid input"""
    
    # Test data
    test_data = {
        "text": "I love this product, it's amazing!"
    }
    
    # API call
    response = client.post("/predict", json=test_data)
    
    # Critical validations
    assert response.status_code == 200
    
    data = response.json()
    
    # Response structure
    required_fields = [
        "sentiment", "confidence", 
        "probability_positive", "probability_negative"
    ]
    for field in required_fields:
        assert field in data
    
    # Data consistency
    assert data["sentiment"] in ["Positive", "Negative"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_positive"] <= 1.0
    assert 0.0 <= data["probability_negative"] <= 1.0
    
    # Probabilities sum to 1
    total_prob = data["probability_positive"] + data["probability_negative"]
    assert abs(total_prob - 1.0) < 0.01
    
    print(f"✅ Prediction OK: {data['sentiment']} ({data['confidence']:.2f})")

def test_predict_endpoint_invalid():
    """Test the /predict endpoint with invalid input"""
    
    # Test 1: Empty text
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    
    # Test 2: Text too long (> 280 characters)
    long_text = "a" * 300
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422
    
    # Test 3: Invalid JSON format
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    print("✅ Validation of errors OK")

def test_explain_endpoint():
    """Test the /explain endpoint with LIME"""
    
    # Test data
    test_data = {
        "text": "This movie is absolutely terrible, I hate it!"
    }
    
    # Measure time (LIME can be slow)
    start_time = time.time()
    response = client.post("/explain", json=test_data)
    duration = time.time() - start_time
    
    # Critical validations
    assert response.status_code == 200
    
    data = response.json()
    
    # Response structure
    required_fields = [
        "sentiment", "explanation", "html_explanation"
    ]
    for field in required_fields:
        assert field in data
    
    # Explanation validation
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    
    # HTML explanation validation
    html_content = data["html_explanation"]
    assert isinstance(html_content, str)
    assert len(html_content) > 100  # Substantial HTML
    assert "<div" in html_content   # Contains HTML
    
    # Acceptable performance (< 120 seconds)
    assert duration < 120
    
    print(f"✅ LIME OK: {len(data['explanation'])} words explained")
    print(f"⏱️ Time: {duration:.1f}s")

def test_health_endpoint():
    """Test the /health endpoint"""
    
    # API call
    response = client.get("/health")
    
    # Critical validations
    assert response.status_code == 200
    
    data = response.json()
    
    # Response structure
    assert "status" in data
    assert data["status"] == "ok"
    
    print("✅ Health check OK")

def test_ui_homepage():
    """Test the UI homepage"""
    
    # API call to the root endpoint
    response = client.get("/")
    
    # Critical validations
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    
    html_content = response.text
    assert "<title>Sentiment Analysis</title>" in html_content
    assert "Welcome to the Sentiment Analysis App" in html_content
    
    print("✅ UI Homepage OK")

def test_ui_submit_form():
    """Test the UI form submission"""
    
    # Simulate form submission
    form_data = {
        "text": "This is a test input for the form."
    }
    response = client.post("/submit", data=form_data)
    
    # Critical validations
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    
    html_content = response.text
    assert "Sentiment Analysis Result" in html_content
    assert "Positive" in html_content or "Negative" in html_content
    
    print("✅ UI Form Submission OK")

def test_explain_robustness():
    """Test the robustness of LIME with various texts"""
    
    # Test cases
    test_cases = [
        "Super !",  # Very short text
        "😊" * 10,  # Emojis only
        "http://example.com test"  # Text with URLs
    ]
    
    for text in test_cases:
        response = client.post("/explain", json={"text": text})
        
        # Must handle all cases
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            
            # Validate response structure
            assert "html_explanation" in data
            assert isinstance(data["html_explanation"], str)
            assert len(data["html_explanation"]) > 100  # Substantial HTML
            assert "<div" in data["html_explanation"]  # Contains HTML
    
    print("✅ Robustness of LIME OK")

if __name__ == "__main__":
    # Run the tests
    test_predict_endpoint_valid()
    test_predict_endpoint_invalid()
    test_explain_robustness()
    test_explain_endpoint()
    test_health_endpoint()
    test_ui_homepage()
    test_ui_submit_form()
