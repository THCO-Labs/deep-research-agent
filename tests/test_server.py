from fastapi.testclient import TestClient
from deep_research.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "runs_dir" in data

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
