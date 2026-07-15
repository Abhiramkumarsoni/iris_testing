# This is a simple test for the /predict endpoint of the API.
import pytest
from api import app


def test_predict_endpoint():
    client = app.test_client()
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})

    assert response.status_code == 200
    assert response.json["prediction"] == 0

# pytest test_api.py