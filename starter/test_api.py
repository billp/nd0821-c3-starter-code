from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_post_predict_leq_50k():
    sample = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_gt_50k():
    sample = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
