import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import compute_model_metrics, inference, train_model


def test_train_model_returns_random_forest():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_ndarray_with_correct_shape():
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
    assert pytest.approx(precision, abs=1e-4) == 2 / 3
    assert pytest.approx(recall, abs=1e-4) == 2 / 3
