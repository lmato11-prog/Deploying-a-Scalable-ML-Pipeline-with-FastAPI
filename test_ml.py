import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def sample_data():
    """Return a small synthetic binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.random((50, 4))
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def test_train_model_returns_random_forest(sample_data):
    """train_model should return a fitted RandomForestClassifier."""
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    # A fitted model exposes the predict method
    assert hasattr(model, "predict")


def test_compute_model_metrics_returns_expected_values():
    """compute_model_metrics should return precision, recall, fbeta as floats."""
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    # All metrics must be valid probabilities between 0 and 1
    for metric in (precision, recall, fbeta):
        assert 0.0 <= metric <= 1.0


def test_inference_returns_correct_shape(sample_data):
    """inference should return one prediction per input row."""
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    # Predictions for binary classification should only be 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})
