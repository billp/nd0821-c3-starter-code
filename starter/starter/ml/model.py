import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_model(obj, path):
    """Save a model or encoder to disk using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path):
    """Load a model or encoder from disk using pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_slice_metrics(df, feature, y, preds):
    """
    Compute model metrics for each unique value of a categorical feature.

    Inputs
    ------
    df : pd.DataFrame
        Original test dataframe (before encoding), index-aligned with y
        and preds.
    feature : str
        Name of the categorical feature to slice on.
    y : np.ndarray
        True labels (binarized).
    preds : np.ndarray
        Model predictions (binarized).
    Returns
    -------
    results : list[dict]
        Each dict has keys: feature, value, precision, recall, fbeta,
        count.
    """
    results = []
    for value in sorted(df[feature].unique()):
        mask = (df[feature] == value).values
        if mask.sum() == 0:
            continue
        precision, recall, fbeta = compute_model_metrics(
            y[mask], preds[mask]
        )
        results.append({
            "feature": feature,
            "value": value,
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
            "count": int(mask.sum()),
        })
    return results
