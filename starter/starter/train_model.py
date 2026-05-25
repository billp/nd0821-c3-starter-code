# Script to train machine learning model.
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_model,
    train_model,
)

# Load data.
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "data", "census.csv"),
    skipinitialspace=True,
)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)

# Evaluate on test set.
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F-beta: {fbeta:.4f}")

# Save model artifacts.
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)
save_model(model, os.path.join(model_dir, "trained_model.pkl"))
save_model(encoder, os.path.join(model_dir, "encoder.pkl"))
save_model(lb, os.path.join(model_dir, "lb.pkl"))

# Compute and save slice metrics.
test_df = test.reset_index(drop=True)
output_path = os.path.join(os.path.dirname(__file__), "..", "slice_output.txt")
with open(output_path, "w") as f:
    for feature in cat_features:
        results = compute_slice_metrics(test_df, feature, y_test, preds)
        for row in results:
            line = (
                f"{row['feature']}: {row['value']} | "
                f"Precision: {row['precision']:.4f} | "
                f"Recall: {row['recall']:.4f} | "
                f"F-beta: {row['fbeta']:.4f} | "
                f"Count: {row['count']}"
            )
            f.write(line + "\n")
            print(line)
