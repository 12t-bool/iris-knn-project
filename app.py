
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import Counter
import os

app = Flask(__name__)

from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent / "IRIS.csv")

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

X = df[features].to_numpy(dtype=float)
y = df["species"].to_numpy()

# Standardize measurements so each feature
# contributes fairly to distance calculations
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1

X_scaled = (X - mean) / std


# Manual KNN algorithm (no KNN library)
def knn_predict(values, k=5):
    values = np.array(values, dtype=float)
    scaled_values = (values - mean) / std

    # Calculate Euclidean distances
    distances = np.sqrt(
        np.sum((X_scaled - scaled_values) ** 2, axis=1)
    )

    # Find the k nearest flowers
    nearest_indices = np.argsort(distances)[:k]
    nearest_species = y[nearest_indices]

    # Majority voting
    return Counter(nearest_species).most_common(1)[0][0]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"])
            ]

            if not np.all(np.isfinite(values)):
                raise ValueError

            if any(value <= 0 or value > 10 for value in values):
                raise ValueError

            prediction = knn_predict(values)

        except (ValueError, KeyError):
            error = "Enter valid measurements between 0 and 10."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        port=int(os.environ.get("PORT", 5000))
    )
