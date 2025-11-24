import os
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, request, jsonify

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    try:
        _ = model  # just ensure model is loaded
        return Response(response="OK", status=200)
    except Exception as e:
        return Response(response=str(e), status=500)


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Inference endpoint.

    Expects:
      - Content-Type: text/csv
      - Body: CSV data without header, columns in the same order as training.
    Returns:
      - JSON list of predictions.
    """
    if request.content_type != "text/csv":
        return Response(
            response="This predictor only supports text/csv input.",
            status=415,
        )

    data = request.data.decode("utf-8")
    if not data.strip():
        return Response(response="Empty request body.", status=400)

    # We don't know the column names here, but the order must match training
    df = pd.read_csv(io.StringIO(data), header=None)

    try:
        preds = model.predict(df.values)
    except Exception as e:
        return Response(response=f"Error generating prediction: {e}", status=500)

    preds_list = preds.tolist()
    return jsonify(preds_list)


if __name__ == "__main__":
    # For local testing; in SageMaker we use gunicorn via `serve`
    app.run(host="0.0.0.0", port=8080, debug=False)
