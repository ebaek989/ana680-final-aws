
import io
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, request, jsonify

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bundle not found at {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
model = bundle.get("model", bundle)
feature_columns = bundle.get("feature_columns")

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check: SageMaker calls this to see if the container is alive."""
    health = model is not None
    status = 200 if health else 500
    return Response(response="\n", status=status, mimetype="application/json")


def _json_to_dataframe(payload):
    """Convert a JSON payload to a pandas DataFrame."""
    if isinstance(payload, dict) and "instances" in payload:
        data = np.array(payload["instances"])
    else:
        data = np.array(payload)
    df = pd.DataFrame(data)
    if feature_columns is not None:
        if df.shape[1] != len(feature_columns):
            raise ValueError(
                f"Expected {len(feature_columns)} features, got {df.shape[1]}"
            )
        df.columns = feature_columns
    return df


def _csv_to_dataframe(raw_text):
    """Convert CSV text payload to a pandas DataFrame."""
    df = pd.read_csv(io.StringIO(raw_text), header=None)
    if feature_columns is not None:
        if df.shape[1] != len(feature_columns):
            raise ValueError(
                f"Expected {len(feature_columns)} features, got {df.shape[1]}"
            )
        df.columns = feature_columns
    return df


@app.route("/invocations", methods=["POST"])
def invocations():
    """Main inference endpoint."""
    if request.content_type == "application/json":
        payload = request.get_json()
        df = _json_to_dataframe(payload)
    elif request.content_type in ("text/csv", "text/plain"):
        data = request.data.decode("utf-8")
        df = _csv_to_dataframe(data)
    else:
        return Response(
            response=f"Unsupported content type: {request.content_type}",
            status=415,
            mimetype="text/plain",
        )

    try:
        preds = model.predict(df.values)
    except Exception as e:
        return Response(
            response=f"Error generating prediction: {e}",
            status=500,
            mimetype="text/plain",
        )

    return jsonify(preds.tolist())


if __name__ == "__main__":
    # Optional local test: python app.py and POST to localhost:8080/invocations
    app.run(host="0.0.0.0", port=8080, debug=True)
