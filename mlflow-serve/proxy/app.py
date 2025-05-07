import os
import sys

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:8888")


@app.route("/invocations", methods=["POST"])
def invocations():
    data = request.get_json()
    print("Received data:", data, file=sys.stderr)

    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{MLFLOW_URL}/invocations",
            json=data,  # Forward the raw input format expected by MLflow
            headers=headers,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}", file=sys.stderr)
        print("Response content:", response.text, file=sys.stderr)
        return jsonify({"error": str(e), "raw_response": response.text}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8890)
