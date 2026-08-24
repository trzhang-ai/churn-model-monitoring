"""Flask API for predictions, scoring, and operational diagnostics."""

import json
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from diagnostics import (
    count_na,
    dataframe_summary,
    execution_time,
    model_predictions,
    outdated_packages_list,
)
from scoring import score_model


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


app = Flask(__name__)


def _prediction_dataframe():
    uploaded_file = request.files.get("file")
    if uploaded_file is not None and uploaded_file.filename:
        return pd.read_csv(uploaded_file)

    request_data = request.get_json(silent=True) or {}
    filepath = request_data.get("filepath")
    if filepath:
        input_path = Path(filepath)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        return pd.read_csv(input_path)
    raise ValueError("Upload a CSV file or provide a JSON filepath")


@app.route("/prediction", methods=["POST"])
def predict():
    try:
        predictions = model_predictions(_prediction_dataframe())
    except (OSError, ValueError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"predictions": predictions})


@app.route("/scoring", methods=["GET"])
def get_f1():
    f1_score = score_model(
        model_path=_project_path(config["prod_deployment_path"]) / "trainedmodel.pkl",
        dataset_path=_project_path(config["test_data_path"]) / "testdata.csv",
        write_f1_score=False,
    )
    return jsonify(float(f1_score))


@app.route("/summarystats", methods=["GET"])
def stats():
    return jsonify(
        dataframe_summary(
            _project_path(config["output_folder_path"]) / "finaldata.csv"
        )
    )


@app.route("/diagnostics", methods=["GET"])
def diagnose_model():
    final_data_path = _project_path(config["output_folder_path"]) / "finaldata.csv"
    return jsonify(
        {
            "timing": execution_time(),
            "missing_data": count_na(final_data_path),
            "dependencies_check": outdated_packages_list().to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
