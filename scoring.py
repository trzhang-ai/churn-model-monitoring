"""Score a saved model with F1 and optionally persist the result."""

import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn import metrics


FEATURES = [
    "lastmonth_activity",
    "lastyear_activity",
    "number_of_employees",
]
TARGET = "exited"
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def score_model(
    model_path=None,
    dataset_path=None,
    score_record_path=None,
    write_f1_score=True,
):
    """Return the F1 score for a saved model and configured evaluation data."""
    model_path = Path(model_path) if model_path else (
        _project_path(config["output_model_path"]) / "trainedmodel.pkl"
    )
    dataset_path = Path(dataset_path) if dataset_path else (
        _project_path(config["test_data_path"]) / "testdata.csv"
    )
    score_record_path = Path(score_record_path) if score_record_path else (
        _project_path(config["output_model_path"]) / "latestscore.txt"
    )

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    dataframe = pd.read_csv(dataset_path, usecols=FEATURES + [TARGET])
    predictions = model.predict(dataframe[FEATURES])
    f1_score = float(
        metrics.f1_score(dataframe[TARGET], predictions, zero_division=0)
    )

    if write_f1_score:
        score_record_path.parent.mkdir(parents=True, exist_ok=True)
        score_record_path.write_text(str(f1_score), encoding="utf-8")
    return f1_score


if __name__ == "__main__":
    score_model()
