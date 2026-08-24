"""Train the configured churn model and save it as a pickle artifact."""

import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


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


dataset_csv_path = _project_path(config["output_folder_path"]) / "finaldata.csv"
model_path = _project_path(config["output_model_path"]) / "trainedmodel.pkl"


def train_model():
    """Fit and persist a logistic regression classifier."""
    dataframe = pd.read_csv(dataset_csv_path, usecols=FEATURES + [TARGET])
    classifier = LogisticRegression(random_state=0, solver="liblinear")
    classifier.fit(dataframe[FEATURES], dataframe[TARGET])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(classifier, model_file)
    return classifier


if __name__ == "__main__":
    train_model()
