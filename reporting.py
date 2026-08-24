"""Generate a confusion-matrix image for the deployed model."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from diagnostics import model_predictions


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def score_model(dataset_path=None, cm_plot_path=None):
    """Generate and save a confusion matrix for the deployed model."""
    dataset_path = Path(dataset_path) if dataset_path else (
        _project_path(config["test_data_path"]) / "testdata.csv"
    )
    cm_plot_path = Path(cm_plot_path) if cm_plot_path else (
        _project_path(config["output_model_path"]) / "confusionmatrix.png"
    )

    dataframe = pd.read_csv(dataset_path)
    predictions = model_predictions(dataframe)
    display = ConfusionMatrixDisplay.from_predictions(
        dataframe["exited"],
        predictions,
    )
    cm_plot_path.parent.mkdir(parents=True, exist_ok=True)
    display.figure_.savefig(cm_plot_path, bbox_inches="tight")
    plt.close(display.figure_)
    return str(cm_plot_path)


if __name__ == "__main__":
    score_model()
