"""Model, data, runtime, and dependency diagnostics for the deployed system."""

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


FEATURES = [
    "lastmonth_activity",
    "lastyear_activity",
    "number_of_employees",
]
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _native_values(series):
    return {column: float(value) for column, value in series.items()}


def model_predictions(input_dataframe):
    """Return deployed-model predictions for the supplied DataFrame."""
    missing_features = [name for name in FEATURES if name not in input_dataframe]
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")

    import pickle

    model_path = _project_path(config["prod_deployment_path"]) / "trainedmodel.pkl"
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    return model.predict(input_dataframe[FEATURES]).tolist()


def dataframe_summary(dataset_path=None):
    """Return mean, median, and mode for every numeric column."""
    dataset_path = Path(dataset_path) if dataset_path else (
        _project_path(config["output_folder_path"]) / "finaldata.csv"
    )
    numeric_data = pd.read_csv(dataset_path).select_dtypes(include="number")
    if numeric_data.empty:
        raise ValueError("The dataset has no numeric columns")
    return [
        {"statistic": "mean", "values": _native_values(numeric_data.mean())},
        {"statistic": "median", "values": _native_values(numeric_data.median())},
        {"statistic": "mode", "values": _native_values(numeric_data.mode().iloc[0])},
    ]


def count_na(dataset_path=None):
    """Return missing-value percentages for numeric columns only."""
    dataset_path = Path(dataset_path) if dataset_path else (
        _project_path(config["output_folder_path"]) / "finaldata.csv"
    )
    dataframe = pd.read_csv(dataset_path)
    if dataframe.empty:
        raise ValueError("The dataset is empty")
    numeric_data = dataframe.select_dtypes(include="number")
    return _native_values(numeric_data.isna().mean() * 100)


def missing_data(dataset_path=None):
    """Return missing-value diagnostics."""
    return count_na(dataset_path)


def execution_time():
    """Time ingestion and training separately without changing project artifacts."""
    configured_input = _project_path(config["input_folder_path"])
    configured_test = _project_path(config["test_data_path"])

    with tempfile.TemporaryDirectory(prefix="churn-diagnostics-") as temp_directory:
        temp_root = Path(temp_directory)
        temp_input = temp_root / "input"
        temp_input.mkdir()
        for csv_path in configured_input.glob("*.csv"):
            shutil.copy2(csv_path, temp_input / csv_path.name)

        temp_config = {
            "input_folder_path": str(temp_input),
            "output_folder_path": str(temp_root / "ingested"),
            "test_data_path": str(configured_test),
            "output_model_path": str(temp_root / "models"),
            "prod_deployment_path": str(temp_root / "production"),
        }
        temp_config_path = temp_root / "config.json"
        temp_config_path.write_text(json.dumps(temp_config), encoding="utf-8")

        environment = os.environ.copy()
        environment["PROJECT_CONFIG_PATH"] = str(temp_config_path)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"

        ingestion_start = time.perf_counter()
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "ingestion.py")],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=120,
        )
        ingestion_seconds = time.perf_counter() - ingestion_start

        training_start = time.perf_counter()
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "training.py")],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=120,
        )
        training_seconds = time.perf_counter() - training_start

    return [
        {"data_ingestion_time": float(ingestion_seconds)},
        {"model_training_time": float(training_seconds)},
    ]


def _canonical_package_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements():
    requirements = []
    for raw_line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
        pinned_match = re.search(r"==\s*([^;\s]+)", line)
        requirements.append((name, pinned_match.group(1) if pinned_match else "unknown"))
    return requirements


def outdated_packages_list():
    """Report installed and latest versions for every direct requirement."""
    requirements = _requirements()
    outdated_by_name = {}
    environment = os.environ.copy()
    environment.setdefault("PIP_DEFAULT_TIMEOUT", "5")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "list",
                "--outdated",
                "--format=json",
                "--disable-pip-version-check",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=environment,
        )
        if result.returncode == 0:
            for package in json.loads(result.stdout or "[]"):
                outdated_by_name[_canonical_package_name(package["name"])] = package
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        outdated_by_name = {}

    rows = []
    for requirement_name, pinned_version in requirements:
        try:
            installed_version = importlib.metadata.version(requirement_name)
        except importlib.metadata.PackageNotFoundError:
            installed_version = "not installed"
        outdated = outdated_by_name.get(_canonical_package_name(requirement_name))
        if outdated:
            latest_version = outdated["latest_version"]
        elif installed_version != "not installed":
            latest_version = installed_version
        else:
            latest_version = pinned_version
        rows.append(
            {
                "module_name": requirement_name,
                "installed_version": installed_version,
                "latest_version": latest_version,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["module_name", "installed_version", "latest_version"],
    )


if __name__ == "__main__":
    final_data_path = _project_path(config["output_folder_path"]) / "finaldata.csv"
    print("summary_statistics=", dataframe_summary(final_data_path))
    print("missing_data_percentages=", count_na(final_data_path))
    print("execution_times=", execution_time())
    print(outdated_packages_list().to_string(index=False))
