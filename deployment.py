"""Copy the current model, score, and ingestion record into production."""

import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


model_path = _project_path(config["output_model_path"]) / "trainedmodel.pkl"
score_record_path = _project_path(config["output_model_path"]) / "latestscore.txt"
ingest_file_path = _project_path(config["output_folder_path"]) / "ingestedfiles.txt"
prod_deployment_path = _project_path(config["prod_deployment_path"])


def deploy_artifacts():
    """Copy the model, score, and ingestion manifest to the deployment directory."""
    prod_deployment_path.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for source_path in (model_path, score_record_path, ingest_file_path):
        copied_files.append(shutil.copy2(source_path, prod_deployment_path))
    return copied_files


def store_model_into_pickle():
    """Compatibility entry point for the original project interface."""
    return deploy_artifacts()


if __name__ == "__main__":
    deploy_artifacts()
