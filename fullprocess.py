"""Run data ingestion, drift detection, and conditional redeployment."""

import json
import os
import socket
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

from deployment import store_model_into_pickle
from ingestion import merge_multiple_dataframe
from reporting import score_model as report_model
from scoring import score_model
from training import train_model


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_new_files():
    """Compare configured CSV names with the deployed ingestion record."""
    record_path = _project_path(config["prod_deployment_path"]) / "ingestedfiles.txt"
    if record_path.exists():
        recorded_files = {
            Path(item).name
            for item in record_path.read_text(encoding="utf-8").strip().split(",")
            if item.strip()
        }
    else:
        recorded_files = set()
    source_files = {
        Path(path).name
        for path in glob(str(_project_path(config["input_folder_path"]) / "*.csv"))
    }
    return sorted(source_files - recorded_files)


def _wait_for_api(server, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("The API process exited before becoming ready")
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("The API did not become ready on port 8000")


def run_api_calls(apicalls_return_filename):
    """Start the API, collect endpoint responses, and stop the server."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    server = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "app.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_api(server)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "apicalls.py"),
                apicalls_return_filename,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            timeout=240,
        )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def main():
    new_files = find_new_files()
    if not new_files:
        return {"status": "no_new_data", "new_files": []}

    merge_multiple_dataframe()
    deployed_score_path = (
        _project_path(config["prod_deployment_path"]) / "latestscore.txt"
    )
    deployed_score = float(deployed_score_path.read_text(encoding="utf-8"))
    new_data_score = score_model(
        model_path=_project_path(config["prod_deployment_path"]) / "trainedmodel.pkl",
        dataset_path=_project_path(config["output_folder_path"]) / "finaldata.csv",
        write_f1_score=False,
    )

    if new_data_score >= deployed_score:
        return {
            "status": "new_data_no_drift",
            "new_files": new_files,
            "deployed_score": deployed_score,
            "new_data_score": new_data_score,
        }

    train_model()
    score_model()
    store_model_into_pickle()
    report_model(
        cm_plot_path=(
            _project_path(config["output_model_path"]) / "confusionmatrix2.png"
        )
    )
    run_api_calls("apireturns2.txt")
    return {
        "status": "model_redeployed",
        "new_files": new_files,
        "deployed_score": deployed_score,
        "new_data_score": new_data_score,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
