"""Call the running project API and persist its combined responses."""

import argparse
import json
import os
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))
URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def call_api(output_filename="apireturns.txt"):
    """Call all four HTTP endpoints and write their JSON responses."""
    test_data_path = _project_path(config["test_data_path"]) / "testdata.csv"
    with test_data_path.open("rb") as csv_file:
        prediction_response = requests.post(
            f"{URL}/prediction",
            files={"file": (test_data_path.name, csv_file, "text/csv")},
            timeout=30,
        )

    scoring_response = requests.get(f"{URL}/scoring", timeout=30)
    summary_response = requests.get(f"{URL}/summarystats", timeout=30)
    diagnostics_response = requests.get(f"{URL}/diagnostics", timeout=180)
    responses = [
        prediction_response,
        scoring_response,
        summary_response,
        diagnostics_response,
    ]
    for response in responses:
        response.raise_for_status()

    combined_responses = {
        "/prediction": prediction_response.json(),
        "/scoring": scoring_response.json(),
        "/summarystats": summary_response.json(),
        "/diagnostics": diagnostics_response.json(),
    }
    output_path = _project_path(config["output_model_path"]) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined_responses, indent=2),
        encoding="utf-8",
    )
    return combined_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_filename", nargs="?", default="apireturns.txt")
    arguments = parser.parse_args()
    call_api(arguments.output_filename)


if __name__ == "__main__":
    main()
