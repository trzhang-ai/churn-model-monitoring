"""Discover, combine, de-duplicate, and record configured CSV inputs."""

import json
import os
from glob import glob
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PROJECT_CONFIG_PATH", PROJECT_ROOT / "config.json"))

with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = json.load(config_file)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


input_folder_path = _project_path(config["input_folder_path"])
output_folder_path = _project_path(config["output_folder_path"])


def merge_multiple_dataframe():
    """Merge every configured CSV, remove duplicate rows, and record inputs."""
    source_paths = sorted(Path(path) for path in glob(str(input_folder_path / "*.csv")))
    if not source_paths:
        raise ValueError(f"No CSV files were found in {input_folder_path}")

    dataframes = [pd.read_csv(path) for path in source_paths]
    expected_columns = dataframes[0].columns
    if not all(frame.columns.equals(expected_columns) for frame in dataframes[1:]):
        raise ValueError("Input CSV columns or column order do not match")

    output_dataframe = pd.concat(dataframes, axis=0, ignore_index=True).drop_duplicates()
    output_folder_path.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_folder_path / "finaldata.csv", index=False)

    recorded_paths = []
    for path in source_paths:
        try:
            recorded_paths.append(path.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            recorded_paths.append(str(path))
    (output_folder_path / "ingestedfiles.txt").write_text(
        ",".join(recorded_paths),
        encoding="utf-8",
    )
    return output_dataframe


if __name__ == "__main__":
    merge_multiple_dataframe()
