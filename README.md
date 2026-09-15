# Churn Model Monitoring

This repository contains an end-to-end monitoring workflow for a client-churn classifier. The system ingests new data, evaluates the deployed model, exposes operational diagnostics, and retrains only when F1 performance declines.

The trigger compares labeled-batch F1 with a stored baseline. It monitors
predictive performance; it does not independently test feature-distribution
shift or establish concept drift. The legacy response status
`new_data_no_drift` means that F1 did not fall below the stored baseline.

## Architecture

1. `ingestion.py` discovers configured CSV batches, validates their schema, removes duplicate records, and writes an ingestion manifest.
2. `scoring.py` evaluates a persisted logistic-regression model.
3. `fullprocess.py` compares newly ingested performance with the deployed score and performs a conditional retraining and deployment.
4. `app.py` exposes predictions, model scoring, summary statistics, and runtime diagnostics over HTTP.
5. `reporting.py` produces a confusion matrix for the deployed model.

## Operation

Install the pinned dependencies, then start the API:

```bash
python app.py
```

From a second terminal, collect the endpoint responses:

```bash
python apicalls.py
```

Run the complete ingestion and performance-evaluation workflow with:

```bash
python fullprocess.py
```

`cronjob.txt` contains the ten-minute schedule used in the target workspace.

The bundled Flask server is intended for local or controlled workspace use. Add authentication and restrict file access before exposing it to an untrusted network.

## Archived local deployment state

The configured source batches score `0.6923076923` with the deployed model, above the stored deployment score of `0.5714285714`. The automation therefore preserves the existing deployment. The numbered API and confusion-matrix files are second snapshots of that unchanged deployment.

Diagnostics return mean, median, and mode for numeric columns, numeric-column missing percentages, execution timings, and installed/latest direct-dependency versions.

## Provenance and use

This private archive contains an implementation developed from Udacity-provided starter materials for project `cd0583`. Udacity course and starter materials remain subject to the included `LICENSE` and the Udacity Terms of Use. The archive is retained for personal learning and assessment provenance and is not intended for redistribution or use as another learner's submission.
