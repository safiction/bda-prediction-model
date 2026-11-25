# BDA project implementation plan

## 1. Goal and agreed scope

Build a reproducible application that predicts whether a recently active subscriber will have **no CDR activity during the next 14 days**.

The application will replay the provided 90-day CDR and tickets dataset in batches. It will clean incoming records, store processed data in Parquet, build historical features, train and version models, display pipeline progress and quality metrics, and provide individual predictions through a browser interface.

This target is a **14-day inactivity proxy for churn**, not verified contract cancellation. Use this wording in the interface, README, and presentation.

### Included

- Reuse and adapt homework code rather than rewrite the analytical logic unnecessarily.
- CDR and tickets as the modeling sources.
- Parquet storage, Spark processing, subscriber-level features, and batch retraining.
- A logistic regression baseline and an XGBoost candidate; Random Forest is an optional homework comparison.
- A Streamlit interface and a Docker-based local launch.
- Explicit historical dates, delayed outcome evaluation, model artifacts, logs, and reproducible runs.

### Excluded from the first version

- `retention_contact`, unrestricted ticket text, and any outcome-derived fields as model inputs.
- Telemetry, verified cancellation prediction, true online learning, and automatic claims of improved quality after every batch.
- Separate frontend/API services, a distributed production cluster, and a mandatory Airflow deployment.

Python notebook comments must be in Russian. Preserve the simple style used in the homework. Extract small functions where they improve reuse; avoid unnecessary class hierarchies.

## 2. Target and time conventions

Use human-readable day numbers 1–90 relative to the observed dataset start. Discover actual dates from the allowed data, validate the coverage, and persist the resulting calendar. Do not depend on a removed reference file.

For a prediction date **D**:

| Element | Definition |
|---|---|
| Feature window | The 14 calendar days D−13 through D, inclusive |
| Eligible population | Subscribers with at least one valid CDR event in that feature window |
| Outcome window | The 14 calendar days D+1 through D+14, inclusive |
| Target 1 | No valid CDR events for that subscriber in the complete outcome window |
| Target 0 | At least one valid CDR event in the complete outcome window |
| Unknown target | The complete outcome window is not available or required CDR partitions are missing |
| Example key | `(subscriber_key, snapshot_date)` |

A valid CDR event is a deduplicated record with a usable subscriber key and event timestamp. A zero-duration call or zero charge alone does not invalidate activity. Numeric defects may affect particular features without erasing the event from activity counts.

Use explicit date boundaries, consistently parsed timestamps, and half-open timestamp filters in code. Never use the computer's current date to construct historical features.

An event after D must never contribute to features at D. Future events are used only by the separate label-building step. Unknown labels must never be filled with zero or one.

The same subscriber can occur at several training snapshots. Keep each snapshot unique, preserve chronological separation, and report that observations from the same person are correlated. Generalization is to future activity of the subscriber population; this is not a claim of performance on entirely unseen people.

## 3. Replay and evaluation design

### Default schedule

| Replay day | Operation | Training snapshots | Validation snapshot and outcomes |
|---|---|---|---|
| 60 | Initial training and validation | 14, 21, 28 | Features through 46; outcomes 47–60 |
| 67 | Ingest days 61–67; train candidate | 14, 21, 28, 35 | Features through 53; outcomes 54–67 |
| 74 | Ingest days 68–74; train candidate | 14, 21, 28, 35, 42 | Features through 60; outcomes 61–74 |
| 76 | Ingest days 75–76; freeze final predictions | No new training | Save predictions from all completed model versions at snapshot 76 |
| 90 | Ingest remaining days and reveal final outcomes | No training before final evaluation | Evaluate frozen day-76 predictions against outcomes 77–90 |

At replay day T, training snapshot dates must be at most T−28. Their labels finish by T−14. The validation snapshot is T−14 and its outcomes finish at T. Generate training snapshots every seven days beginning at day 14, subject to these constraints.

**Agreed after the data audit:** keep the snapshot dates and 14-day event windows above, but allow one additional day for record delivery (`label_delivery_lag_days = 1`). Processing/evaluation for milestones 60, 67, 74, and 90 can finish at delivery-clock days 61, 68, 75, and 91. Prediction freezing remains at day 76. Day 91 requires no new event partition: it only exposes existing day-77–90 records whose `ingest_ts` falls on that date. Drop malformed CSV rows from processed outputs and report their counts (`malformed_csv_policy = "drop"`); preserve raw files.

This conservative schedule keeps training outcome windows from extending into the validation outcome period. Overlapping windows within training are allowed and must not be mistaken for independent samples.

### Model selection and comparison

1. Fit preprocessing and model parameters using training data only.
2. Evaluate baseline and candidate on the same current validation snapshot.
3. If an incumbent model exists, evaluate it on that same snapshot before deciding whether to replace it.
4. Use validation PR-AUC as the initial selection metric, with log loss and recall shown alongside it. Record this rule in configuration; do not select using the final test.
5. Keep the incumbent when the candidate does not satisfy the selection rule. A failed training job must not replace a working model.
6. Validation windows move between runs. Display their dates and do not present their scores as a controlled comparison on identical data.
7. At day 76, apply every completed version to the same feature table and save the probabilities, threshold, version, and subscriber IDs.
8. At day 90, evaluate those frozen predictions on the shared final outcome window. This is the direct final comparison. Do not tune on these outcomes.

The active model is chosen before the final outcomes are revealed. Later retraining beyond this demonstration requires a new, untouched evaluation period.

### Feasibility gate

Before committing to model training, measure eligibility, label balance, and labeled row counts for all scheduled snapshots. Both classes must be present in every training and evaluation set used for classification metrics.

Do not promise ROC-AUC above a chosen value or steadily falling loss. If the inactivity target is too rare or unsupported, document the evidence and revisit the agreed target/window before proceeding. Do not silently redefine activity or use forbidden labels to improve scores.

## 4. Data arrival and quality rules

The replay clock, not file visibility on disk, determines what the application can access at each stage. Files for future replay days may already exist locally but must remain unavailable to the pipeline.

For the first version, use a **complete-day replay**: discover the daily partitions under both `raw/` and `late_delivery/`, and expose each historical day's available files together when that day is ingested. This is a declared simulation assumption, not a reconstruction of actual delivery delays. Respect trustworthy record-level availability timestamps where provided. Do not invent delivery timestamps that are absent.

- Inventory filenames, date partitions, schemas, encodings, and sizes from allowed sources.
- Validate CDR and ticket date coverage separately. Only mark a day complete after the required files have been processed successfully.
- Fail or pause label construction when an outcome day is incomplete. Missing files do not mean subscriber inactivity.
- Reuse the homework's encoding fallback, category normalization, type conversion, deduplication, and defect reporting where applicable.
- Preserve raw files. Write processed results to separate paths.
- Deduplicate CDR by its record identifier using deterministic rules. Report conflicting duplicate versions.
- Handle repeated ticket IDs deterministically using only versions available by the replay cutoff.
- At snapshot D, use ticket creation through D. Ticket closure-dependent fields may be used only when closure was observable by D; otherwise treat the ticket as unresolved at D.
- Exclude `retention_contact` explicitly. Keep a model feature allowlist and reject unexpected input columns.
- Record unusable keys, timestamps, numeric defects, duplicate counts, and before/after row counts.
- Reingesting the same batch must not duplicate records or change results without an input/configuration change.

## 5. Storage and proposed repository structure

```text
bda-prediction-model/
  README.md
  IMPLEMENTATION_PLAN.md
  requirements.txt
  config.json
  Dockerfile
  compose.yaml
  src/
    ingest.py
    preprocess.py
    features.py
    labels.py
    train.py
    evaluate.py
    predict.py
    pipeline.py
    app.py
  tests/
    test_temporal_rules.py
    test_pipeline_consistency.py
  data/
    raw/                         # Existing source files
    late_delivery/               # Existing source files
    processed/
      clean/cdr/
      clean/tickets/
      daily/cdr/
      features/snapshot_date=.../
      labels/snapshot_date=.../
  artifacts/
    models/<version>/
    runs/<run_id>/
    final_evaluation/
  state/
    replay.json
    jobs/
  logs/
```

This is a proposed layout, not a requirement to create empty modules. Combine very small modules if doing so keeps the code clearer.

Partition clean Parquets by event date so batch ingestion and historical filtering are practical. Store feature and label tables by snapshot date. Avoid loading all event-level CDR into pandas; use Spark for the large transformations and collect only manageable subscriber-level modeling tables.

Daily sums and counts may be reused for rolling windows. Distinct counts, medians, and modes must be computed correctly across the entire window; do not sum daily distinct counts as though they were exact window-level values.

## 6. Initial feature set

All features refer to the same 14-day historical window unless explicitly stated otherwise.

| Source | Initial features |
|---|---|
| CDR activity | Event count, actual distinct active days, days since last activity |
| Calls | Voice-call count, total and average valid call duration |
| Traffic | Total upload/download bytes, average traffic per event |
| Charges | Total and average valid charge |
| Service behavior | Roaming share and service-type counts/shares |
| Tickets | Ticket count, category counts, unresolved tickets as of D |
| Historical closures | Average valid resolution duration for tickets closed by D, plus missingness/availability indicators |
| Change over time | Compare activity in the last seven days with the preceding seven days |

Use clear denominators. CDR contains multiple service types, so an event count is not automatically a call count. Count active days explicitly rather than calling the span between first and last event the number of active days.

Do not show a registration date unless an allowed source actually provides one. The first observed event may be displayed as **first observed activity in this dataset**, not as registration.

Fit imputation/scaling only on training data. Package the fitted transformations together with the model so the interface applies exactly the same transformations.

## 7. Homework reuse map

| Homework | Reuse | Required adaptation |
|---|---|---|
| `../hw/HW_1.ipynb`, `../hw/HW_2.ipynb` | Profiling, encoding handling, ticket cleaning, late-delivery handling, Parquet writes | Local configurable paths, cutoff-aware versions, deterministic batch processing |
| `../hw/HW_3_1.ipynb`, `../hw/HW_3_2.ipynb` | Spark DataFrames, execution-plan inspection, caching, joins, performance measurements | Apply optimizations only where measured; remove dependencies on removed reference tables |
| `../hw/HW_4_1.ipynb`, `../hw/HW_4_2.ipynb` | Aggregate before joining, derived features, EDA | Historical windows, corrected activity definitions, explicit snapshot dates |
| `../hw/HW_5_1.ipynb` | Logistic regression, Random Forest comparison, metrics, model persistence | Replace static truth labels and random split with the agreed inactivity target and temporal design |
| `../hw/HW_5_2.ipynb` | XGBoost, interpretation, threshold analysis, saved artifacts | Tune only on validation; bounded runtime; optional SHAP sample |
| Lecture/lab 6.1 material | Functions, configuration, logging, idempotency, Docker, README | A small runnable pipeline suitable for local use |

Existing notebooks contain orchestration descriptions, but a working scheduler template has not been identified in the eight homework files inspected. Begin with an explicit stage runner; add another provided template only if it is available and useful.

## 8. Training, metrics, and artifacts

Start with logistic regression and one bounded XGBoost configuration. Establish correctness and runtime before adding tuning. Full batch retraining creates a new version; do not describe it as online learning or assume the old model supports incremental updates.

For each run, save:

- Run ID, model version, parent/incumbent version, replay date, and job status.
- Exact training/validation snapshot dates and feature/outcome boundaries.
- Input manifest/fingerprints, code commit when available, configuration, random seed, and package versions.
- Fitted preprocessing, model, feature names/order/types, and prediction threshold.
- Row counts, class balance, metrics, duration, and candidate-selection decision.
- Validation probabilities and identifiers sufficient to reproduce evaluation.
- Optional feature importance and SHAP outputs on a bounded sample.

Display log loss, ROC-AUC, PR-AUC, precision, recall, F1, and the confusion matrix. Use the same log-loss definition across runs; calculate it on unmodified validation observations. Report undefined metrics rather than hiding missing classes.

Begin with a threshold of 0.5 for a transparent baseline. If threshold tuning is added, choose it using validation under a documented rule. Business cost estimates require explicit assumptions; do not invent real retention savings.

## 9. Browser interface

### View A — Data and pipeline

- Current replay date, loaded date range, complete-day status, and source record counts.
- Current stage: ingest, clean, aggregate, label, train, evaluate, save, completed, or failed.
- Buttons for the next scheduled batch and candidate training, enabled only when their prerequisites are met.
- Quality summary and recent logs, including an actionable error if processing fails.

### View B — Model history

- Active model version and its training/validation boundaries.
- Run table with candidate/incumbent status, metrics, class balance, and duration.
- Validation log-loss and PR-AUC charts with the validation dates visible.
- Final shared-cohort comparison after day 90 only.
- Clear pending-outcome state when future labels are unavailable.

### View C — Subscriber prediction

- Select an eligible subscriber at the selected snapshot. Use the reserved final cohort for the day-76 final demonstration.
- Display historical features in readable units, the snapshot date, and model version.
- Predict inactivity probability and threshold-based risk classification through the saved preprocessing/model pipeline.
- Reveal the observed outcome only after the full 14-day window is available.
- Optional what-if edits must be clearly labeled synthetic scenarios. They must not alter the original feature table, saved predictions, or observed outcome.

Show genuine progress: stage changes and completed work counts. Do not invent smooth percentages or simulated loss values. Per-iteration boosting loss is an optional later enhancement, separate from loss across model versions.

## 10. Execution and orchestration

Use a sequential stage runner with explicit dependencies:

1. Validate replay request and available inputs.
2. Ingest and clean the newly exposed batch.
3. Write clean Parquets and update daily aggregates.
4. Build requested historical features and mature labels.
5. Validate temporal rules and class balance.
6. Train, evaluate, select, and persist a candidate when requested.
7. Update replay and active-model state only after successful completion.

Keep long-running work outside the Streamlit render cycle. Use one background worker process and durable job-status files for the local application. Permit one mutating job at a time. Browser refresh must not launch duplicate jobs or lose their status.

Write outputs to temporary locations and publish them atomically after success where practical. Preserve a working model if a later job fails. Persist datasets, artifacts, state, and logs in mounted directories so container recreation does not erase them.

The Docker image must include the Java runtime required by Spark and the pinned Python dependencies. Mount source data read-only and output locations writable. Test both the pipeline command and the browser application inside the container.

## 11. Configuration

Keep the following in `config.json` or a similarly simple explicit configuration:

- Source/output paths and allowed source names.
- Feature window 14 days, outcome horizon 14 days, and snapshot stride 7 days.
- Replay milestones 60, 67, 74, 76, 90.
- Timestamp parsing/timezone assumptions and accepted quality rules.
- Seed, Spark resources, model parameters, selection metric, and threshold.
- Feature allowlist and explicitly forbidden columns.

Derive dates from the discovered dataset calendar. Record the resolved configuration with every run.

## 12. Implementation milestones and acceptance checks

### Milestone 1 — Allowed-data audit and target feasibility

- Inventory CDR/tickets under the remaining source directories without accessing removed reference data.
- Verify the 90-day calendar, late-delivery coverage, available timestamp semantics, and schema.
- Calculate eligible population and target balance for the planned training, validation, and final snapshots.
- Confirm enough positive and negative examples exist; otherwise resolve the target issue before training.

**Done when:** the target is demonstrably constructible and a compact audit/EDA report records coverage, exclusions, and label counts.

### Milestone 2 — Reusable Parquet pipeline

- Adapt cleaning code into simple functions and configuration.
- Implement batch discovery, date exposure, deduplication, quality reporting, and partitioned writes.
- Add daily aggregates and verify repeat ingestion.

**Done when:** ingesting the same batch twice leaves the same logical data and counts; raw files remain unchanged.

### Milestone 3 — Historical features and labels

- Implement `build_features(snapshot_date)` and a separate `build_labels(snapshot_date, available_through)`.
- Build scheduled snapshots and a temporal split manifest.
- Apply source/feature allowlists and historical ticket-state rules.

**Done when:** future records cannot change earlier features, incomplete horizons produce unknown labels, and the example key is unique.

### Milestone 4 — Models and reproducible evaluation

- Train the baseline and bounded candidate model.
- Compare them and the incumbent on the same current validation set.
- Save complete versioned artifacts and reload them for prediction.

**Done when:** loaded artifacts reproduce saved probabilities within numeric tolerance, and model selection never reads final outcomes.

### Milestone 5 — Replay worker and interface

- Implement durable replay/job state, stage reporting, and single-job execution.
- Add the three interface views and eligible subscriber selection.
- Separate pending predictions from mature evaluations.

**Done when:** a browser user can run the planned replay, inspect model history, and request a prediction without editing code.

### Milestone 6 — Docker and final demonstration

- Containerize the verified pipeline and interface.
- Run days 60, 67, and 74; freeze predictions at 76; reveal outcomes at 90.
- Produce the final comparison and explain limitations, including the inactivity proxy and rolling validation windows.
- Complete README launch instructions, architecture description, and refactoring report.

**Done when:** a fresh container launch can reproduce the workflow using mounted data and persist results across restart.

## 13. Focused correctness tests

Use small synthetic fixtures for temporal logic and one bounded real-data smoke run. Avoid rerunning all 90 days for every test.

- An event on D is a feature input; an event on D+1 is outcome-only.
- Events at the last outcome day D+14 count; D+15 does not.
- Adding future data cannot change features for an earlier snapshot.
- Missing future partitions leave labels unknown.
- Ticket closure after D cannot affect ticket state or duration at D.
- Forbidden columns and paths cannot enter model construction.
- Duplicate ingestion preserves logical row counts and aggregate values.
- Training labels end before the validation outcome period; preprocessing fits only training data.
- Final day-76 prediction files are immutable after publication and are evaluated without refitting.
- Reloaded artifacts predict consistently, and failed jobs preserve the active model.

## 14. Definition of project completion

The project is complete when it demonstrates, using only allowed data:

1. A documented 14-day inactivity target with validated historical windows.
2. A deterministic CDR/tickets-to-Parquet-to-features pipeline using course techniques.
3. Several saved model versions from the scheduled replay, with honest quality reporting.
4. A final evaluation of frozen predictions on the untouched day-77–90 outcomes.
5. A browser interface for processing status, model history, and individual prediction.
6. A reproducible Docker launch, persistent artifacts, clear documentation, and passing temporal correctness checks.

Milestones **1–6 are complete**. All 32 Docker tests pass; the expanded final report and real-data interface checks also pass. Training milestones 60, 67, and 74 ran at delivery clocks 61, 68, and 75, saving six model versions. The milestone-67 XGBoost remained active after milestone 74 under the same-cohort validation PR-AUC rule. At day 76, all six versions produced frozen predictions for 153,785 subscribers. Delivery day 91 revealed the day-77–90 outcomes: 22,443 inactive and 131,342 active subscribers, matching the independent audit. Evaluation did not refit, reselect, or modify models or frozen files. Historical features matched after full ingestion. A fresh app container retained identical checksums for 36 model/result/state files; all six loaded pipelines reproduced frozen probabilities on 50 subscribers, and the completed demo resumed without retraining. README contains concise Docker launch instructions; architecture, homework reuse/refactoring, final metrics, and limitations are documented in `docs/`. The active model's final PR-AUC is 0.247688, but recall at the fixed 0.5 threshold is only 0.000936; this limitation is explicitly reported.
