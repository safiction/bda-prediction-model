"""HW_5 metrics on unchanged validation observations; PR-AUC means average precision."""


def evaluate_predictions(target, probabilities, threshold=0.5):
    import numpy as np
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score)
    # Одинаковая точность log loss для float32- и float64-выходов моделей.
    y, probability = np.asarray(target), np.asarray(probabilities, dtype=float)
    if not len(y) or y.ndim != 1 or probability.shape != y.shape or not set(y) <= {0, 1}:
        raise ValueError("Metrics require matching nonempty binary targets and probabilities")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1")
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    def defined(value):
        return float(value) if np.isfinite(value) else None

    both_classes = len(np.unique(y)) == 2
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "accuracy": float((tn + tp) / len(y)),
        "roc_auc": float(roc_auc_score(y, probability)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probability)) if both_classes else None,
        "precision": defined(precision_score(y, predicted, zero_division=np.nan)),
        "recall": defined(recall_score(y, predicted, zero_division=np.nan)),
        "f1": defined(f1_score(y, predicted, zero_division=np.nan)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_model(candidate_scores, incumbent_version, incumbent_metrics, config):
    settings = config["modeling"]
    if settings["selection_metric"] != "pr_auc" or settings["pr_auc_definition"] != "average_precision_score":
        raise ValueError("The initial selection rule supports validation average precision only")
    gain = settings["minimum_pr_auc_gain"]
    if gain < 0 or not candidate_scores or any(metrics["pr_auc"] is None for metrics in candidate_scores.values()):
        raise ValueError("Selection requires defined PR-AUC and a nonnegative minimum gain")
    best = max(candidate_scores, key=lambda version: candidate_scores[version]["pr_auc"])
    if incumbent_version is None:
        return best, "Initial model: highest validation PR-AUC; ties use candidate order."
    if incumbent_metrics["pr_auc"] is None:
        raise ValueError("Incumbent PR-AUC is undefined")
    if candidate_scores[best]["pr_auc"] > incumbent_metrics["pr_auc"] + gain:
        return best, "Candidate improved PR-AUC on the same validation cohort."
    return incumbent_version, "Kept incumbent: no strict PR-AUC improvement on the same validation cohort."


def write_comparison(path, run):
    def show(value):
        return "undefined" if value is None else f"{value:.6f}"

    lines = ["# Validation comparison", "", "Target: 14-day inactivity proxy for churn.", "",
             f"Milestone: {run['milestone']}; validation snapshot: {run['split']['validation']['snapshot_date']}.",
             "PR-AUC is average precision. Threshold is saved with each model. Validation rows are not resampled.", "",
             "| Version | Role | Log loss | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 | TN / FP / FN / TP |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    entries = [(version, "new candidate", info["metrics"]) for version, info in run["candidates"].items()]
    if run["incumbent"] is not None:
        entries.append((run["parent_version"], "incumbent on current validation", run["incumbent"]["metrics"]))
    for version, role, metrics in entries:
        scores = " | ".join(show(metrics[name]) for name in ("log_loss", "accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1"))
        matrix = " / ".join(str(metrics["confusion_matrix"][name]) for name in ("tn", "fp", "fn", "tp"))
        lines.append(f"| {version} | {role} | {scores} | {matrix} |")
    lines += ["", f"Selected: {run['selected_version']}. {run['selection_reason']}", "",
              "Validation windows move between runs; cross-run scores use different observations.",
              "Subscribers repeat across training snapshots, so those observations are correlated.",
              "Final day-76 outcomes are not used for fitting or selection.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
