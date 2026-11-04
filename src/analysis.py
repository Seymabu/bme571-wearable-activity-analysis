"""Participant-aware analysis of UCI HAR, designed for a CPU or Google Colab.

The supplied 561 columns are already filtered, engineered and normalized by the
dataset creators. This module keeps its own fitted preprocessing inside training
folds; it does not claim to reconstruct upstream preprocessing provenance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import re
from urllib.request import Request, urlopen
import warnings
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
DATA_DOI = "https://doi.org/10.24432/C54S4K"
EXPECTED_PAYLOAD_SHA256 = "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"
LABELS = np.arange(1, 7)
SEED = 571
C_GRID = (0.01, 0.1, 1.0, 10.0)
COMPACT_PATTERN = re.compile(r"^t(?:BodyAcc|GravityAcc|BodyAccJerk)-(?:mean|std)\(\)-[XYZ]$")
MODEL_LABELS = {
    "dummy": "Most-frequent baseline",
    "full_logistic": "Logistic · all 561 features",
    "compact_logistic": "Logistic · 18 accelerometer features",
}
COLORS = {"dummy": "#8497A6", "full_logistic": "#174A6E", "compact_logistic": "#009E8F"}


@dataclass
class HARData:
    """Arrays preserve official row order; subject identifiers are never features."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    subject_train: np.ndarray
    subject_test: np.ndarray
    features: pd.DataFrame
    activity_names: dict[int, str]
    provenance: dict


def validate_subject_split(train_subjects: np.ndarray, test_subjects: np.ndarray) -> None:
    """Reject a participant-overlapping or empty train/validation/test split."""
    train, test = set(np.asarray(train_subjects).tolist()), set(np.asarray(test_subjects).tolist())
    if not train or not test:
        raise ValueError("Both subject sets must be nonempty.")
    overlap = sorted(train & test)
    if overlap:
        raise ValueError(f"Participant leakage: overlapping subject IDs {overlap}")


def compact_feature_indices(features: pd.DataFrame) -> np.ndarray:
    """Predeclared 18 accelerometer-only mean/std XYZ features, no outcome selection."""
    names = features["original_name"].astype(str)
    indices = np.flatnonzero(names.map(lambda value: bool(COMPACT_PATTERN.fullmatch(value))))
    expected = {
        f"{signal}-{stat}()-{axis}"
        for signal in ("tBodyAcc", "tGravityAcc", "tBodyAccJerk")
        for stat in ("mean", "std") for axis in "XYZ"
    }
    if len(indices) != 18 or set(names.iloc[indices]) != expected:
        raise ValueError("Compact feature contract requires each of the 18 specified features exactly once.")
    return indices


def _download_archive(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATA_URL, headers={"User-Agent": "wearable-activity-analysis/1.0"})
    partial = target.with_suffix(".download")
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def load_data(data_dir: str | Path) -> HARData:
    """Load the official ZIP from ``data_dir/uci_har.zip``, downloading if absent.

    Accepts the UCI wrapper ZIP or its original nested ZIP. The known original
    payload hash is verified. Only exact, needed archive members are read in
    memory; no ZIP paths are extracted onto the filesystem.
    """
    archive_path = Path(data_dir) / "uci_har.zip"
    if not archive_path.exists():
        _download_archive(archive_path)
    archive_bytes = archive_path.read_bytes()
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    with ZipFile(BytesIO(archive_bytes)) as outer:
        nested = [name for name in outer.namelist() if name == "UCI HAR Dataset.zip"]
        payload = outer.read(nested[0]) if nested else archive_bytes
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != EXPECTED_PAYLOAD_SHA256:
        raise ValueError("The original UCI HAR payload hash changed. Audit its contents before changing EXPECTED_PAYLOAD_SHA256.")

    with ZipFile(BytesIO(payload)) as archive:
        prefix = "UCI HAR Dataset/"

        def read(name: str) -> bytes:
            member = archive.getinfo(prefix + name)
            if member.file_size > 100_000_000:
                raise ValueError(f"Unexpectedly large member: {name}")
            return archive.read(member)

        feature_rows = [line.split(maxsplit=1) for line in read("features.txt").decode().splitlines()]
        features = pd.DataFrame(feature_rows, columns=["feature_index", "original_name"])
        features["feature_index"] = features["feature_index"].astype(int)
        features["unique_name"] = [f"{i:03d}__{name}" for i, name in features.itertuples(index=False, name=None)]
        label_rows = [line.split(maxsplit=1) for line in read("activity_labels.txt").decode().splitlines()]
        activity_names = {int(index): name for index, name in label_rows}
        arrays = {}
        for split in ("train", "test"):
            for kind in ("X", "y", "subject"):
                arrays[f"{kind}_{split}"] = np.loadtxt(
                    BytesIO(read(f"{split}/{kind}_{split}.txt")),
                    dtype=float if kind == "X" else int,
                )
        readme = read("README.txt").decode("latin-1")
        feature_info = read("features_info.txt").decode("latin-1")

    data = HARData(
        **arrays, features=features, activity_names=activity_names,
        provenance={
            "source_url": DATA_URL, "dataset_page": DATA_PAGE, "doi": DATA_DOI,
            "archive_sha256": archive_hash, "payload_sha256": payload_hash,
            "archive_bytes": len(archive_bytes), "payload_hash_verified": True,
            "current_uci_license": "CC BY 4.0 (dataset landing page checked 2026-11-03)",
            "historical_readme_commercial_restriction": "Any commercial use is prohibited." in readme,
            "license_note": "The current UCI landing page and historical archive README differ. Data are downloaded from UCI, not redistributed or relicensed by this project.",
            "upstream_processing": "Creator-provided filtered and engineered features normalized to [-1,1]. The source does not establish train-only fitting of every upstream transformation.",
            "sensor_units": "Underlying acceleration: g; angular velocity: radians/second. Released feature columns are normalized and do not retain those physical units.",
            "sampling_hz": 50, "window_seconds": 2.56, "window_overlap_fraction": 0.5,
            "readme_sha256": hashlib.sha256(readme.encode("latin-1")).hexdigest(),
            "features_info_sha256": hashlib.sha256(feature_info.encode("latin-1")).hexdigest(),
        },
    )
    validate_subject_split(data.subject_train, data.subject_test)
    if data.X_train.shape != (7352, 561) or data.X_test.shape != (2947, 561):
        raise ValueError("Unexpected official dataset dimensions.")
    if len(np.unique(data.subject_train)) != 21 or len(np.unique(data.subject_test)) != 9:
        raise ValueError("Expected 21 training and 9 test participants.")
    if not np.isfinite(data.X_train).all() or not np.isfinite(data.X_test).all():
        raise ValueError("Non-finite feature values found.")
    if not np.array_equal(features["feature_index"], np.arange(1, 562)):
        raise ValueError("The original feature indices are not contiguous 1–561.")
    for split in ("train", "test"):
        y, subjects = getattr(data, f"y_{split}"), getattr(data, f"subject_{split}")
        if len(y) != len(subjects) or len(y) != len(getattr(data, f"X_{split}")):
            raise ValueError("Labels, participants and feature rows do not align.")
        if not np.array_equal(np.unique(y), LABELS):
            raise ValueError("Expected all six original activity labels.")
    compact_feature_indices(features)
    return data


def subject_scores(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    """One score per person, using the same fixed six-label macro-F1 definition."""
    y_true, y_pred, subjects = map(np.asarray, (y_true, y_pred, subjects))
    if not (len(y_true) == len(y_pred) == len(subjects)):
        raise ValueError("True labels, predictions and subjects must align.")
    rows = []
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        rows.append({
            "subject_id": int(subject), "n_windows": int(mask.sum()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], labels=LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
        })
    return pd.DataFrame(rows)


def bootstrap_subject_scores(score_matrix: np.ndarray, n_bootstrap: int = 2000, seed: int = SEED) -> dict:
    """Paired cluster bootstrap: rows are independent people, columns are models.

    Resampling rows keeps all model results from a person together. This is the
    bootstrap of equally weighted person-level macro-F1, not of individual windows.
    """
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 1:
        raise ValueError("Provide a finite matrix with at least two participants and one model.")
    if not np.isfinite(scores).all() or n_bootstrap < 2:
        raise ValueError("Finite scores and at least two bootstrap replicates are required.")
    draws = np.random.default_rng(seed).integers(0, len(scores), size=(n_bootstrap, len(scores)))
    replicates = scores[draws].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975], axis=0)
    return {"mean": scores.mean(axis=0), "ci_low": low, "ci_high": high, "replicates": replicates}


def paired_subject_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray, n_bootstrap: int = 2000, seed: int = SEED) -> dict:
    """Estimate a minus b using corresponding, identically ordered participants."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Both inputs must be aligned one-dimensional participant-score arrays.")
    result = bootstrap_subject_scores(np.column_stack([a, b]), n_bootstrap=n_bootstrap, seed=seed)
    difference = result["replicates"][:, 0] - result["replicates"][:, 1]
    low, high = np.quantile(difference, [0.025, 0.975])
    return {"mean_difference": float(np.mean(a - b)), "ci_low": float(low), "ci_high": float(high)}


def make_classifier(c_value: float) -> Pipeline:
    """All learned scaling is contained in the fitted training pipeline."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000, random_state=SEED)),
    ])


def _tune(X: np.ndarray, y: np.ndarray, subjects: np.ndarray, folds: list, model_name: str) -> tuple[float, float, list]:
    rows, candidates = [], []
    for c_value in C_GRID:
        participant_values = []
        candidate_rows = []
        for fold_id, (train, validation) in enumerate(folds, start=1):
            validate_subject_split(subjects[train], subjects[validation])
            classifier = make_classifier(c_value)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(X[train], y[train])
            prediction = classifier.predict(X[validation])
            scores = subject_scores(y[validation], prediction, subjects[validation])
            participant_values.extend(scores["macro_f1"].tolist())
            candidate_rows.append({
                "model": model_name, "C": c_value, "fold": fold_id,
                "mean_subject_macro_f1": float(scores["macro_f1"].mean()),
                "pooled_macro_f1": float(f1_score(y[validation], prediction, labels=LABELS, average="macro", zero_division=0)),
                "n_validation_subjects": len(scores),
                "validation_subjects": ";".join(map(str, sorted(np.unique(subjects[validation])))),
                "training_subjects": ";".join(map(str, sorted(np.unique(subjects[train])))),
            })
        if len(participant_values) != len(np.unique(subjects)):
            raise AssertionError("Cross-validation must evaluate each training person once per C.")
        score = float(np.mean(participant_values))
        for row in candidate_rows:
            row["candidate_mean_subject_macro_f1"] = score
        rows.extend(candidate_rows)
        candidates.append((score, c_value))
    # The grid is ascending: a tie selects the smaller C (stronger regularization).
    best_score, best_c = max(candidates, key=lambda pair: (pair[0], -pair[1]))
    return best_c, best_score, rows


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _figures(data: HARData, summary: pd.DataFrame, per_subject: pd.DataFrame, predictions: dict, folder: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 180,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelcolor": "#183342", "text.color": "#183342",
    })
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(folder / f"{name}.png", bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    labels = [data.activity_names[i].replace("_", " ").title() for i in LABELS]
    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    positions = np.arange(6)
    for offset, y, color, name in [(-.19, data.y_train, "#174A6E", "Train · 21 participants"), (.19, data.y_test, "#009E8F", "Test · 9 participants")]:
        counts = np.array([(y == label).sum() for label in LABELS])
        ax.bar(positions + offset, 100 * counts / len(y), width=.36, color=color, label=name)
    ax.set(xticks=positions, xticklabels=labels, ylabel="Windows within each split (%)", title="Activity balance in the official participant-separated split")
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(frameon=False)
    save(fig, "01_activity_balance")

    scaler = StandardScaler().fit(data.X_train)
    pca = PCA(n_components=2, svd_solver="full").fit(scaler.transform(data.X_train))
    coordinates = pca.transform(scaler.transform(data.X_train))
    fig, ax = plt.subplots(figsize=(8.8, 5.6), layout="constrained")
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    rng = np.random.default_rng(SEED)
    for label, color, name in zip(LABELS, palette, labels):
        eligible = np.flatnonzero(data.y_train == label)
        selected = rng.choice(eligible, size=min(300, len(eligible)), replace=False)
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=12, alpha=.5, color=color, label=name, linewidths=0)
    ax.set(title="Training data: an exploratory two-dimensional view", xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.legend(frameon=False, markerscale=1.5, loc="lower right", fontsize=9)
    save(fig, "02_training_pca")

    matrix = confusion_matrix(data.y_test, predictions["full_logistic"], labels=LABELS)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.1, 6.6), layout="constrained")
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(6):
        for col in range(6):
            ax.text(col, row, f"{normalized[row, col]:.0%}\n({matrix[row, col]})", ha="center", va="center", color="white" if normalized[row, col] > .5 else "#183342", fontsize=9)
    ax.set(xticks=positions, yticks=positions, xticklabels=labels, yticklabels=labels, ylabel="True activity", xlabel="Predicted activity", title="Unseen participants: full-feature logistic regression")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="Fraction of true-class windows", shrink=.8)
    save(fig, "03_confusion_matrix")

    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    subject_ids = sorted(per_subject["subject_id"].unique())
    for model in ("dummy", "full_logistic", "compact_logistic"):
        scores = per_subject[per_subject["model"] == model].set_index("subject_id").loc[subject_ids, "macro_f1"]
        ax.plot(np.arange(len(subject_ids)), scores, marker="o", lw=1.7, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set(xticks=np.arange(len(subject_ids)), xticklabels=subject_ids, ylim=(0, 1.03), xlabel="Held-out participant ID", ylabel="Six-class macro-F1", title="Performance varies across held-out people")
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.grid(axis="y", alpha=.15)
    save(fig, "04_participant_performance")

    fig, ax = plt.subplots(figsize=(8.4, 5.1), layout="constrained")
    for model in ("compact_logistic", "full_logistic"):
        row = summary.set_index("model").loc[model]
        mean = row["mean_subject_macro_f1"]
        ax.errorbar(row["n_features"], mean, yerr=[[mean - row["ci_low"]], [row["ci_high"] - mean]], fmt="o", color=COLORS[model], capsize=6, markersize=9, lw=2, label=MODEL_LABELS[model])
    dummy = summary.set_index("model").loc["dummy", "mean_subject_macro_f1"]
    ax.axhline(dummy, color=COLORS["dummy"], linestyle="--", label="Most-frequent baseline")
    ax.set(xlim=(-30, 610), ylim=(0, 1.05), xticks=[18, 561], xlabel="Number of input features", ylabel="Mean participant macro-F1", title="Feature count versus performance on unseen people")
    ax.legend(frameon=False, loc="center right", fontsize=9)
    fig.text(.12, -.015, "Bars: 95% participant-bootstrap intervals; 9 held-out people, 2,000 paired resamples.", fontsize=9)
    save(fig, "05_features_vs_performance")


def run_analysis(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Tune using training people, freeze models, evaluate the official test once.

    Writes summary JSON/CSVs and PNG/SVG figures. The returned dict is the same
    serializable content as metrics.json. No fitted model or raw data is exported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    compact = compact_feature_indices(data.features)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(data.X_train, data.y_train, groups=data.subject_train))
    columns = {"full_logistic": np.arange(561), "compact_logistic": compact}
    models, cv_rows, chosen = {}, [], {}
    # Restrict native BLAS/OpenMP as well as Python-level parallelism to one CPU thread.
    with threadpool_limits(limits=1):
        for name, indices in columns.items():
            print(f"Tuning {name} on training participants only...", flush=True)
            c_value, cv_score, rows = _tune(data.X_train[:, indices], data.y_train, data.subject_train, folds, name)
            chosen[name] = {"C": c_value, "training_cv_mean_subject_macro_f1": cv_score}
            cv_rows.extend(rows)
            model = make_classifier(c_value)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(data.X_train[:, indices], data.y_train)
            models[name] = model

        dummy = DummyClassifier(strategy="most_frequent").fit(data.X_train[:, :1], data.y_train)
        # No test metric above this point influences selection, parameters or features.
        predictions = {"dummy": dummy.predict(data.X_test[:, :1])}
        for name, model in models.items():
            predictions[name] = model.predict(data.X_test[:, columns[name]])

        model_order = ["dummy", "full_logistic", "compact_logistic"]
        per_subject_frames, reports, matrices = [], {}, {}
        for name in model_order:
            scores = subject_scores(data.y_test, predictions[name], data.subject_test)
            scores["model"] = name
            per_subject_frames.append(scores)
            reports[name] = classification_report(data.y_test, predictions[name], labels=LABELS,
                target_names=[data.activity_names[i] for i in LABELS], output_dict=True, zero_division=0)
            matrices[name] = confusion_matrix(data.y_test, predictions[name], labels=LABELS).tolist()
        per_subject = pd.concat(per_subject_frames, ignore_index=True)
        score_table = per_subject.pivot(index="subject_id", columns="model", values="macro_f1").loc[:, model_order].sort_index()
        intervals = bootstrap_subject_scores(score_table.to_numpy())
        metric_rows = []
        for index, name in enumerate(model_order):
            metric_rows.append({
                "model": name, "n_features": 0 if name == "dummy" else len(columns[name]),
                "selected_C": None if name == "dummy" else chosen[name]["C"],
                "mean_subject_macro_f1": float(intervals["mean"][index]),
                "ci_low": float(intervals["ci_low"][index]), "ci_high": float(intervals["ci_high"][index]),
                "pooled_macro_f1": float(f1_score(data.y_test, predictions[name], labels=LABELS, average="macro", zero_division=0)),
                "pooled_accuracy": float(accuracy_score(data.y_test, predictions[name])),
                "pooled_balanced_accuracy": float(balanced_accuracy_score(data.y_test, predictions[name])),
            })
        summary = pd.DataFrame(metric_rows)
        comparison = paired_subject_bootstrap(score_table["compact_logistic"].to_numpy(), score_table["full_logistic"].to_numpy())
        audit = {
            "dataset": "UCI Human Activity Recognition Using Smartphones", "provenance": data.provenance,
            "train_shape": list(data.X_train.shape), "test_shape": list(data.X_test.shape),
            "train_subjects": sorted(map(int, np.unique(data.subject_train))),
            "test_subjects": sorted(map(int, np.unique(data.subject_test))),
            "subject_intersection": [], "missing_or_nonfinite_values": 0,
            "original_duplicate_feature_name_count": int(data.features["original_name"].duplicated().sum()),
            "unique_indexed_feature_names": bool(data.features["unique_name"].is_unique),
            "activity_labels": {str(k): v for k, v in data.activity_names.items()},
            "class_counts": {split: {data.activity_names[int(label)]: int((getattr(data, f'y_{split}') == label).sum()) for label in LABELS} for split in ("train", "test")},
            "all_six_activities_per_subject": {
                split: all(np.array_equal(np.unique(getattr(data, f'y_{split}')[getattr(data, f'subject_{split}') == person]), LABELS) for person in np.unique(getattr(data, f'subject_{split}')))
                for split in ("train", "test")
            },
            "compact_feature_count": len(compact),
            "compact_definition": "tBodyAcc, tGravityAcc, tBodyAccJerk × mean()/std() × X/Y/Z; declared before test evaluation",
        }
        results = {
            "study": "Wearable activity recognition: generalization to unseen participants",
            "author": "Dr. Şeyma Bozkurt Uzan", "seed": SEED,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
            "primary_metric": "Equal-weight mean of participant-level six-class macro-F1",
            "selection": {"training_only": True, "cv": "5-fold StratifiedGroupKFold by subject", "candidate_C": list(C_GRID), "group_equal_tuning": "Each training participant contributes once; fold means are not averaged with unequal person counts.", "chosen": chosen},
            "test_evaluation": {"participants": 9, "windows": len(data.y_test), "fixed_official_split": True},
            "models": metric_rows,
            "compact_minus_full": comparison,
            "bootstrap": {"method": "Paired resampling of participant-level score rows", "replicates": 2000, "confidence_level": 0.95, "interval": "percentile", "seed": SEED,
                "limitation": "Nine held-out participants; intervals are descriptive and conditional on fitted models and this split, not training uncertainty or clinical-population validation."},
            "class_reports": reports, "confusion_matrices": matrices,
            "dataset_provenance": data.provenance,
            "interpretation_limits": [
                "Six scripted activities from waist-mounted smartphones; no clinical outcomes or rehabilitation efficacy are tested.",
                "Overlapping windows are correlated; people, not windows, are the uncertainty resampling unit.",
                "Compact feature counts suggest a design tradeoff; device energy, runtime and clinical utility were not measured.",
                "The same held-out people compare prespecified models. Further model development requires fresh validation rather than repeated test optimization.",
                "Creator-provided upstream feature normalization is inherited; only the additional pipeline scaling is demonstrably fitted on training rows.",
            ],
        }
        _write_json(output_dir / "metrics.json", results)
        _write_json(output_dir / "dataset_audit.json", audit)
        summary.to_csv(output_dir / "model_metrics.csv", index=False)
        per_subject.to_csv(output_dir / "per_subject_metrics.csv", index=False)
        pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
        data.features.iloc[compact].to_csv(output_dir / "compact_features.csv", index=False)
        data.features.to_csv(output_dir / "feature_manifest.csv", index=False)
        _figures(data, summary, per_subject, predictions, output_dir / "figures")
    print(summary.to_string(index=False), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_analysis(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
