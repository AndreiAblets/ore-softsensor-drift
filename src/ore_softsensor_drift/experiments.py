from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .conformal import run_aps
from .data import (
    Preprocessor,
    apply_channel_drift,
    apply_global_drift,
    apply_jitter,
    batch_moment_correction,
    highest_inverse_cv_feature,
    inverse_cv_scores,
    inverse_cv_table,
    load_dataset,
    make_split,
)
from .metrics import aggregate, apply_temperature, evaluate, fit_temperature
from .models import MLPConfig, TorchMLP, apply_coral, fit_coral, fit_hgb, fit_logreg


DRIFT_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
DETECTION_THRESHOLDS = [2.0, 3.0, 5.0]
JITTER_SCALES = [0.01, 0.02, 0.05, 0.10]
DIAGNOSTIC_SPLIT_SEED = 0
DIAGNOSTIC_SUBSAMPLE_SEED = 42
REGIME_KMEANS_SEED = 42


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path
    output_dir: Path
    seeds: list[int]
    device: str = "cpu"
    fast: bool = False


class Reproduction:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.tables = config.output_dir / "tables"
        self.figures = config.output_dir / "figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.data = load_dataset(config.data_dir)

    def run(self) -> None:
        baseline = self.baseline()
        drift, per_channel, per_channel_005 = self.drift_fragility()
        robustness = self.model_robustness()
        preprocessing = self.preprocessing_robustness()
        remedies, coral = self.remedies()
        conformal = self.conformal_reliability()
        detection = self.drift_detection()
        regime = self.regime_ood()
        amplification = inverse_cv_table(self.data)
        high_inverse_cv = self.high_inverse_cv_selection()
        coef_audit = self.logreg_coefficient_audit()

        self.write_table("baseline_mean_std.csv", aggregate(baseline, ["model"], ["macro_f1", "accuracy", "ece", "brier"]))
        self.write_table("drift_global_mean_std.csv", aggregate(drift, ["drift_eps"], ["macro_f1", "accuracy", "ece", "brier"]))
        self.write_table("drift_per_channel_all.csv", aggregate(per_channel, ["feature", "tag", "drift_eps"], ["macro_f1", "accuracy"]))
        self.write_table("drift_per_channel_005.csv", aggregate(per_channel_005, ["feature", "tag"], ["macro_f1", "accuracy"]))
        self.write_table("model_robustness_long.csv", aggregate(robustness, ["drift_eps", "model"], ["macro_f1", "accuracy"]))
        self.write_wide_metric("model_robustness_mean.csv", robustness, "macro_f1", "mean")
        self.write_wide_metric("model_robustness_std.csv", robustness, "macro_f1", "std")
        self.write_table("preprocessing_robustness_long.csv", aggregate(preprocessing, ["drift_eps", "preprocessing"], ["macro_f1", "accuracy"]))
        self.write_wide_metric("preprocessing_robustness_mean.csv", preprocessing, "macro_f1", "mean", columns="preprocessing")
        self.write_wide_metric("preprocessing_robustness_std.csv", preprocessing, "macro_f1", "std", columns="preprocessing")
        self.write_table("remedies_mean_std.csv", aggregate(remedies, ["solution", "model"], ["macro_f1", "accuracy", "ece", "brier"]))
        self.write_table("coral_under_drift.csv", coral)
        self.write_table("conformal_solutions_aps.csv", aggregate(conformal, ["drift_eps", "scenario"], ["coverage", "avg_set_size", "singleton_rate", "macro_f1"]))
        self.write_table("drift_detection.csv", detection)
        self.write_table("regime_ood_mean_std.csv", aggregate(regime, ["condition", "model"], ["macro_f1", "accuracy"]))
        self.write_table("inverse_cv_amplification.csv", amplification)
        self.write_table("high_inverse_cv_selection.csv", high_inverse_cv)
        self.write_table("logreg_coefficient_audit.csv", coef_audit)
        self.write_table(
            "logreg_coefficient_audit_mean_std.csv",
            aggregate(
                coef_audit,
                ["feature", "tag"],
                [
                    "max_abs_coef",
                    "l2_coef",
                    "inverse_cv",
                    "max_logit_displacement_1pct",
                    "l2_logit_displacement_1pct",
                ],
            ),
        )

        self.plot_drift(drift)
        self.plot_channel_drift(per_channel_005)
        self.plot_channel_drift_heatmap(per_channel)
        self.plot_amplification(amplification)
        self.plot_model_robustness(robustness)
        self.plot_preprocessing_robustness(preprocessing)
        self.plot_remedies(remedies)
        self.plot_coral(coral)
        self.plot_conformal(conformal)
        self.plot_detection(detection)
        self.plot_regime(regime)

    def baseline(self) -> pd.DataFrame:
        rows = []
        for seed in self.config.seeds:
            x_train, y_train, x_val, y_val, x_test, y_test, x_adapt = self.standard_arrays(seed)
            for name in ["LogReg", "HistGB", "TorchMLP", "CORAL+LogReg"]:
                proba = self.fit_predict(name, x_train, y_train, x_val, y_val, x_test, seed, x_adapt)
                scores = evaluate(y_test, proba)
                rows.append({"seed": seed, "model": name, **scores.__dict__})
        return pd.DataFrame(rows)

    def drift_fragility(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        global_rows = []
        channel_rows = []
        for seed in self.config.seeds:
            data = self.data
            split = make_split(data.y_src, len(data.y_tar), seed)
            pre = Preprocessor().fit(data.x_src[split.train])
            x_train = pre.transform(data.x_src[split.train])
            x_val = pre.transform(data.x_src[split.val])
            y_train = data.y_src[split.train]
            y_val = data.y_src[split.val]
            y_test = data.y_tar
            model = fit_logreg(x_train, y_train, x_val, y_val, seed)
            temp = fit_temperature(model.predict_proba(x_val), y_val)

            for eps in DRIFT_GRID:
                x_test = pre.transform(apply_global_drift(data.x_tar, eps))
                scores = evaluate(y_test, apply_temperature(model.predict_proba(x_test), temp))
                global_rows.append({"seed": seed, "drift_eps": eps, **scores.__dict__})

                for j, tag in enumerate(data.feature_tags):
                    x_channel = pre.transform(apply_channel_drift(data.x_tar, j, eps))
                    scores = evaluate(y_test, apply_temperature(model.predict_proba(x_channel), temp))
                    channel_rows.append(
                        {
                            "seed": seed,
                            "drift_eps": eps,
                            "feature": data.feature_names[j],
                            "tag": tag,
                            **scores.__dict__,
                        }
                    )

        channel = pd.DataFrame(channel_rows)
        channel_005 = channel[channel["drift_eps"] == 0.05].copy()
        return pd.DataFrame(global_rows), channel, channel_005

    def model_robustness(self) -> pd.DataFrame:
        rows = []
        for seed in self.config.seeds:
            data = self.data
            split = make_split(data.y_src, len(data.y_tar), seed)
            x_train_raw = data.x_src[split.train]
            x_val_raw = data.x_src[split.val]
            y_train = data.y_src[split.train]
            y_val = data.y_src[split.val]
            y_test = data.y_tar
            drop_idx = highest_inverse_cv_feature(x_train_raw)
            keep_inverse_cv = [i for i in range(data.x_src.shape[1]) if i != drop_idx]

            for model_name, keep in [
                ("LogReg", None),
                ("HistGB", None),
                ("TorchMLP", None),
                ("LogReg_no_high_inverse_cv", keep_inverse_cv),
            ]:
                pre, model, temp = self.fit_standard_pipeline(
                    "LogReg" if model_name == "LogReg_no_high_inverse_cv" else model_name,
                    x_train_raw,
                    y_train,
                    x_val_raw,
                    y_val,
                    seed,
                    keep,
                )
                for eps in DRIFT_GRID:
                    x_test = pre.transform(apply_global_drift(data.x_tar, eps))
                    proba = apply_temperature(model.predict_proba(x_test), temp)
                    scores = evaluate(y_test, proba)
                    rows.append({"seed": seed, "drift_eps": eps, "model": model_name, **scores.__dict__})
        return pd.DataFrame(rows)

    def preprocessing_robustness(self) -> pd.DataFrame:
        rows = []
        for seed in self.config.seeds:
            data = self.data
            split = make_split(data.y_src, len(data.y_tar), seed)
            x_train_raw = data.x_src[split.train]
            x_val_raw = data.x_src[split.val]
            y_train = data.y_src[split.train]
            y_val = data.y_src[split.val]
            y_test = data.y_tar

            for name, scaler in [
                ("StandardScaler", StandardScaler()),
                ("RobustScaler", RobustScaler()),
                ("MinMaxScaler", MinMaxScaler()),
                ("NoScaling", None),
            ]:
                if scaler is None:
                    x_train = x_train_raw.astype(np.float32)
                    x_val = x_val_raw.astype(np.float32)

                    def transform(x: np.ndarray) -> np.ndarray:
                        return x.astype(np.float32)

                else:
                    scaler.fit(x_train_raw)
                    x_train = scaler.transform(x_train_raw).astype(np.float32)
                    x_val = scaler.transform(x_val_raw).astype(np.float32)

                    def transform(x: np.ndarray, fitted=scaler) -> np.ndarray:
                        return fitted.transform(x).astype(np.float32)

                model = fit_logreg(x_train, y_train, x_val, y_val, seed)
                temp = fit_temperature(model.predict_proba(x_val), y_val)
                for eps in DRIFT_GRID:
                    x_test = transform(apply_global_drift(data.x_tar, eps))
                    proba = apply_temperature(model.predict_proba(x_test), temp)
                    scores = evaluate(y_test, proba)
                    rows.append({"seed": seed, "drift_eps": eps, "preprocessing": name, **scores.__dict__})
        return pd.DataFrame(rows)

    def remedies(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        coral_rows = []
        for seed in self.config.seeds:
            data = self.data
            split = make_split(data.y_src, len(data.y_tar), seed)
            x_train_raw = data.x_src[split.train]
            x_val_raw = data.x_src[split.val]
            y_train = data.y_src[split.train]
            y_val = data.y_src[split.val]
            y_test = data.y_tar
            x_test_drift = apply_global_drift(data.x_tar, 0.05)
            rng = np.random.RandomState(seed)
            drop_idx = highest_inverse_cv_feature(x_train_raw)
            keep_inverse_cv = [i for i in range(data.x_src.shape[1]) if i != drop_idx]

            for model_name in ["LogReg", "HistGB", "TorchMLP"]:
                proba = self.fit_standard_model(model_name, x_train_raw, y_train, x_val_raw, y_val, x_test_drift, seed)
                rows.append({"seed": seed, "solution": "none", "model": model_name, **evaluate(y_test, proba).__dict__})

                proba = self.fit_standard_model(model_name, x_train_raw, y_train, x_val_raw, y_val, x_test_drift, seed, keep_inverse_cv)
                rows.append({"seed": seed, "solution": "drop_high_inverse_cv", "model": model_name, **evaluate(y_test, proba).__dict__})

                for scale in JITTER_SCALES:
                    x_train_jitter = apply_jitter(x_train_raw, scale, rng)
                    proba = self.fit_standard_model(model_name, x_train_jitter, y_train, x_val_raw, y_val, x_test_drift, seed)
                    rows.append({"seed": seed, "solution": f"jitter_{scale:g}", "model": model_name, **evaluate(y_test, proba).__dict__})

                x_train_drop_jitter = apply_jitter(x_train_raw, 0.05, rng)
                proba = self.fit_standard_model(model_name, x_train_drop_jitter, y_train, x_val_raw, y_val, x_test_drift, seed, keep_inverse_cv)
                rows.append({"seed": seed, "solution": "drop_high_inverse_cv+jitter", "model": model_name, **evaluate(y_test, proba).__dict__})

                proba = self.fit_batch_corrected(model_name, x_train_raw, y_train, x_val_raw, y_val, x_test_drift, seed)
                rows.append({"seed": seed, "solution": "batch_correction", "model": model_name, **evaluate(y_test, proba).__dict__})

            proba = self.fit_offline_coral(x_train_raw, y_train, x_val_raw, y_val, data.x_tar, x_test_drift, seed)
            rows.append({"seed": seed, "solution": "coral", "model": "CORAL+LogReg", **evaluate(y_test, proba).__dict__})

            for eps in DRIFT_GRID:
                no_fix = self.fit_standard_model("LogReg", x_train_raw, y_train, x_val_raw, y_val, apply_global_drift(data.x_tar, eps), seed)
                coral_fix = self.fit_deployment_coral(x_train_raw, y_train, x_val_raw, y_val, apply_global_drift(data.x_tar, eps), seed)
                batch_fix = self.fit_batch_corrected("LogReg", x_train_raw, y_train, x_val_raw, y_val, apply_global_drift(data.x_tar, eps), seed)
                for name, proba in [("LogReg", no_fix), ("CORAL+LogReg", coral_fix), ("Batch correction", batch_fix)]:
                    scores = evaluate(y_test, proba)
                    coral_rows.append({"seed": seed, "drift_eps": eps, "method": name, **scores.__dict__})

        return pd.DataFrame(rows), pd.DataFrame(coral_rows)

    def conformal_reliability(self) -> pd.DataFrame:
        rows = []
        for seed in self.config.seeds:
            data = self.data
            split = make_split(data.y_src, len(data.y_tar), seed)
            y_train = data.y_src[split.train]
            y_val = data.y_src[split.val]
            y_test = data.y_tar
            drop_idx = highest_inverse_cv_feature(data.x_src[split.train])
            keep_inverse_cv = [i for i in range(data.x_src.shape[1]) if i != drop_idx]

            scenarios = {
                "original_all_features": None,
                "drop_high_inverse_cv": keep_inverse_cv,
                "batch_correction": None,
            }
            for scenario, keep in scenarios.items():
                pre = Preprocessor(keep_features=keep).fit(data.x_src[split.train])
                x_train = pre.transform(data.x_src[split.train])
                x_val = pre.transform(data.x_src[split.val])
                model = fit_logreg(x_train, y_train, x_val, y_val, seed)
                temp = fit_temperature(model.predict_proba(x_val), y_val)
                proba_val = apply_temperature(model.predict_proba(x_val), temp)

                for eps in DRIFT_GRID:
                    x_raw = apply_global_drift(data.x_tar, eps)
                    x_test = pre.transform(x_raw)
                    if scenario == "batch_correction":
                        x_test = batch_moment_correction(x_test, x_train)
                    proba = apply_temperature(model.predict_proba(x_test), temp)
                    set_scores = run_aps(proba_val, y_val, proba, y_test, alpha=0.10)
                    clf_scores = evaluate(y_test, proba)
                    rows.append(
                        {
                            "seed": seed,
                            "drift_eps": eps,
                            "scenario": scenario,
                            "coverage": set_scores["coverage"],
                            "avg_set_size": set_scores["avg_set_size"],
                            "singleton_rate": set_scores["singleton_rate"],
                            "macro_f1": clf_scores.macro_f1,
                        }
                    )
        return pd.DataFrame(rows)

    def high_inverse_cv_selection(self) -> pd.DataFrame:
        rows = []
        data = self.data
        for seed in self.config.seeds:
            split = make_split(data.y_src, len(data.y_tar), seed)
            scores = inverse_cv_scores(data.x_src[split.train])
            idx = int(np.argmax(scores))
            rows.append(
                {
                    "seed": seed,
                    "feature": data.feature_names[idx],
                    "tag": data.feature_tags[idx],
                    "inverse_cv_train": float(scores[idx]),
                }
            )
        return pd.DataFrame(rows)

    def logreg_coefficient_audit(self) -> pd.DataFrame:
        rows = []
        data = self.data
        for seed in self.config.seeds:
            split = make_split(data.y_src, len(data.y_tar), seed)
            x_train_raw = data.x_src[split.train]
            y_train = data.y_src[split.train]
            pre = Preprocessor().fit(x_train_raw)
            x_train = pre.transform(x_train_raw)
            x_val = pre.transform(data.x_src[split.val])
            y_val = data.y_src[split.val]
            model = fit_logreg(x_train, y_train, x_val, y_val, seed)

            mean = x_train_raw.mean(axis=0)
            std = np.clip(x_train_raw.std(axis=0), 1e-10, None)
            inverse_cv = np.abs(mean) / std
            abs_coef = np.abs(model.coef_)
            max_abs_coef = abs_coef.max(axis=0)
            l2_coef = np.sqrt(np.square(model.coef_).sum(axis=0))
            max_shift = max_abs_coef * 0.01 * inverse_cv
            l2_shift = l2_coef * 0.01 * inverse_cv

            for j, tag in enumerate(data.feature_tags):
                rows.append(
                    {
                        "seed": seed,
                        "feature": data.feature_names[j],
                        "tag": tag,
                        "max_abs_coef": float(max_abs_coef[j]),
                        "l2_coef": float(l2_coef[j]),
                        "inverse_cv": float(inverse_cv[j]),
                        "max_logit_displacement_1pct": float(max_shift[j]),
                        "l2_logit_displacement_1pct": float(l2_shift[j]),
                    }
                )
        return pd.DataFrame(rows)

    def drift_detection(self) -> pd.DataFrame:
        rows = []
        data = self.data
        split = make_split(data.y_src, len(data.y_tar), seed=DIAGNOSTIC_SPLIT_SEED)
        train = data.x_src[split.train]
        target = data.x_tar[split.target]
        mean = train.mean(axis=0)
        std = np.clip(train.std(axis=0), 1e-10, None)
        for eps in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
            shifted = apply_global_drift(target, eps)
            for batch_size in [50, 100, 200, 400, 800]:
                rng = np.random.RandomState(DIAGNOSTIC_SUBSAMPLE_SEED)
                idx = rng.choice(len(shifted), size=min(batch_size, len(shifted)), replace=False)
                batch = shifted[idx]
                z = np.abs(batch.mean(axis=0) - mean) / (std / np.sqrt(len(batch)) + 1e-10)
                for threshold in DETECTION_THRESHOLDS:
                    rows.append(
                        {
                            "drift_eps": eps,
                            "batch_size": batch_size,
                            "z_threshold": threshold,
                            "detected": bool(np.any(z > threshold)),
                            "max_z": float(np.max(z)),
                            "max_z_feature": data.feature_tags[int(np.argmax(z))],
                            "n_features_flagged": int(np.sum(z > threshold)),
                            "flagged_features": ", ".join(tag for tag, score in zip(data.feature_tags, z) if score > threshold),
                        }
                    )
        return pd.DataFrame(rows)

    def regime_ood(self) -> pd.DataFrame:
        rows = []
        data = self.data
        x_all = np.vstack([data.x_src, data.x_tar])
        y_all = np.concatenate([data.y_src, data.y_tar])
        x_scaled = StandardScaler().fit_transform(x_all)
        for k in [3, 4, 5]:
            clusters = KMeans(n_clusters=k, random_state=REGIME_KMEANS_SEED, n_init=10).fit_predict(x_scaled)
            for cluster in sorted(np.unique(clusters)):
                test_mask = clusters == cluster
                train_mask = ~test_mask
                if test_mask.sum() < 10 or len(np.unique(y_all[train_mask])) < 2:
                    continue
                for seed in self.config.seeds:
                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    train_idx, val_idx = next(splitter.split(x_all[train_mask], y_all[train_mask]))
                    x_pool = x_all[train_mask]
                    y_pool = y_all[train_mask]
                    x_test = x_all[test_mask]
                    y_test = y_all[test_mask]
                    for name in ["LogReg", "HistGB", "TorchMLP"]:
                        proba = self.fit_standard_model(name, x_pool[train_idx], y_pool[train_idx], x_pool[val_idx], y_pool[val_idx], x_test, seed)
                        scores = evaluate(y_test, proba)
                        rows.append({"seed": seed, "condition": f"k{k}_holdout{cluster}", "model": name, **scores.__dict__})
        return pd.DataFrame(rows)

    def standard_arrays(self, seed: int) -> tuple[np.ndarray, ...]:
        data = self.data
        split = make_split(data.y_src, len(data.y_tar), seed)
        pre = Preprocessor().fit(data.x_src[split.train])
        return (
            pre.transform(data.x_src[split.train]),
            data.y_src[split.train],
            pre.transform(data.x_src[split.val]),
            data.y_src[split.val],
            pre.transform(data.x_tar),
            data.y_tar,
            pre.transform(data.x_tar),
        )

    def fit_standard_pipeline(
        self,
        model_name: str,
        x_train_raw: np.ndarray,
        y_train: np.ndarray,
        x_val_raw: np.ndarray,
        y_val: np.ndarray,
        seed: int,
        keep: list[int] | None = None,
    ):
        pre = Preprocessor(keep_features=keep).fit(x_train_raw)
        x_train = pre.transform(x_train_raw)
        x_val = pre.transform(x_val_raw)
        if model_name == "LogReg":
            model = fit_logreg(x_train, y_train, x_val, y_val, seed)
        elif model_name == "HistGB":
            model = fit_hgb(x_train, y_train, x_val, y_val, seed)
        elif model_name == "TorchMLP":
            epochs = 50 if self.config.fast else 250
            patience = 8 if self.config.fast else 30
            model = TorchMLP(MLPConfig(epochs=epochs, patience=patience), seed=seed, device=self.config.device)
            model.fit(x_train, y_train, x_val, y_val)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        temp = fit_temperature(model.predict_proba(x_val), y_val)
        return pre, model, temp

    def fit_standard_model(
        self,
        model_name: str,
        x_train_raw: np.ndarray,
        y_train: np.ndarray,
        x_val_raw: np.ndarray,
        y_val: np.ndarray,
        x_test_raw: np.ndarray,
        seed: int,
        keep: list[int] | None = None,
    ) -> np.ndarray:
        pre, model, temp = self.fit_standard_pipeline(model_name, x_train_raw, y_train, x_val_raw, y_val, seed, keep)
        x_test = pre.transform(x_test_raw)
        return apply_temperature(model.predict_proba(x_test), temp)

    def fit_batch_corrected(
        self,
        model_name: str,
        x_train_raw: np.ndarray,
        y_train: np.ndarray,
        x_val_raw: np.ndarray,
        y_val: np.ndarray,
        x_test_raw: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        pre = Preprocessor().fit(x_train_raw)
        x_train = pre.transform(x_train_raw)
        x_val = pre.transform(x_val_raw)
        x_test = batch_moment_correction(pre.transform(x_test_raw), x_train)
        return self.fit_predict(model_name, x_train, y_train, x_val, y_val, x_test, seed, x_test)

    def fit_deployment_coral(self, x_train_raw: np.ndarray, y_train: np.ndarray, x_val_raw: np.ndarray, y_val: np.ndarray, x_test_raw: np.ndarray, seed: int) -> np.ndarray:
        pre = Preprocessor().fit(x_train_raw)
        x_train = pre.transform(x_train_raw)
        x_val = pre.transform(x_val_raw)
        x_test = pre.transform(x_test_raw)
        state = fit_coral(x_train, x_test)
        x_train_c = apply_coral(x_train, state)
        x_val_c = apply_coral(x_val, state)
        model = fit_logreg(x_train_c, y_train, x_val_c, y_val, seed)
        temp = fit_temperature(model.predict_proba(x_val_c), y_val)
        return apply_temperature(model.predict_proba(x_test), temp)

    def fit_offline_coral(
        self,
        x_train_raw: np.ndarray,
        y_train: np.ndarray,
        x_val_raw: np.ndarray,
        y_val: np.ndarray,
        x_adapt_raw: np.ndarray,
        x_test_raw: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        pre = Preprocessor().fit(x_train_raw)
        x_train = pre.transform(x_train_raw)
        x_val = pre.transform(x_val_raw)
        x_adapt = pre.transform(x_adapt_raw)
        x_test = pre.transform(x_test_raw)
        state = fit_coral(x_train, x_adapt)
        x_train_c = apply_coral(x_train, state)
        x_val_c = apply_coral(x_val, state)
        model = fit_logreg(x_train_c, y_train, x_val_c, y_val, seed)
        temp = fit_temperature(model.predict_proba(x_val_c), y_val)
        return apply_temperature(model.predict_proba(x_test), temp)

    def fit_predict(
        self,
        model_name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        seed: int,
        x_adapt: np.ndarray,
    ) -> np.ndarray:
        if model_name == "LogReg":
            model = fit_logreg(x_train, y_train, x_val, y_val, seed)
            proba_val = model.predict_proba(x_val)
            proba_test = model.predict_proba(x_test)
        elif model_name == "HistGB":
            model = fit_hgb(x_train, y_train, x_val, y_val, seed)
            proba_val = model.predict_proba(x_val)
            proba_test = model.predict_proba(x_test)
        elif model_name == "TorchMLP":
            epochs = 50 if self.config.fast else 250
            patience = 8 if self.config.fast else 30
            model = TorchMLP(MLPConfig(epochs=epochs, patience=patience), seed=seed, device=self.config.device)
            model.fit(x_train, y_train, x_val, y_val)
            proba_val = model.predict_proba(x_val)
            proba_test = model.predict_proba(x_test)
        elif model_name == "CORAL+LogReg":
            state = fit_coral(x_train, x_adapt)
            x_train_c = apply_coral(x_train, state)
            x_val_c = apply_coral(x_val, state)
            model = fit_logreg(x_train_c, y_train, x_val_c, y_val, seed)
            proba_val = model.predict_proba(x_val_c)
            proba_test = model.predict_proba(x_test)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        temp = fit_temperature(proba_val, y_val)
        return apply_temperature(proba_test, temp)

    def write_table(self, name: str, table: pd.DataFrame) -> None:
        table.to_csv(self.tables / name, index=False)

    def write_wide_metric(self, name: str, table: pd.DataFrame, metric: str, aggfunc: str, columns: str = "model") -> None:
        wide = table.pivot_table(index="drift_eps", columns=columns, values=metric, aggfunc=aggfunc)
        wide = wide.reset_index()
        wide.columns.name = None
        wide.to_csv(self.tables / name, index=False)

    def plot_drift(self, drift: pd.DataFrame) -> None:
        agg = aggregate(drift, ["drift_eps"], ["macro_f1"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(agg["drift_eps"], agg["macro_f1_mean"], yerr=agg["macro_f1_std"], marker="o")
        ax.set_xlabel("Multiplicative drift")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Global calibration drift")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures / "drift_global_f1_vs_eps.png", dpi=150)
        plt.close(fig)

    def plot_channel_drift(self, per_channel: pd.DataFrame) -> None:
        agg = aggregate(per_channel, ["tag"], ["macro_f1"]).sort_values("macro_f1_mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(agg["tag"], agg["macro_f1_mean"])
        ax.set_xlabel("Macro-F1 at 5% drift")
        ax.set_title("Per-channel drift sensitivity")
        fig.tight_layout()
        fig.savefig(self.figures / "drift_per_channel_005.png", dpi=150)
        plt.close(fig)

    def plot_channel_drift_heatmap(self, per_channel: pd.DataFrame) -> None:
        agg = aggregate(per_channel, ["tag", "drift_eps"], ["macro_f1"])
        pivot = agg.pivot(index="tag", columns="drift_eps", values="macro_f1_mean")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        image = ax.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{float(v):.2f}" for v in pivot.columns], rotation=45, ha="right")
        ax.set_xlabel("Multiplicative drift")
        ax.set_title("Per-channel drift stress test")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Macro-F1")
        fig.tight_layout()
        fig.savefig(self.figures / "drift_heatmap_channel_eps.png", dpi=150)
        plt.close(fig)

    def plot_amplification(self, amplification: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = amplification.sort_values("inverse_cv")
        ax.barh(data["tag"], data["inverse_cv"])
        ax.set_xlabel("Inverse-CV gain-amplification score")
        ax.set_title("Feature drift amplification")
        fig.tight_layout()
        fig.savefig(self.figures / "inverse_cv_amplification.png", dpi=150)
        plt.close(fig)

    def plot_model_robustness(self, robustness: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, sub in robustness.groupby("model"):
            agg = aggregate(sub, ["drift_eps"], ["macro_f1"])
            ax.errorbar(
                agg["drift_eps"],
                agg["macro_f1_mean"],
                yerr=agg["macro_f1_std"],
                marker="o",
                label=model,
            )
        ax.set_xlabel("Multiplicative drift")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Model robustness under drift")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures / "model_robustness_drift.png", dpi=150)
        plt.close(fig)

    def plot_preprocessing_robustness(self, preprocessing: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, sub in preprocessing.groupby("preprocessing"):
            agg = aggregate(sub, ["drift_eps"], ["macro_f1"])
            ax.plot(agg["drift_eps"], agg["macro_f1_mean"], marker="o", label=name)
        ax.set_xlabel("Multiplicative drift")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Preprocessing robustness")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures / "preprocessing_robustness.png", dpi=150)
        plt.close(fig)

    def plot_remedies(self, remedies: pd.DataFrame) -> None:
        sub = remedies[remedies["model"] == "LogReg"]
        agg = aggregate(sub, ["solution"], ["macro_f1"]).sort_values("macro_f1_mean")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(agg["solution"], agg["macro_f1_mean"])
        ax.set_xlabel("Macro-F1")
        ax.set_title("Remedies at 5% drift")
        fig.tight_layout()
        fig.savefig(self.figures / "remedies_logreg.png", dpi=150)
        plt.close(fig)

    def plot_coral(self, coral: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, sub in coral.groupby("method"):
            agg = aggregate(sub, ["drift_eps"], ["macro_f1"])
            ax.plot(agg["drift_eps"], agg["macro_f1_mean"], marker="o", label=method)
        ax.set_xlabel("Multiplicative drift")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Deployment-time alignment")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures / "coral_under_drift.png", dpi=150)
        plt.close(fig)

    def plot_conformal(self, conformal: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for scenario, sub in conformal.groupby("scenario"):
            agg = aggregate(sub, ["drift_eps"], ["coverage"])
            ax.plot(agg["drift_eps"], agg["coverage_mean"], marker="o", label=scenario)
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Multiplicative drift")
        ax.set_ylabel("APS coverage")
        ax.set_title("Conformal coverage under drift")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures / "conformal_solutions_coverage_vs_drift.png", dpi=150)
        plt.close(fig)

    def plot_detection(self, detection: pd.DataFrame) -> None:
        agg = detection.groupby("drift_eps", as_index=False)["max_z"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(agg["drift_eps"], agg["max_z"], marker="o")
        ax.axhline(3.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Multiplicative drift")
        ax.set_ylabel("Max z-score")
        ax.set_title("Batch drift detection")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures / "drift_detection.png", dpi=150)
        plt.close(fig)

    def plot_regime(self, regime: pd.DataFrame) -> None:
        if regime.empty:
            return
        agg = aggregate(regime, ["condition", "model"], ["macro_f1"])
        pivot = agg.pivot(index="condition", columns="model", values="macro_f1_mean")
        ax = pivot.plot(kind="bar", figsize=(8, 4))
        ax.set_ylabel("Macro-F1")
        ax.set_title("Regime holdout check")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(self.figures / "regime_ood_bar.png", dpi=150)
        plt.close(fig)
