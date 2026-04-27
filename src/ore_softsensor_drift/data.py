from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


LABELS = [1, 2, 3]

FEATURE_TAGS = [
    "FIT2003",
    "FIT2005",
    "PIT2001",
    "DIT2001",
    "Sand",
    "AT2001",
    "LIT2004",
    "DIT2002",
]


@dataclass(frozen=True)
class Dataset:
    x_src: np.ndarray
    y_src: np.ndarray
    x_tar: np.ndarray
    y_tar: np.ndarray
    feature_names: list[str]
    feature_tags: list[str]


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    target: np.ndarray


@dataclass
class Preprocessor:
    keep_features: list[int] | None = None
    scaler: StandardScaler | None = None
    fitted_keep: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "Preprocessor":
        keep = self.keep_features
        if keep is None:
            self.fitted_keep = np.arange(x_train.shape[1], dtype=int)
        else:
            self.fitted_keep = np.array(sorted(keep), dtype=int)
        self.scaler = StandardScaler()
        self.scaler.fit(x_train[:, self.fitted_keep])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.fitted_keep is None:
            raise RuntimeError("Preprocessor is not fitted.")
        return self.scaler.transform(x[:, self.fitted_keep]).astype(np.float32)


def load_dataset(data_dir: str | Path) -> Dataset:
    path = Path(data_dir)
    required = ["X_src.csv", "Y_src.csv", "X_tar.csv", "Y_tar.csv"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {path}: {missing}")

    x_src_raw = pd.read_csv(path / "X_src.csv", header=None).iloc[1:, :]
    x_tar_raw = pd.read_csv(path / "X_tar.csv", header=None).iloc[1:, :]
    y_src = pd.read_csv(path / "Y_src.csv", header=None).iloc[:, 0].to_numpy(dtype=int)
    y_tar = pd.read_csv(path / "Y_tar.csv", header=None).iloc[:, 0].to_numpy(dtype=int)

    x_src = x_src_raw.to_numpy(dtype=np.float32).T
    x_tar = x_tar_raw.to_numpy(dtype=np.float32).T

    if x_src.shape[0] != len(y_src):
        raise ValueError("X_src and Y_src have incompatible sample counts.")
    if x_tar.shape[0] != len(y_tar):
        raise ValueError("X_tar and Y_tar have incompatible sample counts.")

    return Dataset(
        x_src=x_src,
        y_src=y_src,
        x_tar=x_tar,
        y_tar=y_tar,
        feature_names=[f"row{i + 1}" for i in range(x_src.shape[1])],
        feature_tags=list(FEATURE_TAGS),
    )


def make_split(y_src: np.ndarray, n_target: int, seed: int) -> Split:
    idx = np.arange(len(y_src))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, val = next(splitter.split(idx, y_src))
    return Split(train=train.astype(int), val=val.astype(int), target=np.arange(n_target, dtype=int))


def apply_global_drift(x: np.ndarray, eps: float) -> np.ndarray:
    return (x * (1.0 + float(eps))).astype(np.float32)


def apply_channel_drift(x: np.ndarray, channel: int, eps: float) -> np.ndarray:
    out = x.copy()
    out[:, int(channel)] *= 1.0 + float(eps)
    return out.astype(np.float32)


def apply_jitter(x: np.ndarray, scale: float, rng: np.random.RandomState) -> np.ndarray:
    std = np.clip(x.std(axis=0, keepdims=True), 1e-8, None)
    noise = rng.randn(*x.shape).astype(np.float32) * std * float(scale)
    return (x + noise).astype(np.float32)


def batch_moment_correction(x_test: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    train_mean = x_train.mean(axis=0, keepdims=True)
    train_std = np.clip(x_train.std(axis=0, keepdims=True), 1e-8, None)
    test_mean = x_test.mean(axis=0, keepdims=True)
    test_std = np.clip(x_test.std(axis=0, keepdims=True), 1e-8, None)
    return ((x_test - test_mean) / test_std * train_std + train_mean).astype(np.float32)


def vulnerability_scores(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0)
    std = np.clip(np.std(x, axis=0), 1e-12, None)
    return np.abs(mean) / std


def highest_vi_feature(x_train: np.ndarray) -> int:
    return int(np.argmax(vulnerability_scores(x_train)))


def vulnerability_table(data: Dataset) -> pd.DataFrame:
    rows = []
    for j, tag in enumerate(data.feature_tags):
        x = data.x_src[:, j]
        mean = float(np.mean(x))
        std = float(np.std(x))
        vi = abs(mean) / max(std, 1e-12)
        rows.append(
            {
                "feature": data.feature_names[j],
                "tag": tag,
                "mean": mean,
                "std": std,
                "VI": vi,
                "drift_1pct_sigma": vi * 0.01,
            }
        )
    return pd.DataFrame(rows)
