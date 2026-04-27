from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .data import LABELS


@dataclass(frozen=True)
class Scores:
    macro_f1: float
    accuracy: float
    ece: float
    brier: float


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> Scores:
    pred_idx = proba.argmax(axis=1)
    pred = np.array([LABELS[i] for i in pred_idx], dtype=int)
    return Scores(
        macro_f1=float(f1_score(y_true, pred, average="macro")),
        accuracy=float(accuracy_score(y_true, pred)),
        ece=expected_calibration_error(y_true, proba),
        brier=brier_score(y_true, proba),
    )


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    idx = np.array([label_to_idx[int(y)] for y in y_true], dtype=int)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(idx)), idx] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    idx = np.array([label_to_idx[int(y)] for y in y_true], dtype=int)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if mask.any():
            out += abs(correct[mask].mean() - conf[mask].mean()) * mask.mean()
    return float(out)


def fit_temperature(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    logits = torch.tensor(np.log(np.clip(proba_val, 1e-12, 1.0)), dtype=torch.float32)
    y = torch.tensor(y_val.astype(int) - 1, dtype=torch.long)
    t = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / torch.clamp(t, min=1e-4), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(t.detach(), min=1e-4).item())


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    scaled = logits / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def aggregate(df: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    out = df.groupby(groups, dropna=False)[metrics].agg(["mean", "std"])
    out.columns = [f"{name}_{stat}" for name, stat in out.columns]
    return out.reset_index()
