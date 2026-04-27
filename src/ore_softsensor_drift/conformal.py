from __future__ import annotations

import numpy as np

from .data import LABELS


def aps_scores(proba: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    scores = np.zeros(len(y_true), dtype=float)
    for i, y in enumerate(y_true):
        true_idx = label_to_idx[int(y)]
        order = np.argsort(-proba[i])
        rank = int(np.where(order == true_idx)[0][0])
        scores[i] = float(proba[i, order[: rank + 1]].sum())
    return scores


def aps_threshold(proba_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    scores = np.sort(aps_scores(proba_cal, y_cal))
    k = int(np.ceil((len(scores) + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), len(scores) - 1)
    return float(scores[k])


def aps_sets(proba: np.ndarray, threshold: float) -> list[list[int]]:
    out = []
    for row in proba:
        order = np.argsort(-row)
        total = 0.0
        chosen = []
        for idx in order:
            total += float(row[idx])
            chosen.append(LABELS[int(idx)])
            if total > threshold:
                break
        out.append(chosen)
    return out


def evaluate_sets(sets: list[list[int]], y_true: np.ndarray) -> dict[str, float]:
    sizes = np.array([len(s) for s in sets], dtype=float)
    coverage = np.mean([int(int(y) in s) for y, s in zip(y_true, sets)])
    return {
        "coverage": float(coverage),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float(np.mean(sizes == 1)),
    }


def run_aps(proba_cal: np.ndarray, y_cal: np.ndarray, proba_test: np.ndarray, y_test: np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    threshold = aps_threshold(proba_cal, y_cal, alpha)
    metrics = evaluate_sets(aps_sets(proba_test, threshold), y_test)
    metrics["q_hat"] = threshold
    return metrics
