from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_logreg(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> LogisticRegression:
    best = None
    best_f1 = -np.inf
    for c in (0.01, 0.1, 1.0, 10.0, 30.0):
        model = LogisticRegression(C=c, max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed)
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average="macro")
        if score > best_f1:
            best = model
            best_f1 = score
    if best is None:
        raise RuntimeError("Logistic regression failed.")
    return best


def fit_hgb(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    best = None
    best_f1 = -np.inf
    for lr, depth in ((0.03, 2), (0.05, 3), (0.08, 3), (0.1, 4)):
        model = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=depth,
            max_iter=600,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average="macro")
        if score > best_f1:
            best = model
            best_f1 = score
    if best is None:
        raise RuntimeError("Histogram gradient boosting failed.")
    return best


@dataclass(frozen=True)
class MLPConfig:
    hidden: tuple[int, int] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 250
    patience: int = 30


class MLPNet(nn.Module):
    def __init__(self, n_features: int, config: MLPConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, config.hidden[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden[0], config.hidden[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden[1], 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchMLP:
    def __init__(self, config: MLPConfig, seed: int, device: str = "cpu") -> None:
        self.config = config
        self.seed = seed
        self.device = device
        self.model: MLPNet | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "TorchMLP":
        set_seed(self.seed)
        config = self.config
        model = MLPNet(x_train.shape[1], config).to(self.device)
        classes = np.array(sorted(np.unique(y_train)), dtype=int)
        class_weight = compute_class_weight("balanced", classes=classes, y=y_train)
        weight = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train.astype(int) - 1, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        x_val_t = torch.tensor(x_val, dtype=torch.float32, device=self.device)
        best_state = None
        best_f1 = -np.inf
        stale = 0
        for _ in range(config.epochs):
            model.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb, weight=weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(x_val_t).argmax(dim=1).cpu().numpy() + 1
            score = f1_score(y_val, pred, average="macro")
            if score > best_f1 + 1e-4:
                best_f1 = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MLP is not fitted.")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32, device=self.device))
            return F.softmax(logits, dim=1).cpu().numpy()


def fit_coral(x_source: np.ndarray, x_target: np.ndarray, reg: float = 1e-3) -> dict[str, np.ndarray]:
    mean_s = x_source.mean(axis=0, keepdims=True)
    mean_t = x_target.mean(axis=0, keepdims=True)
    cs = covariance(x_source) + reg * np.eye(x_source.shape[1])
    ct = covariance(x_target) + reg * np.eye(x_source.shape[1])
    transform = inv_sqrtm(cs) @ sqrtm(ct)
    return {"mean_s": mean_s, "mean_t": mean_t, "transform": transform}


def apply_coral(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    return (x - state["mean_s"]) @ state["transform"] + state["mean_t"]


def covariance(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / max(1, x.shape[0] - 1)


def sqrtm(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    values = np.clip(values, 1e-8, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def inv_sqrtm(a: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(a)
    values = np.clip(values, 1e-8, None)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T
