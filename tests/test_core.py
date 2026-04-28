from __future__ import annotations

import numpy as np

from ore_softsensor_drift.data import (
    apply_channel_drift,
    apply_global_drift,
    batch_moment_correction,
    highest_inverse_cv_feature,
    inverse_cv_scores,
)


def test_global_and_channel_drift_math() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    np.testing.assert_allclose(apply_global_drift(x, 0.10), x * 1.10)
    shifted = apply_channel_drift(x, channel=1, eps=0.25)

    np.testing.assert_allclose(shifted[:, 0], x[:, 0])
    np.testing.assert_allclose(shifted[:, 1], x[:, 1] * 1.25)


def test_inverse_cv_scores_select_near_constant_feature() -> None:
    x = np.array(
        [
            [0.0, 4.00],
            [1.0, 4.01],
            [2.0, 3.99],
            [3.0, 4.00],
        ],
        dtype=np.float32,
    )

    scores = inverse_cv_scores(x)

    assert scores[1] > scores[0] * 100
    assert highest_inverse_cv_feature(x) == 1


def test_batch_moment_correction_matches_train_moments() -> None:
    rng = np.random.RandomState(0)
    x_train = rng.normal(loc=[0.0, 2.0], scale=[1.0, 0.5], size=(100, 2)).astype(np.float32)
    x_test = rng.normal(loc=[3.0, -4.0], scale=[2.0, 1.5], size=(80, 2)).astype(np.float32)

    corrected = batch_moment_correction(x_test, x_train)

    np.testing.assert_allclose(corrected.mean(axis=0), x_train.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(corrected.std(axis=0), x_train.std(axis=0), atol=1e-5)
