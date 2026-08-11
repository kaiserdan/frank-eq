import numpy as np

from frank_eq.rate_compute.calibration import (
    balanced_accuracy,
    brier_score,
    fit_platt_calibrator,
)


def test_platt_calibration_can_recover_a_stably_inverted_answer_channel() -> None:
    targets = np.asarray([0.02, 0.02, 0.98, 0.98, 0.02, 0.98], dtype=np.float64)
    inverted_scores = np.asarray([3.0, 2.0, -3.0, -2.0, 1.5, -1.5], dtype=np.float64)

    calibrator = fit_platt_calibrator(inverted_scores, targets, l2=1e-3, max_steps=200)
    prediction = calibrator.predict(inverted_scores)

    assert calibrator.alpha < 0.0
    assert balanced_accuracy(targets, prediction) == 1.0
    assert brier_score(targets, prediction) < brier_score(
        targets, np.full_like(targets, targets.mean())
    )


def test_platt_calibration_accepts_smoothed_binary_targets() -> None:
    scores = np.linspace(-3.0, 3.0, 20)
    targets = np.where(scores > 0.0, 0.98, 0.02)
    calibrator = fit_platt_calibrator(scores, targets)
    prediction = calibrator.predict(scores)

    assert prediction.shape == targets.shape
    assert np.all(np.isfinite(prediction))
    assert np.all((prediction > 0.0) & (prediction < 1.0))
