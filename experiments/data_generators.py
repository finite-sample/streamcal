"""
Data generators for calibration experiments.
"""

import numpy as np
from typing import Iterator, Callable

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


DRIFT_SCENARIOS = {
    "stationary": lambda t, T: 0.0,
    "linear_up": lambda t, T: 0.7 * t / T,
    "linear_down": lambda t, T: 0.7 * (1 - t / T),
    "sinusoidal": lambda t, T: 0.35 * np.sin(2 * np.pi * t / T),
    "sudden_shift": lambda t, T: 0.0 if t < T / 2 else 0.5,
    "random_walk": "random_walk",
}


def synthetic_stream(
    n_total: int = 200_000,
    batch_size: int = 5_000,
    n_buckets: int = 100,
    drift_scenario: str = "stationary",
    miscal_stretch: float = 1.5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic streaming data for theory validation.

    Args:
        n_total: Total number of observations
        batch_size: Observations per batch
        n_buckets: Number of buckets (unused, kept for API consistency)
        drift_scenario: One of DRIFT_SCENARIOS keys
        miscal_stretch: Logit stretch factor for miscalibration
        seed: Random seed

    Yields:
        Tuple of (raw_probabilities, binary_outcomes)
    """
    rng = np.random.default_rng(seed)
    T = n_total // batch_size
    drift_fn = DRIFT_SCENARIOS.get(drift_scenario)

    random_walk_state = 0.0

    for batch_idx in range(T):
        size = min(batch_size, n_total - batch_idx * batch_size)

        p_raw = rng.beta(2, 5, size)
        p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)

        logits = np.log(p_raw / (1 - p_raw))
        stretched_logits = logits * miscal_stretch

        if drift_scenario == "random_walk":
            random_walk_state += rng.normal(0, 0.05)
            random_walk_state = np.clip(random_walk_state, -0.7, 0.7)
            mu = random_walk_state
        else:
            mu = drift_fn(batch_idx, T)

        true_logits = stretched_logits + mu
        true_probs = 1 / (1 + np.exp(-true_logits))
        y = rng.binomial(1, true_probs)

        yield p_raw, y


def semi_synthetic_stream(
    domain: str = "ad_ctr",
    n_train: int = 60_000,
    n_stream: int = 40_000,
    batch_size: int = 2_000,
    n_features: int = 20,
    drift_scenario: str = "linear_up",
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate semi-synthetic streaming data mimicking real datasets.

    Trains a LightGBM model on simulated data, then streams with drift.

    Args:
        domain: One of "ad_ctr" (~5% positive), "income" (~25% positive), "loan" (~10% positive)
        n_train: Training set size
        n_stream: Stream size (deployment set)
        batch_size: Observations per batch
        n_features: Number of features
        drift_scenario: Drift pattern
        seed: Random seed

    Yields:
        Tuple of (raw_probabilities, binary_outcomes)
    """
    rng = np.random.default_rng(seed)

    base_rates = {"ad_ctr": 0.05, "income": 0.25, "loan": 0.10}
    base_rate = base_rates.get(domain, 0.1)

    intercept = np.log(base_rate / (1 - base_rate))

    X_train = rng.standard_normal((n_train, n_features))
    true_weights = rng.standard_normal(n_features) * 0.5
    logits_train = X_train @ true_weights + intercept
    p_train = 1 / (1 + np.exp(-logits_train))
    y_train = rng.binomial(1, p_train)

    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

    X_stream = rng.standard_normal((n_stream, n_features))
    drift_fn = DRIFT_SCENARIOS.get(drift_scenario)
    T = n_stream // batch_size
    random_walk_state = 0.0

    for batch_idx in range(T):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_stream)
        X_batch = X_stream[start:end]

        if HAS_LIGHTGBM:
            p_raw = model.predict_proba(X_batch)[:, 1]
        else:
            p_raw = model.predict_proba(X_batch)[:, 1]

        p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)

        if drift_scenario == "random_walk":
            random_walk_state += rng.normal(0, 0.05)
            random_walk_state = np.clip(random_walk_state, -0.7, 0.7)
            mu = random_walk_state
        else:
            mu = drift_fn(batch_idx, T)

        true_logits = X_batch @ true_weights + intercept + mu
        true_probs = 1 / (1 + np.exp(-true_logits))
        y = rng.binomial(1, true_probs)

        yield p_raw, y


def production_scale_stream(
    n_total: int = 1_000_000,
    batch_size: int = 10_000,
    n_buckets: int = 100,
    base_ctr: float = 0.05,
    n_features: int = 50,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Generate production-scale ad click simulation.

    Args:
        n_total: Total impressions
        batch_size: Impressions per batch
        n_buckets: Number of buckets (unused)
        base_ctr: Base click-through rate
        n_features: Number of features
        seed: Random seed

    Yields:
        Tuple of (raw_probabilities, binary_outcomes)
    """
    rng = np.random.default_rng(seed)

    intercept = np.log(base_ctr / (1 - base_ctr))
    n_train = n_total // 10
    T = n_total // batch_size

    X_train = rng.standard_normal((n_train, n_features))
    true_weights = rng.standard_normal(n_features) * 0.3
    logits_train = X_train @ true_weights + intercept
    p_train = 1 / (1 + np.exp(-logits_train))
    y_train = rng.binomial(1, p_train)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=500, n_jobs=1)
    model.fit(X_train, y_train)

    for batch_idx in range(T):
        X_batch = rng.standard_normal((batch_size, n_features))

        p_raw = model.predict_proba(X_batch)[:, 1]
        p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)

        mu = 0.2 * np.sin(2 * np.pi * batch_idx / T)

        true_logits = X_batch @ true_weights + intercept + mu
        true_probs = 1 / (1 + np.exp(-true_logits))
        y = rng.binomial(1, true_probs)

        yield p_raw, y


def get_data_generator(
    setting: str, **kwargs
) -> Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]]:
    """
    Factory function for data generators.

    Args:
        setting: One of "synthetic", "semi_synthetic", "production"
        **kwargs: Additional arguments for the generator

    Returns:
        Data generator function
    """
    generators = {
        "synthetic": synthetic_stream,
        "semi_synthetic": semi_synthetic_stream,
        "production": production_scale_stream,
    }

    if setting not in generators:
        raise ValueError(f"Unknown setting: {setting}")

    return lambda: generators[setting](**kwargs)
