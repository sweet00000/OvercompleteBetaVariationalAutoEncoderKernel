import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.integrate import solve_ivp
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from obvaekernel import OBVAEKernel


@dataclass
class ProfileConfig:
    mode: str
    target_components: int
    obvaek_epochs: int
    obvaek_hidden_dim: int
    obvaek_expansion_factor: int
    obvaek_beta: float
    cv_splits: int
    trust_neighbors: int
    manifold_lift_dim: int
    chaotic_lift_dim: int
    manifold_noise: float
    chaotic_noise: float
    trajectories_per_family: int
    integration_points: int
    window_size: int
    window_stride: int
    weather_sample_cap: int


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[ProfileConfig, np.random.Generator, int], Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]]
    task_type: str
    has_labels: bool
    sample_cap_quick: Optional[int] = None
    sample_cap_full: Optional[int] = None


@dataclass
class MethodSpec:
    name: str
    factory: Callable[[], Any]
    supports_inverse: bool


@dataclass
class MetricRecord:
    dataset: str
    method: str
    metric_name: str
    metric_value: float
    higher_is_better: bool


CORE_METRICS = ("trustworthiness", "silhouette", "ari", "cv_accuracy")

METRIC_DIRECTIONS = {
    "trustworthiness": True,
    "silhouette": True,
    "ari": True,
    "cv_accuracy": True,
    "reconstruction_mse": False,
    "fit_transform_seconds": False,
}


def get_profile_config(mode: str) -> ProfileConfig:
    if mode == "quick":
        return ProfileConfig(
            mode="quick",
            target_components=12,
            obvaek_epochs=30,
            obvaek_hidden_dim=48,
            obvaek_expansion_factor=8,
            obvaek_beta=2.0,
            cv_splits=3,
            trust_neighbors=7,
            manifold_lift_dim=48,
            chaotic_lift_dim=64,
            manifold_noise=0.02,
            chaotic_noise=0.01,
            trajectories_per_family=3,
            integration_points=600,
            window_size=10,
            window_stride=6,
            weather_sample_cap=900,
        )

    return ProfileConfig(
        mode="full",
        target_components=24,
        obvaek_epochs=120,
        obvaek_hidden_dim=64,
        obvaek_expansion_factor=12,
        obvaek_beta=2.5,
        cv_splits=5,
        trust_neighbors=10,
        manifold_lift_dim=64,
        chaotic_lift_dim=80,
        manifold_noise=0.02,
        chaotic_noise=0.01,
        trajectories_per_family=6,
        integration_points=900,
        window_size=12,
        window_stride=5,
        weather_sample_cap=2500,
    )


def nonlinear_feature_lift(
    x: np.ndarray,
    out_dim: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    k = min(8, x.shape[1])
    interaction_features = []
    for i in range(k):
        for j in range(i + 1, k):
            interaction_features.append((x[:, i] * x[:, j])[:, None])
    interaction_block = (
        np.hstack(interaction_features).astype(np.float32) if interaction_features else np.empty((x.shape[0], 0), dtype=np.float32)
    )

    feature_block = np.hstack(
        [
            x,
            x**2,
            np.sin(x),
            np.cos(x),
            interaction_block,
        ]
    ).astype(np.float32)

    if feature_block.shape[1] > out_dim:
        proj = rng.normal(0.0, 1.0, size=(feature_block.shape[1], out_dim)).astype(np.float32)
        feature_block = feature_block @ (proj / np.sqrt(feature_block.shape[1], dtype=np.float32))
    elif feature_block.shape[1] < out_dim:
        extra_dim = out_dim - feature_block.shape[1]
        proj = rng.normal(0.0, 1.0, size=(feature_block.shape[1], extra_dim)).astype(np.float32)
        extra = feature_block @ (proj / np.sqrt(feature_block.shape[1], dtype=np.float32))
        feature_block = np.hstack([feature_block, extra]).astype(np.float32)

    noise = noise_scale * rng.normal(0.0, 1.0, size=feature_block.shape).astype(np.float32)
    return (feature_block + noise).astype(np.float32)


def build_windows(traj: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    rows = []
    for start in range(0, max(0, traj.shape[0] - window_size), stride):
        window = traj[start : start + window_size]
        if window.shape[0] == window_size:
            rows.append(window.reshape(-1))
    if not rows:
        return np.empty((0, traj.shape[1] * window_size), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def lorenz_rhs(_: float, s: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    x, y, z = s
    return np.array(
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z],
        dtype=np.float64,
    )


def rossler_rhs(_: float, s: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x, y, z = s
    return np.array(
        [-y - z, x + a * y, b + z * (x - c)],
        dtype=np.float64,
    )


def _simulate_chaotic_dataset(
    profile: ProfileConfig,
    rng: np.random.Generator,
    system_name: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    t_eval = np.linspace(0.0, 40.0, profile.integration_points, dtype=np.float64)

    if system_name == "lorenz":
        families = [
            {"init": np.array([0.1, 1.0, 1.05]), "params": (10.0, 8.0 / 3.0, 28.0)},
            {"init": np.array([1.2, -1.0, 20.0]), "params": (10.0, 8.0 / 3.0, 35.0)},
            {"init": np.array([-8.0, 7.0, 27.0]), "params": (10.0, 8.0 / 3.0, 24.0)},
            {"init": np.array([2.0, 3.0, 18.0]), "params": (14.0, 3.0, 35.0)},
        ]
        rhs = lorenz_rhs
        jitter = 0.35
    else:
        families = [
            {"init": np.array([1.0, 0.0, 0.0]), "params": (0.2, 0.2, 5.7)},
            {"init": np.array([2.0, -1.0, 1.5]), "params": (0.15, 0.2, 6.0)},
            {"init": np.array([-1.0, 2.0, 0.2]), "params": (0.2, 0.15, 5.9)},
            {"init": np.array([0.5, -2.0, 3.0]), "params": (0.25, 0.2, 5.5)},
        ]
        rhs = rossler_rhs
        jitter = 0.3

    windows_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    for family_idx, family in enumerate(families):
        for _ in range(profile.trajectories_per_family):
            init = family["init"] + rng.normal(0.0, jitter, size=3)
            sol = solve_ivp(
                fun=lambda t, s, p=family["params"]: rhs(t, s, *p),
                t_span=(float(t_eval[0]), float(t_eval[-1])),
                y0=init.astype(np.float64),
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8,
            )
            if not sol.success:
                continue

            trajectory = sol.y.T.astype(np.float32)
            warmup = max(20, int(0.1 * trajectory.shape[0]))
            trajectory = trajectory[warmup:]
            windows = build_windows(
                trajectory,
                window_size=profile.window_size,
                stride=profile.window_stride,
            )
            if windows.shape[0] == 0:
                continue
            windows_all.append(windows)
            labels_all.append(np.full(windows.shape[0], family_idx, dtype=np.int64))

    if not windows_all:
        raise RuntimeError(f"Failed to generate {system_name} chaotic windows.")

    x = np.vstack(windows_all).astype(np.float32)
    y = np.concatenate(labels_all).astype(np.int64)
    x = nonlinear_feature_lift(
        x=x,
        out_dim=profile.chaotic_lift_dim,
        rng=rng,
        noise_scale=profile.chaotic_noise,
    )

    return x, y, {"source": f"{system_name}_ode", "n_features": int(x.shape[1])}


def load_lorenz_lifted(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    del seed
    return _simulate_chaotic_dataset(profile, rng, system_name="lorenz")


def load_rossler_lifted(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    del seed
    return _simulate_chaotic_dataset(profile, rng, system_name="rossler")


def _bin_continuous_labels(values: np.ndarray, bins: int = 6) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    labels = np.digitize(values, qs[1:-1], right=False)
    return labels.astype(np.int64)


def load_swiss_roll_lifted(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_samples = 1300 if profile.mode == "quick" else 2500
    x_raw, t = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=seed)
    y = _bin_continuous_labels(t, bins=6)
    x = nonlinear_feature_lift(
        x=x_raw.astype(np.float32),
        out_dim=profile.manifold_lift_dim,
        rng=rng,
        noise_scale=profile.manifold_noise,
    )
    return x, y, {"source": "make_swiss_roll", "n_features": int(x.shape[1])}


def load_s_curve_lifted(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_samples = 1300 if profile.mode == "quick" else 2500
    x_raw, t = make_s_curve(n_samples=n_samples, noise=0.05, random_state=seed)
    y = _bin_continuous_labels(t, bins=6)
    x = nonlinear_feature_lift(
        x=x_raw.astype(np.float32),
        out_dim=profile.manifold_lift_dim,
        rng=rng,
        noise_scale=profile.manifold_noise,
    )
    return x, y, {"source": "make_s_curve", "n_features": int(x.shape[1])}


def load_digits_dataset(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    del profile, rng, seed
    ds = load_digits()
    return ds.data.astype(np.float32), ds.target.astype(np.int64), {"source": "sklearn.load_digits"}


def load_wine_dataset(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    del profile, rng, seed
    ds = load_wine()
    return ds.data.astype(np.float32), ds.target.astype(np.int64), {"source": "sklearn.load_wine"}


def load_breast_cancer_dataset(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    del profile, rng, seed
    ds = load_breast_cancer()
    return ds.data.astype(np.float32), ds.target.astype(np.int64), {"source": "sklearn.load_breast_cancer"}


def load_weather_dataset(
    profile: ProfileConfig, rng: np.random.Generator, seed: int
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    del seed
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=33&longitude=-84&start_date=2022-01-01&end_date=2025-12-31"
        "&hourly=rain,temperature_2m,relative_humidity_2m,precipitation,"
        "surface_pressure,cloud_cover,wind_speed_10m,vapor_pressure_deficit"
    )
    res = requests.get(url, timeout=60)
    res.raise_for_status()
    payload = res.json()
    if "hourly" not in payload:
        raise RuntimeError("Weather response missing 'hourly' payload.")

    df = pd.DataFrame(payload["hourly"])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")
    df = df.ffill().fillna(0.0)
    x = df.to_numpy(dtype=np.float32)

    cap = profile.weather_sample_cap
    if x.shape[0] > cap:
        idx = rng.permutation(x.shape[0])[:cap]
        x = x[idx]
    return x, None, {"source": "open-meteo-archive", "n_features": int(x.shape[1])}


def subsample_dataset(
    x: np.ndarray,
    y: Optional[np.ndarray],
    cap: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if cap is None or x.shape[0] <= cap:
        return x, y
    idx = rng.permutation(x.shape[0])[:cap]
    x_sub = x[idx]
    if y is None:
        return x_sub, None
    return x_sub, y[idx]


def build_dataset_registry(mode: str, include_weather: bool) -> List[DatasetSpec]:
    if mode == "quick":
        specs = [
            DatasetSpec("lorenz_lifted", load_lorenz_lifted, task_type="classification", has_labels=True, sample_cap_quick=900),
            DatasetSpec("swiss_roll_lifted", load_swiss_roll_lifted, task_type="classification", has_labels=True, sample_cap_quick=900),
            DatasetSpec("digits", load_digits_dataset, task_type="classification", has_labels=True, sample_cap_quick=900),
        ]
    else:
        specs = [
            DatasetSpec("lorenz_lifted", load_lorenz_lifted, task_type="classification", has_labels=True, sample_cap_full=1800),
            DatasetSpec("rossler_lifted", load_rossler_lifted, task_type="classification", has_labels=True, sample_cap_full=1800),
            DatasetSpec("swiss_roll_lifted", load_swiss_roll_lifted, task_type="classification", has_labels=True, sample_cap_full=1800),
            DatasetSpec("s_curve_lifted", load_s_curve_lifted, task_type="classification", has_labels=True, sample_cap_full=1800),
            DatasetSpec("digits", load_digits_dataset, task_type="classification", has_labels=True, sample_cap_full=1797),
            DatasetSpec("wine", load_wine_dataset, task_type="classification", has_labels=True, sample_cap_full=178),
            DatasetSpec("breast_cancer", load_breast_cancer_dataset, task_type="classification", has_labels=True, sample_cap_full=569),
        ]

    if include_weather:
        specs.append(
            DatasetSpec(
                "weather_optional",
                load_weather_dataset,
                task_type="unsupervised",
                has_labels=False,
                sample_cap_quick=900,
                sample_cap_full=2500,
            )
        )

    return specs


def _resolve_n_components(x: np.ndarray, profile: ProfileConfig) -> int:
    upper = max(2, min(x.shape[0] - 1, x.shape[1] - 1))
    return max(2, min(profile.target_components, upper))


def build_method_specs(
    x: np.ndarray,
    profile: ProfileConfig,
    seed: int,
) -> List[MethodSpec]:
    n_components = _resolve_n_components(x, profile)
    feature_count = max(1, x.shape[1])
    base_gamma = 1.0 / feature_count

    return [
        MethodSpec(
            name="OBVAEKernel",
            factory=lambda: OBVAEKernel(
                latent_dim="auto",
                expansion_factor=profile.obvaek_expansion_factor,
                hidden_dim=profile.obvaek_hidden_dim,
                beta=profile.obvaek_beta,
                epochs=profile.obvaek_epochs,
                lr=0.004 if profile.mode == "quick" else 0.003,
                batch_size=128 if profile.mode == "full" else 96,
                random_state=seed,
                verbose=False,
            ),
            supports_inverse=True,
        ),
        MethodSpec(
            name="PCA",
            factory=lambda: PCA(n_components=n_components, random_state=seed),
            supports_inverse=True,
        ),
        MethodSpec(
            name="KernelPCA_RBF",
            factory=lambda: KernelPCA(
                n_components=n_components,
                kernel="rbf",
                gamma=max(0.005, base_gamma),
                fit_inverse_transform=True,
                alpha=1e-3,
                random_state=seed,
            ),
            supports_inverse=True,
        ),
        MethodSpec(
            name="KernelPCA_POLY",
            factory=lambda: KernelPCA(
                n_components=n_components,
                kernel="poly",
                gamma=max(0.005, base_gamma),
                degree=3,
                coef0=1.0,
                fit_inverse_transform=True,
                alpha=1e-3,
                random_state=seed,
            ),
            supports_inverse=True,
        ),
    ]


def build_kpca_fallback(
    method_name: str,
    x: np.ndarray,
    n_components: int,
    seed: int,
) -> KernelPCA:
    gamma = max(1e-3, 1.0 / max(1, x.shape[1]))
    if method_name == "KernelPCA_RBF":
        return KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=gamma,
            fit_inverse_transform=True,
            alpha=1e-2,
            eigen_solver="randomized",
            random_state=seed,
        )

    return KernelPCA(
        n_components=n_components,
        kernel="poly",
        gamma=gamma,
        degree=2,
        coef0=1.0,
        fit_inverse_transform=True,
        alpha=1e-2,
        eigen_solver="randomized",
        random_state=seed,
    )


def safe_fit_transform(
    method_spec: MethodSpec,
    x_scaled: np.ndarray,
    seed: int,
    profile: ProfileConfig,
) -> Tuple[Any, np.ndarray, float, Optional[str]]:
    estimator = method_spec.factory()
    start = time.perf_counter()
    fallback_note = None

    try:
        z = estimator.fit_transform(x_scaled)
    except Exception as first_exc:
        if method_spec.name.startswith("KernelPCA_"):
            n_components = _resolve_n_components(x_scaled, profile)
            fallback_estimator = build_kpca_fallback(
                method_name=method_spec.name,
                x=x_scaled,
                n_components=n_components,
                seed=seed,
            )
            try:
                z = fallback_estimator.fit_transform(x_scaled)
                estimator = fallback_estimator
                fallback_note = f"fallback_used_after: {first_exc}"
            except Exception as second_exc:
                raise RuntimeError(
                    f"{method_spec.name} failed primary and fallback paths: {second_exc}"
                ) from second_exc
        else:
            raise RuntimeError(f"{method_spec.name} failed: {first_exc}") from first_exc

    elapsed = time.perf_counter() - start
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != x_scaled.shape[0]:
        raise RuntimeError(f"{method_spec.name} produced invalid embedding shape: {z.shape}")
    return estimator, z, float(elapsed), fallback_note


def _safe_n_clusters(y: Optional[np.ndarray], n_samples: int) -> int:
    if n_samples < 3:
        return 2
    if y is not None and y.size > 0:
        n_unique = int(np.unique(y).size)
        return max(2, min(10, n_unique, n_samples - 1))
    heuristic = int(math.sqrt(max(4, n_samples // 2)))
    return max(2, min(10, heuristic, n_samples - 1))


def _add_metric(
    records: List[MetricRecord],
    dataset: str,
    method: str,
    metric_name: str,
    metric_value: float,
) -> None:
    records.append(
        MetricRecord(
            dataset=dataset,
            method=method,
            metric_name=metric_name,
            metric_value=float(metric_value),
            higher_is_better=METRIC_DIRECTIONS[metric_name],
        )
    )


def evaluate_method_metrics(
    dataset_name: str,
    method_name: str,
    estimator: Any,
    x_scaled: np.ndarray,
    z: np.ndarray,
    y: Optional[np.ndarray],
    supports_inverse: bool,
    seed: int,
    profile: ProfileConfig,
    records: List[MetricRecord],
    warnings_out: List[str],
) -> None:
    n_samples = x_scaled.shape[0]
    n_neighbors = max(2, min(profile.trust_neighbors, n_samples - 1))
    if n_samples > 10 and z.shape[0] == x_scaled.shape[0]:
        try:
            tw = trustworthiness(x_scaled, z, n_neighbors=n_neighbors)
            _add_metric(records, dataset_name, method_name, "trustworthiness", tw)
        except Exception as exc:
            warnings_out.append(f"{dataset_name}/{method_name}: trustworthiness failed ({exc})")

    n_clusters = _safe_n_clusters(y, n_samples)
    if n_samples > n_clusters:
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
            cluster_labels = km.fit_predict(z)
            sil = silhouette_score(z, cluster_labels)
            _add_metric(records, dataset_name, method_name, "silhouette", sil)

            if y is not None and np.unique(y).size > 1:
                ari = adjusted_rand_score(y, cluster_labels)
                _add_metric(records, dataset_name, method_name, "ari", ari)
        except Exception as exc:
            warnings_out.append(f"{dataset_name}/{method_name}: clustering metrics failed ({exc})")

    if y is not None and np.unique(y).size > 1:
        y_int = y.astype(np.int64, copy=False)
        counts = np.bincount(y_int)
        counts = counts[counts > 0]
        min_count = int(np.min(counts)) if counts.size else 0
        splits = min(profile.cv_splits, min_count)
        if splits >= 2:
            try:
                cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
                clf = LogisticRegression(max_iter=2000, solver="lbfgs")
                acc = float(np.mean(cross_val_score(clf, z, y_int, cv=cv, scoring="accuracy")))
                _add_metric(records, dataset_name, method_name, "cv_accuracy", acc)
            except Exception as exc:
                warnings_out.append(f"{dataset_name}/{method_name}: cv_accuracy failed ({exc})")

    if supports_inverse and hasattr(estimator, "inverse_transform"):
        try:
            x_recon = np.asarray(estimator.inverse_transform(z), dtype=np.float32)
            if x_recon.shape == x_scaled.shape:
                mse = float(np.mean((x_scaled - x_recon) ** 2))
                _add_metric(records, dataset_name, method_name, "reconstruction_mse", mse)
        except Exception as exc:
            warnings_out.append(f"{dataset_name}/{method_name}: inverse/reconstruction failed ({exc})")


def compute_composite_scores(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    norm_rows: List[Dict[str, Any]] = []
    for dataset_name in sorted(metrics_df["dataset"].unique()):
        ds = metrics_df[(metrics_df["dataset"] == dataset_name) & (metrics_df["metric_name"].isin(CORE_METRICS))]
        if ds.empty:
            continue

        for metric_name in CORE_METRICS:
            ms = ds[ds["metric_name"] == metric_name]
            if ms.empty:
                continue
            vals = ms["metric_value"].to_numpy(dtype=float)
            methods = ms["method"].to_numpy()
            higher_is_better = bool(ms["higher_is_better"].iloc[0])
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            if math.isclose(vmin, vmax):
                norm = np.ones_like(vals, dtype=float)
            else:
                if higher_is_better:
                    norm = (vals - vmin) / (vmax - vmin)
                else:
                    norm = (vmax - vals) / (vmax - vmin)

            for method_name, raw_value, norm_value in zip(methods, vals, norm):
                norm_rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "metric_name": metric_name,
                        "raw_value": float(raw_value),
                        "normalized_value": float(norm_value),
                    }
                )

    normalized_df = pd.DataFrame(norm_rows)
    if normalized_df.empty:
        return normalized_df, pd.DataFrame()

    dataset_scores = (
        normalized_df.groupby(["dataset", "method"], as_index=False)["normalized_value"]
        .mean()
        .rename(columns={"normalized_value": "dataset_composite"})
    )
    global_scores = (
        dataset_scores.groupby("method", as_index=False)["dataset_composite"]
        .mean()
        .rename(columns={"dataset_composite": "global_composite"})
        .sort_values("global_composite", ascending=False)
        .reset_index(drop=True)
    )
    global_scores["rank"] = np.arange(1, len(global_scores) + 1)
    return normalized_df, global_scores


def print_table(title: str, frame: pd.DataFrame, index: Optional[str] = None) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if frame.empty:
        print("(empty)")
        return
    if index is not None and index in frame.columns:
        disp = frame.set_index(index)
    else:
        disp = frame
    print(disp.round(4).to_string())


def compute_scree_curve(z: np.ndarray, max_components: int = 64) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        return np.empty(0, dtype=np.float32)

    z_centered = z - np.mean(z, axis=0, keepdims=True)
    rank_cap = min(z_centered.shape[0] - 1, z_centered.shape[1], max_components)
    if rank_cap <= 0:
        return np.empty(0, dtype=np.float32)

    singular_values = np.linalg.svd(z_centered, full_matrices=False, compute_uv=False)
    variances = (singular_values**2) / float(max(1, z_centered.shape[0] - 1))
    variances = variances[:rank_cap]
    total = float(np.sum(variances))
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros(rank_cap, dtype=np.float32)

    explained_ratio = variances / total
    return np.cumsum(explained_ratio, dtype=np.float64).astype(np.float32)


def save_artifacts(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    global_scores: pd.DataFrame,
    dataset_scores: pd.DataFrame,
    scree_curves: Dict[str, Dict[str, np.ndarray]],
    run_config: Dict[str, Any],
    no_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics_long.csv", index=False)

    summary_rows = []
    for _, row in dataset_scores.iterrows():
        summary_rows.append(
            {
                "scope": "dataset",
                "dataset": row["dataset"],
                "method": row["method"],
                "score_name": "dataset_composite",
                "score_value": row["dataset_composite"],
            }
        )
    for _, row in global_scores.iterrows():
        summary_rows.append(
            {
                "scope": "global",
                "dataset": "__global__",
                "method": row["method"],
                "score_name": "global_composite",
                "score_value": row["global_composite"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "scores_summary.csv", index=False)

    scree_rows: List[Dict[str, Any]] = []
    for dataset_name, methods in sorted(scree_curves.items()):
        for method_name, curve in sorted(methods.items()):
            for idx, value in enumerate(curve, start=1):
                scree_rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method_name,
                        "component": idx,
                        "cumulative_explained_variance": float(value),
                    }
                )
    pd.DataFrame(scree_rows).to_csv(output_dir / "scree_long.csv", index=False)

    metric_heatmap = (
        normalized_df.groupby(["metric_name", "method"], as_index=False)["normalized_value"]
        .mean()
        .pivot(index="metric_name", columns="method", values="normalized_value")
        if not normalized_df.empty
        else pd.DataFrame()
    )
    dataset_panel = (
        dataset_scores.pivot(index="dataset", columns="method", values="dataset_composite")
        if not dataset_scores.empty
        else pd.DataFrame()
    )

    if not no_plots:
        if not global_scores.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(global_scores["method"], global_scores["global_composite"], color="#2a9d8f")
            ax.set_ylabel("Global Composite Score")
            ax.set_title("Method Effectiveness Ranking")
            ax.set_ylim(0.0, 1.05)
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            fig.savefig(output_dir / "composite_bar.png", dpi=160)
            plt.close(fig)

        if not metric_heatmap.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            matrix = metric_heatmap.to_numpy(dtype=float)
            im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks(np.arange(metric_heatmap.shape[1]))
            ax.set_xticklabels(metric_heatmap.columns, rotation=20)
            ax.set_yticks(np.arange(metric_heatmap.shape[0]))
            ax.set_yticklabels(metric_heatmap.index)
            ax.set_title("Mean Normalized Core Metrics")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / "metric_heatmap.png", dpi=160)
            plt.close(fig)

        if not dataset_panel.empty:
            n_datasets = dataset_panel.shape[0]
            n_cols = 2
            n_rows = int(math.ceil(n_datasets / n_cols))
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(12, max(4, 3 * n_rows)),
                squeeze=False,
            )
            method_labels = list(dataset_panel.columns)
            for idx, dataset_name in enumerate(dataset_panel.index):
                r = idx // n_cols
                c = idx % n_cols
                ax = axes[r][c]
                vals = dataset_panel.loc[dataset_name].to_numpy(dtype=float)
                ax.bar(method_labels, vals, color="#264653")
                ax.set_title(dataset_name)
                ax.set_ylim(0.0, 1.05)
                ax.tick_params(axis="x", rotation=20)
            for idx in range(n_datasets, n_rows * n_cols):
                r = idx // n_cols
                c = idx % n_cols
                axes[r][c].axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / "dataset_panels.png", dpi=160)
            plt.close(fig)

        for dataset_name, methods in sorted(scree_curves.items()):
            if not methods:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            for method_name, curve in sorted(methods.items()):
                x_vals = np.arange(1, curve.shape[0] + 1)
                ax.plot(x_vals, curve, marker="o", markersize=3, linewidth=2, label=method_name)
            ax.axhline(y=0.95, color="r", linestyle=":", linewidth=1.2, label="95% threshold")
            ax.set_title(f"Scree Plot: {dataset_name}")
            ax.set_xlabel("Components")
            ax.set_ylabel("Cumulative Explained Variance")
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            safe_name = dataset_name.replace(" ", "_").replace("/", "_")
            fig.savefig(output_dir / f"scree_{safe_name}.png", dpi=160)
            plt.close(fig)

    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    profile = get_profile_config(args.mode)
    output_dir = Path(args.output_dir)

    dataset_specs = build_dataset_registry(args.mode, args.include_weather)
    all_records: List[MetricRecord] = []
    scree_curves: Dict[str, Dict[str, np.ndarray]] = {}
    failures: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    run_warnings: List[str] = []

    for ds_index, spec in enumerate(dataset_specs):
        ds_seed = args.seed + (ds_index + 1) * 1009
        ds_rng = np.random.default_rng(ds_seed)
        print(f"\n=== Dataset: {spec.name} ===")

        try:
            x, y, meta = spec.loader(profile, ds_rng, ds_seed)
        except Exception as exc:
            msg = f"Dataset {spec.name} skipped: {exc}"
            print(msg)
            skipped.append({"dataset": spec.name, "reason": str(exc)})
            continue

        cap = spec.sample_cap_quick if args.mode == "quick" else spec.sample_cap_full
        x, y = subsample_dataset(x, y, cap=cap, rng=ds_rng)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 20:
            msg = f"Dataset {spec.name} skipped: insufficient 2D samples ({x.shape})"
            print(msg)
            skipped.append({"dataset": spec.name, "reason": msg})
            continue

        if y is not None:
            y = np.asarray(y).reshape(-1)
            if y.shape[0] != x.shape[0]:
                msg = f"Dataset {spec.name}: label length mismatch {y.shape[0]} vs {x.shape[0]}"
                print(msg)
                skipped.append({"dataset": spec.name, "reason": msg})
                continue

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x).astype(np.float32)

        methods = build_method_specs(x_scaled, profile, seed=ds_seed)
        dataset_transforms: Dict[str, np.ndarray] = {}

        for method_idx, method_spec in enumerate(methods):
            method_seed = ds_seed + method_idx * 97
            print(f"  -> Running {method_spec.name}")
            try:
                estimator, z, elapsed, fallback_note = safe_fit_transform(
                    method_spec=method_spec,
                    x_scaled=x_scaled,
                    seed=method_seed,
                    profile=profile,
                )
                dataset_transforms[method_spec.name] = z
                curve = compute_scree_curve(z, max_components=64)
                if curve.size > 0:
                    scree_curves.setdefault(spec.name, {})[method_spec.name] = curve
                _add_metric(all_records, spec.name, method_spec.name, "fit_transform_seconds", elapsed)
                evaluate_method_metrics(
                    dataset_name=spec.name,
                    method_name=method_spec.name,
                    estimator=estimator,
                    x_scaled=x_scaled,
                    z=z,
                    y=y,
                    supports_inverse=method_spec.supports_inverse,
                    seed=method_seed,
                    profile=profile,
                    records=all_records,
                    warnings_out=run_warnings,
                )
                if fallback_note:
                    run_warnings.append(f"{spec.name}/{method_spec.name}: {fallback_note}")
            except Exception as exc:
                failure = {"dataset": spec.name, "method": method_spec.name, "reason": str(exc)}
                failures.append(failure)
                print(f"     FAILED: {exc}")

        if args.save_transforms and dataset_transforms:
            save_path = output_dir / f"transforms_{spec.name}.npz"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_path, **dataset_transforms)

        if meta:
            print(f"  metadata: {meta}")
        print(f"  completed methods: {len(dataset_transforms)} / {len(methods)}")

    metrics_df = pd.DataFrame([asdict(r) for r in all_records])
    if metrics_df.empty:
        print("\nNo metrics produced. Check dataset or method failures.")
        return

    normalized_df, global_scores = compute_composite_scores(metrics_df)
    dataset_scores = (
        normalized_df.groupby(["dataset", "method"], as_index=False)["normalized_value"]
        .mean()
        .rename(columns={"normalized_value": "dataset_composite"})
        if not normalized_df.empty
        else pd.DataFrame(columns=["dataset", "method", "dataset_composite"])
    )

    for dataset_name in sorted(metrics_df["dataset"].unique()):
        ds = metrics_df[metrics_df["dataset"] == dataset_name].copy()
        pivot = ds.pivot_table(index="metric_name", columns="method", values="metric_value", aggfunc="mean")
        print_table(f"Metrics: {dataset_name}", pivot.reset_index(), index="metric_name")

        ds_comp = dataset_scores[dataset_scores["dataset"] == dataset_name].sort_values(
            "dataset_composite", ascending=False
        )
        print_table(
            f"Composite (dataset): {dataset_name}",
            ds_comp[["method", "dataset_composite"]] if not ds_comp.empty else ds_comp,
        )

    print_table("Global Composite Ranking", global_scores[["rank", "method", "global_composite"]])

    run_config = {
        "mode": args.mode,
        "seed": int(args.seed),
        "include_weather": bool(args.include_weather),
        "output_dir": str(output_dir),
        "no_plots": bool(args.no_plots),
        "save_transforms": bool(args.save_transforms),
        "profile": asdict(profile),
        "datasets_requested": [spec.name for spec in dataset_specs],
        "datasets_observed": sorted(metrics_df["dataset"].unique().tolist()),
        "scree_datasets": sorted(scree_curves.keys()),
        "failures": failures,
        "skipped": skipped,
        "warnings": run_warnings,
    }

    save_artifacts(
        output_dir=output_dir,
        metrics_df=metrics_df,
        normalized_df=normalized_df,
        global_scores=global_scores,
        dataset_scores=dataset_scores,
        scree_curves=scree_curves,
        run_config=run_config,
        no_plots=args.no_plots,
    )

    print("\nArtifacts written:")
    print(f"  - {output_dir / 'metrics_long.csv'}")
    print(f"  - {output_dir / 'scores_summary.csv'}")
    print(f"  - {output_dir / 'scree_long.csv'}")
    print(f"  - {output_dir / 'run_config.json'}")
    if not args.no_plots:
        print(f"  - {output_dir / 'composite_bar.png'}")
        print(f"  - {output_dir / 'metric_heatmap.png'}")
        print(f"  - {output_dir / 'dataset_panels.png'}")
        print(f"  - {output_dir / 'scree_<dataset>.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive effectiveness benchmark for OBVAEKernel and PCA baselines."
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--include-weather", action="store_true")
    parser.add_argument("--output-dir", default="benchmark_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-transforms", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmark(cli_args)
