"""Portable NumPy-only OBVAE kernel module.

This module implements an overcomplete beta-VAE encoder that can be used as a
feature map for kernel methods. Training and inference are fully NumPy-based,
with optional serialization for cross-runtime portability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


class _NumPyVAEEngine:
    """Internal pure NumPy VAE engine with manual backpropagation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        rng: np.random.Generator,
        dtype: np.dtype,
        logvar_clip: Tuple[float, float] = (-30.0, 20.0),
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.rng = rng
        self.dtype = np.dtype(dtype)
        self.logvar_clip = logvar_clip

        self.W_enc = self._init_weights(self.input_dim, self.hidden_dim)
        self.b_enc = np.zeros(self.hidden_dim, dtype=self.dtype)

        self.W_mu = self._init_weights(self.hidden_dim, self.latent_dim)
        self.b_mu = np.zeros(self.latent_dim, dtype=self.dtype)

        self.W_logvar = self._init_weights(self.hidden_dim, self.latent_dim)
        self.b_logvar = np.zeros(self.latent_dim, dtype=self.dtype)

        self.W_dec1 = self._init_weights(self.latent_dim, self.hidden_dim)
        self.b_dec1 = np.zeros(self.hidden_dim, dtype=self.dtype)

        self.W_dec2 = self._init_weights(self.hidden_dim, self.input_dim)
        self.b_dec2 = np.zeros(self.input_dim, dtype=self.dtype)

        self._cache: Dict[str, np.ndarray] = {}

    def _init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / float(fan_in + fan_out))
        return self.rng.uniform(
            low=-limit, high=limit, size=(fan_out, fan_in)
        ).astype(self.dtype, copy=False)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(x.dtype, copy=False)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_enc = x @ self.W_enc.T + self.b_enc
        h_enc = self._relu(z_enc)

        mu = h_enc @ self.W_mu.T + self.b_mu
        logvar = h_enc @ self.W_logvar.T + self.b_logvar
        logvar = np.clip(logvar, self.logvar_clip[0], self.logvar_clip[1])
        return z_enc, h_enc, mu, logvar

    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_dec1 = z @ self.W_dec1.T + self.b_dec1
        h_dec = self._relu(z_dec1)
        x_recon = h_dec @ self.W_dec2.T + self.b_dec2
        return z_dec1, h_dec, x_recon

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_enc, h_enc, mu, logvar = self.encode(x)
        std = np.exp(0.5 * logvar).astype(self.dtype, copy=False)
        eps = self.rng.standard_normal(mu.shape).astype(self.dtype, copy=False)
        z = mu + std * eps
        z_dec1, h_dec, x_recon = self.decode(z)

        self._cache = {
            "x": x,
            "z_enc": z_enc,
            "h_enc": h_enc,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "eps": eps,
            "z": z,
            "z_dec1": z_dec1,
            "h_dec": h_dec,
            "x_recon": x_recon,
        }
        return x_recon, mu, logvar

    def encode_mean(self, x: np.ndarray) -> np.ndarray:
        _, h_enc, mu, _ = self.encode(x)
        # h_enc is computed intentionally to match transform path numerics.
        _ = h_enc
        return mu

    @staticmethod
    def compute_loss(
        x_recon: np.ndarray,
        x: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
        beta: float,
    ) -> Tuple[float, float, float]:
        mse_loss = float(np.sum((x_recon - x) ** 2))
        kld_loss = float(-0.5 * np.sum(1.0 + logvar - mu**2 - np.exp(logvar)))
        loss = mse_loss + float(beta) * kld_loss
        return loss, mse_loss, kld_loss

    def backward(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
        beta: float,
    ) -> Dict[str, np.ndarray]:
        del x, x_recon, mu, logvar
        cache = self._cache

        x_batch = cache["x"]
        z_enc = cache["z_enc"]
        h_enc = cache["h_enc"]
        mu_cached = cache["mu"]
        logvar_cached = cache["logvar"]
        std = cache["std"]
        eps = cache["eps"]
        z = cache["z"]
        z_dec1 = cache["z_dec1"]
        h_dec = cache["h_dec"]
        x_recon_cached = cache["x_recon"]

        dx_recon = 2.0 * (x_recon_cached - x_batch)
        dW_dec2 = dx_recon.T @ h_dec
        db_dec2 = np.sum(dx_recon, axis=0)

        dh_dec = dx_recon @ self.W_dec2
        dz_dec1 = dh_dec * self._relu_derivative(z_dec1)
        dW_dec1 = dz_dec1.T @ z
        db_dec1 = np.sum(dz_dec1, axis=0)

        dz = dz_dec1 @ self.W_dec1
        dmu_kl = mu_cached
        dlogvar_kl = -0.5 * (1.0 - np.exp(logvar_cached))

        dmu = dz + float(beta) * dmu_kl
        dlogvar = dz * (0.5 * std * eps) + float(beta) * dlogvar_kl

        dW_mu = dmu.T @ h_enc
        db_mu = np.sum(dmu, axis=0)

        dW_logvar = dlogvar.T @ h_enc
        db_logvar = np.sum(dlogvar, axis=0)

        dh_enc = dmu @ self.W_mu + dlogvar @ self.W_logvar
        dz_enc = dh_enc * self._relu_derivative(z_enc)
        dW_enc = dz_enc.T @ x_batch
        db_enc = np.sum(dz_enc, axis=0)

        return {
            "W_enc": dW_enc.astype(self.dtype, copy=False),
            "b_enc": db_enc.astype(self.dtype, copy=False),
            "W_mu": dW_mu.astype(self.dtype, copy=False),
            "b_mu": db_mu.astype(self.dtype, copy=False),
            "W_logvar": dW_logvar.astype(self.dtype, copy=False),
            "b_logvar": db_logvar.astype(self.dtype, copy=False),
            "W_dec1": dW_dec1.astype(self.dtype, copy=False),
            "b_dec1": db_dec1.astype(self.dtype, copy=False),
            "W_dec2": dW_dec2.astype(self.dtype, copy=False),
            "b_dec2": db_dec2.astype(self.dtype, copy=False),
        }


class AdamOptimizer:
    """Pure NumPy Adam optimizer."""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, param in self.params.items():
            grad = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad**2)

            m_hat = self.m[name] / (1.0 - self.beta1**self.t)
            v_hat = self.v[name] / (1.0 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class OBVAEKernel:
    """Overcomplete Beta-VAE kernel transformer with NumPy-only core."""

    SERIAL_VERSION = 1

    def __init__(
        self,
        latent_dim: Union[str, int] = "auto",
        expansion_factor: int = 10,
        hidden_dim: int = 64,
        beta: float = 2.0,
        epochs: int = 300,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
        variance_threshold: float = 1e-4,
        dtype: Union[str, np.dtype, type] = np.float32,
        keep_decoder: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.variance_threshold = variance_threshold
        self.dtype = np.dtype(dtype)
        self.keep_decoder = keep_decoder
        self.verbose = verbose
        self.random_state = random_state

        self.W_h: Optional[np.ndarray] = None
        self.b_h: Optional[np.ndarray] = None
        self.W_mu: Optional[np.ndarray] = None
        self.b_mu: Optional[np.ndarray] = None

        self._full_W_mu: Optional[np.ndarray] = None
        self._full_b_mu: Optional[np.ndarray] = None

        self._W_dec1: Optional[np.ndarray] = None
        self._b_dec1: Optional[np.ndarray] = None
        self._W_dec2: Optional[np.ndarray] = None
        self._b_dec2: Optional[np.ndarray] = None

        self.n_features_in_: Optional[int] = None
        self.training_dim_: Optional[int] = None
        self.effective_dim_: Optional[int] = None
        self.active_indices_: Optional[np.ndarray] = None
        self.feature_variances_: Optional[np.ndarray] = None
        self.is_fitted_ = False

    def _validate_hyperparameters(self) -> None:
        if self.latent_dim != "auto":
            if not isinstance(self.latent_dim, (int, np.integer)):
                raise ValueError("latent_dim must be 'auto' or a positive integer.")
            if int(self.latent_dim) <= 0:
                raise ValueError("latent_dim must be a positive integer.")

        if not isinstance(self.expansion_factor, (int, np.integer)) or int(self.expansion_factor) <= 0:
            raise ValueError("expansion_factor must be a positive integer.")
        if not isinstance(self.hidden_dim, (int, np.integer)) or int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be a positive integer.")
        if not isinstance(self.epochs, (int, np.integer)) or int(self.epochs) <= 0:
            raise ValueError("epochs must be a positive integer.")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be greater than 0.")
        if float(self.variance_threshold) < 0.0:
            raise ValueError("variance_threshold must be >= 0.")
        if self.batch_size is not None and int(self.batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

    def _to_2d_array(self, x: ArrayLike, expected_features: Optional[int] = None) -> np.ndarray:
        arr = np.asarray(x, dtype=self.dtype)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {arr.ndim}D.")
        if expected_features is not None and arr.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {arr.shape[1]}."
            )
        return np.ascontiguousarray(arr, dtype=self.dtype)

    def _to_single_sample(self, x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=self.dtype)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2 and arr.shape[0] == 1:
            pass
        else:
            raise ValueError("Input must be a single sample as shape (n_features,) or (1, n_features).")
        return np.ascontiguousarray(arr, dtype=self.dtype)

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling this method.")

    def _resolve_training_dim(self, input_dim: int) -> int:
        if self.latent_dim == "auto":
            return int(min(input_dim * int(self.expansion_factor), 500))
        return int(self.latent_dim)

    def _normalize_rows(self, phi: np.ndarray) -> np.ndarray:
        eps = np.finfo(phi.dtype).eps
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        return phi / np.maximum(norms, eps)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "OBVAEKernel":
        del y
        self._validate_hyperparameters()
        X_arr = self._to_2d_array(X)
        n_samples, input_dim = X_arr.shape

        training_dim = self._resolve_training_dim(input_dim)
        if training_dim <= 0:
            raise ValueError("Resolved training latent dimension must be positive.")

        rng = np.random.default_rng(self.random_state)
        engine = _NumPyVAEEngine(
            input_dim=input_dim,
            hidden_dim=int(self.hidden_dim),
            latent_dim=training_dim,
            rng=rng,
            dtype=self.dtype,
            logvar_clip=(-30.0, 20.0),
        )

        params = {
            "W_enc": engine.W_enc,
            "b_enc": engine.b_enc,
            "W_mu": engine.W_mu,
            "b_mu": engine.b_mu,
            "W_logvar": engine.W_logvar,
            "b_logvar": engine.b_logvar,
            "W_dec1": engine.W_dec1,
            "b_dec1": engine.b_dec1,
            "W_dec2": engine.W_dec2,
            "b_dec2": engine.b_dec2,
        }
        optimizer = AdamOptimizer(params=params, lr=float(self.lr))

        if self.batch_size is None:
            batch_size = n_samples
        else:
            batch_size = min(int(self.batch_size), n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(int(self.epochs)):
            indices = rng.permutation(n_samples)
            X_shuffled = X_arr[indices]
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start:end]

                x_recon, mu, logvar = engine.forward(X_batch)
                loss, _, _ = engine.compute_loss(x_recon, X_batch, mu, logvar, beta=float(self.beta))
                grads = engine.backward(X_batch, x_recon, mu, logvar, beta=float(self.beta))
                optimizer.step(grads)
                epoch_loss += loss

            if self.verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == int(self.epochs)):
                print(f"Epoch [{epoch + 1}/{int(self.epochs)}], Loss: {epoch_loss / n_samples:.6f}")

        final_mu = engine.encode_mean(X_arr)
        feature_variances = np.var(final_mu, axis=0).astype(self.dtype, copy=False)
        active_indices = np.flatnonzero(feature_variances > float(self.variance_threshold)).astype(np.int64)
        if active_indices.size == 0:
            active_indices = np.array([int(np.argmax(feature_variances))], dtype=np.int64)

        self.W_h = engine.W_enc.copy()
        self.b_h = engine.b_enc.copy()
        self.W_mu = engine.W_mu[active_indices, :].copy()
        self.b_mu = engine.b_mu[active_indices].copy()
        self._full_W_mu = engine.W_mu.copy()
        self._full_b_mu = engine.b_mu.copy()

        if self.keep_decoder:
            self._W_dec1 = engine.W_dec1.copy()
            self._b_dec1 = engine.b_dec1.copy()
            self._W_dec2 = engine.W_dec2.copy()
            self._b_dec2 = engine.b_dec2.copy()
        else:
            self._W_dec1 = None
            self._b_dec1 = None
            self._W_dec2 = None
            self._b_dec2 = None

        self.n_features_in_ = input_dim
        self.training_dim_ = training_dim
        self.active_indices_ = active_indices
        self.feature_variances_ = feature_variances.copy()
        self.effective_dim_ = int(active_indices.size)
        self.is_fitted_ = True

        if self.verbose:
            print(
                f"OBVAEKernel: Inflated to {training_dim}D, "
                f"pruned to {self.effective_dim_} active features."
            )

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        X_arr = self._to_2d_array(X, expected_features=int(self.n_features_in_))
        z_hidden = X_arr @ self.W_h.T + self.b_h
        h = np.maximum(z_hidden, 0.0)
        mu = h @ self.W_mu.T + self.b_mu
        if not np.isfinite(mu).all():
            raise ValueError("Non-finite values encountered in transform output.")
        return mu.astype(self.dtype, copy=False)

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        self._require_fitted()
        if self._W_dec1 is None or self._b_dec1 is None or self._W_dec2 is None or self._b_dec2 is None:
            raise RuntimeError("Decoder weights are unavailable (keep_decoder=False during fit).")

        Z_arr = self._to_2d_array(Z)
        if Z_arr.shape[1] == int(self.effective_dim_):
            z_full = np.zeros((Z_arr.shape[0], int(self.training_dim_)), dtype=self.dtype)
            z_full[:, self.active_indices_] = Z_arr
        elif Z_arr.shape[1] == int(self.training_dim_):
            z_full = Z_arr
        else:
            raise ValueError(
                f"Expected latent width {self.effective_dim_} (pruned) "
                f"or {self.training_dim_} (full), got {Z_arr.shape[1]}."
            )

        z_dec1 = z_full @ self._W_dec1.T + self._b_dec1
        h_dec = np.maximum(z_dec1, 0.0)
        x_recon = h_dec @ self._W_dec2.T + self._b_dec2
        if not np.isfinite(x_recon).all():
            raise ValueError("Non-finite values encountered in inverse_transform output.")
        return x_recon.astype(self.dtype, copy=False)

    def kernel_matrix(
        self,
        X: ArrayLike,
        Y: Optional[ArrayLike] = None,
        normalize: bool = True,
        center: bool = False,
    ) -> np.ndarray:
        self._require_fitted()
        phi_x = self.transform(X)
        phi_y = phi_x if Y is None else self.transform(Y)

        if normalize:
            phi_x = self._normalize_rows(phi_x)
            phi_y = self._normalize_rows(phi_y)

        K = phi_x @ phi_y.T

        if center:
            row_means = np.mean(K, axis=1, keepdims=True)
            col_means = np.mean(K, axis=0, keepdims=True)
            global_mean = np.mean(K)
            K = K - row_means - col_means + global_mean

        if Y is None:
            K = 0.5 * (K + K.T)

        if not np.isfinite(K).all():
            raise ValueError("Non-finite values encountered in kernel matrix.")

        return K.astype(self.dtype, copy=False)

    def pairwise_scalar(self, x: ArrayLike, y: ArrayLike, normalize: bool = True) -> float:
        self._require_fitted()
        x_arr = self._to_single_sample(x)
        y_arr = self._to_single_sample(y)
        phi_x = self.transform(x_arr)[0]
        phi_y = self.transform(y_arr)[0]

        if normalize:
            eps = np.finfo(phi_x.dtype).eps
            phi_x = phi_x / max(float(np.linalg.norm(phi_x)), float(eps))
            phi_y = phi_y / max(float(np.linalg.norm(phi_y)), float(eps))

        value = float(np.dot(phi_x, phi_y))
        if not np.isfinite(value):
            raise ValueError("Non-finite value encountered in pairwise kernel output.")
        return value

    def as_pairwise_callable(self, normalize: bool = True) -> Callable[[ArrayLike, ArrayLike], float]:
        def _kernel(a: ArrayLike, b: ArrayLike) -> float:
            return self.pairwise_scalar(a, b, normalize=normalize)

        return _kernel

    def get_feature_variances(self) -> np.ndarray:
        self._require_fitted()
        return self.feature_variances_.copy()

    def _config_dict(self) -> Dict[str, object]:
        return {
            "latent_dim": self.latent_dim,
            "expansion_factor": int(self.expansion_factor),
            "hidden_dim": int(self.hidden_dim),
            "beta": float(self.beta),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "batch_size": None if self.batch_size is None else int(self.batch_size),
            "variance_threshold": float(self.variance_threshold),
            "dtype": self.dtype.name,
            "keep_decoder": bool(self.keep_decoder),
            "verbose": bool(self.verbose),
            "random_state": self.random_state,
        }

    def save(self, path_prefix: str) -> None:
        self._require_fitted()
        prefix = Path(path_prefix)
        npz_path = Path(f"{prefix}.npz")
        json_path = Path(f"{prefix}.json")

        arrays: Dict[str, np.ndarray] = {
            "W_h": self.W_h,
            "b_h": self.b_h,
            "W_mu": self.W_mu,
            "b_mu": self.b_mu,
            "full_W_mu": self._full_W_mu,
            "full_b_mu": self._full_b_mu,
            "active_indices": self.active_indices_.astype(np.int64, copy=False),
            "feature_variances": self.feature_variances_,
        }

        has_decoder = (
            self._W_dec1 is not None
            and self._b_dec1 is not None
            and self._W_dec2 is not None
            and self._b_dec2 is not None
        )
        if has_decoder:
            arrays["W_dec1"] = self._W_dec1
            arrays["b_dec1"] = self._b_dec1
            arrays["W_dec2"] = self._W_dec2
            arrays["b_dec2"] = self._b_dec2

        np.savez(npz_path, **arrays)

        metadata = {
            "version": self.SERIAL_VERSION,
            "class_name": "OBVAEKernel",
            "config": self._config_dict(),
            "fitted": {
                "n_features_in_": int(self.n_features_in_),
                "training_dim_": int(self.training_dim_),
                "effective_dim_": int(self.effective_dim_),
                "has_decoder": bool(has_decoder),
            },
        }
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path_prefix: str) -> "OBVAEKernel":
        prefix = Path(path_prefix)
        npz_path = Path(f"{prefix}.npz")
        json_path = Path(f"{prefix}.json")

        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        config = metadata["config"]
        model = cls(**config)

        with np.load(npz_path, allow_pickle=False) as data:
            model.W_h = np.asarray(data["W_h"], dtype=model.dtype)
            model.b_h = np.asarray(data["b_h"], dtype=model.dtype)
            model.W_mu = np.asarray(data["W_mu"], dtype=model.dtype)
            model.b_mu = np.asarray(data["b_mu"], dtype=model.dtype)
            model._full_W_mu = np.asarray(data["full_W_mu"], dtype=model.dtype)
            model._full_b_mu = np.asarray(data["full_b_mu"], dtype=model.dtype)
            model.active_indices_ = np.asarray(data["active_indices"], dtype=np.int64)
            model.feature_variances_ = np.asarray(data["feature_variances"], dtype=model.dtype)

            has_decoder = bool(metadata["fitted"]["has_decoder"])
            if has_decoder:
                model._W_dec1 = np.asarray(data["W_dec1"], dtype=model.dtype)
                model._b_dec1 = np.asarray(data["b_dec1"], dtype=model.dtype)
                model._W_dec2 = np.asarray(data["W_dec2"], dtype=model.dtype)
                model._b_dec2 = np.asarray(data["b_dec2"], dtype=model.dtype)
            else:
                model._W_dec1 = None
                model._b_dec1 = None
                model._W_dec2 = None
                model._b_dec2 = None

        model.n_features_in_ = int(metadata["fitted"]["n_features_in_"])
        model.training_dim_ = int(metadata["fitted"]["training_dim_"])
        model.effective_dim_ = int(metadata["fitted"]["effective_dim_"])
        model.is_fitted_ = True
        return model


OBVAEK = OBVAEKernel

