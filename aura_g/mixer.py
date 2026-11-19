"""Routing & mixing matrix — combines all modules under latent control."""
import numpy as np


class RoutingMatrix:
    def __init__(self):
        self.weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    def blend(self, amp_out, fx_out, phys_out, length):
        if not isinstance(length, (int, np.integer)) or length < 0:
            raise ValueError("length must be a non-negative integer")
        raw_weights = np.asarray(self.weights)
        if np.iscomplexobj(raw_weights):
            raise ValueError("mixer weights must be real-valued")
        weights = np.asarray(raw_weights, dtype=np.float64)
        if (weights.shape != (3,) or not np.all(np.isfinite(weights)) or
                np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8)):
            raise ValueError("mixer weights must be three finite non-negative values summing to one")
        def _fit(x):
            x = np.asarray(x)
            if np.iscomplexobj(x):
                raise ValueError("mixer inputs must be real-valued")
            if x.ndim != 1:
                raise ValueError("mixer inputs must be mono arrays")
            if not np.all(np.isfinite(x)):
                raise ValueError("mixer input contains NaN or infinity")
            if len(x) >= length:
                return x[:length]
            return np.pad(x, (0, length - len(x)))
        return (weights[0] * _fit(amp_out) +
                weights[1] * _fit(fx_out) +
                weights[2] * _fit(phys_out))

    def morph(self, latent_vector):
        raw = np.asarray(latent_vector)
        if np.iscomplexobj(raw):
            raise ValueError("latent vector must be real-valued")
        latent_vector = np.asarray(raw, dtype=np.float64)
        if latent_vector.shape != (32,) or not np.all(np.isfinite(latent_vector)):
            raise ValueError("latent vector must be a finite 32-dimensional vector")
        expr = np.clip(latent_vector[16:24], 0.0, 1.0)
        weights = np.array([0.3 + 0.4 * expr[0],
                            0.1 + 0.5 * expr[1],
                            0.1 + 0.4 * expr[2]], dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        self.weights = weights / total if total > 1e-12 else np.array([0.5, 0.3, 0.2])
        return self.weights.copy()
