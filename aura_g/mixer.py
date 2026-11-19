"""Routing & mixing matrix — combines all modules under latent control."""
import numpy as np


class RoutingMatrix:
    def __init__(self):
        self.weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    def blend(self, amp_out, fx_out, phys_out, length):
        if length < 0:
            raise ValueError("length must be non-negative")
        def _fit(x):
            x = np.asarray(x)
            if x.ndim != 1:
                raise ValueError("mixer inputs must be mono arrays")
            if not np.all(np.isfinite(x)):
                raise ValueError("mixer input contains NaN or infinity")
            if len(x) >= length:
                return x[:length]
            return np.pad(x, (0, length - len(x)))
        return (self.weights[0] * _fit(amp_out) +
                self.weights[1] * _fit(fx_out) +
                self.weights[2] * _fit(phys_out))

    def morph(self, latent_vector):
        latent_vector = np.asarray(latent_vector, dtype=np.float64)
        if latent_vector.shape != (32,) or not np.all(np.isfinite(latent_vector)):
            raise ValueError("latent vector must be a finite 32-dimensional vector")
        expr = latent_vector[16:24]
        weights = np.array([0.3 + 0.4 * expr[0],
                            0.1 + 0.5 * expr[1],
                            0.1 + 0.4 * expr[2]], dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        self.weights = weights / total if total > 1e-12 else np.array([0.5, 0.3, 0.2])
        return self.weights.copy()
