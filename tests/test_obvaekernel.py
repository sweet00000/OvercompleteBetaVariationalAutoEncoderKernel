import tempfile
import unittest
from pathlib import Path

import numpy as np

from obvaekernel import OBVAEK, OBVAEKernel


class TestOBVAEKernel(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1234)
        self.X = rng.normal(size=(96, 8)).astype(np.float32)

    def _build_model(self, epochs: int = 12) -> OBVAEKernel:
        return OBVAEKernel(
            latent_dim="auto",
            expansion_factor=3,
            hidden_dim=16,
            beta=2.0,
            epochs=epochs,
            lr=0.003,
            batch_size=24,
            random_state=7,
            verbose=False,
        )

    def test_fit_transform_shape_and_finite(self) -> None:
        model = self._build_model()
        z = model.fit_transform(self.X)
        self.assertEqual(z.shape[0], self.X.shape[0])
        self.assertEqual(z.shape[1], model.effective_dim_)
        self.assertGreaterEqual(model.effective_dim_, 1)
        self.assertLessEqual(model.effective_dim_, min(self.X.shape[1] * 3, 500))
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(model.is_fitted_)

    def test_pruning_attributes_valid(self) -> None:
        model = self._build_model()
        model.fit(self.X)
        self.assertEqual(model.feature_variances_.shape[0], model.training_dim_)
        self.assertEqual(model.active_indices_.ndim, 1)
        self.assertGreaterEqual(model.active_indices_.size, 1)
        self.assertLess(int(np.max(model.active_indices_)), model.training_dim_)

    def test_kernel_matrix_symmetry_and_psd(self) -> None:
        model = self._build_model()
        model.fit(self.X)
        k = model.kernel_matrix(self.X, normalize=True, center=False)
        self.assertTrue(np.isfinite(k).all())
        self.assertTrue(np.allclose(k, k.T, atol=1e-5))

        eigvals = np.linalg.eigvalsh(0.5 * (k + k.T))
        self.assertGreaterEqual(float(np.min(eigvals)), -1e-4)

    def test_pairwise_scalar_matches_matrix_entry(self) -> None:
        model = self._build_model()
        model.fit(self.X)
        i, j = 5, 17
        scalar = model.pairwise_scalar(self.X[i], self.X[j], normalize=True)
        matrix_val = model.kernel_matrix(
            self.X[i : i + 1], self.X[j : j + 1], normalize=True, center=False
        )[0, 0]
        self.assertAlmostEqual(float(scalar), float(matrix_val), places=5)

    def test_inverse_transform_shape(self) -> None:
        model = self._build_model()
        model.fit(self.X)
        z = model.transform(self.X[:10])
        x_recon = model.inverse_transform(z)
        self.assertEqual(x_recon.shape, (10, self.X.shape[1]))
        self.assertTrue(np.isfinite(x_recon).all())

    def test_save_load_roundtrip_transform(self) -> None:
        model = self._build_model()
        model.fit(self.X)
        z_before = model.transform(self.X[:20])

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "obvaek_model"
            model.save(str(prefix))
            loaded = OBVAEKernel.load(str(prefix))
            z_after = loaded.transform(self.X[:20])
            max_abs_diff = float(np.max(np.abs(z_before - z_after)))
            self.assertLessEqual(max_abs_diff, 1e-5)
            self.assertEqual(loaded.effective_dim_, model.effective_dim_)
            self.assertEqual(loaded.n_features_in_, model.n_features_in_)

    def test_alias_compatibility(self) -> None:
        self.assertIs(OBVAEK, OBVAEKernel)
        model = OBVAEK(latent_dim=4, hidden_dim=12, epochs=8, random_state=3, verbose=False)
        z = model.fit_transform(self.X)
        self.assertEqual(z.shape[0], self.X.shape[0])
        self.assertEqual(z.shape[1], model.effective_dim_)


if __name__ == "__main__":
    unittest.main()

