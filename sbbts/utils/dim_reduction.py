"""
Dimensionality reduction for high-dimensional time series.

Implements the PCA + k-means approach from Appendix C.2.4:
- PCA to reduce d=433 dimensions to m factors (m=16 fixed or Marchenko-Pastur auto)
- k-means clustering to group factors into 3 clusters
- SBBTS fitted independently to each cluster
- Per-asset GMM for idiosyncratic residuals (heavy tails)
"""

from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Marchenko-Pastur threshold
# ---------------------------------------------------------------------------


def marchenko_pastur_n_components(X_flat: np.ndarray) -> int:
    """
    Determine number of signal PCA components via random matrix theory.

    Marchenko-Pastur upper edge:  λ+ = σ²(1 + √(d/n))²

    Any eigenvalue above λ+ is a signal component (not noise). This is
    data-adaptive and avoids both underfitting (fixed m too small) and
    overfitting (fixed m too large).

    Args:
        X_flat: Flattened data, shape (n_observations, d_features)

    Returns:
        Number of signal components (≥ 1)
    """
    n, d = X_flat.shape
    if n <= d:
        # Underdetermined: fall back to a conservative estimate
        return max(1, d // 4)

    sigma2 = np.var(X_flat)
    ratio = d / n
    lambda_plus = sigma2 * (1 + np.sqrt(ratio)) ** 2

    cov = np.cov(X_flat.T)
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending order

    n_signal = int(np.sum(eigenvalues > lambda_plus))
    return max(1, n_signal)


def to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


class PCAReducer:
    """
    PCA-based dimensionality reduction.

    From Appendix C.2.4: "we project the data onto a lower-dimensional
    factor space F ∈ R^{N×m} using PCA, with m << d."

    n_components can be an integer (fixed) or 'auto' to use the
    Marchenko-Pastur random-matrix-theory threshold.
    """

    def __init__(self, n_components: Union[int, str] = 16):
        """
        Args:
            n_components: Number of PCA components, or 'auto' for
                Marchenko-Pastur adaptive selection.
        """
        self.n_components = n_components
        self.pca = None
        self._fitted = False
        self._original_shape = None
        self._n_components_: Optional[int] = None  # resolved at fit time

    def fit(self, X: Union[np.ndarray, Tensor]) -> "PCAReducer":
        """
        Fit PCA on the data.

        Args:
            X: Data, shape (N, T, d) or (N*T, d)

        Returns:
            self
        """
        X = to_numpy(X)
        self._original_shape = X.shape

        X_flat = X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X

        if self.n_components == "auto":
            self._n_components_ = marchenko_pastur_n_components(X_flat)
        else:
            self._n_components_ = int(self.n_components)

        self.pca = PCA(n_components=self._n_components_)
        self.pca.fit(X_flat)
        self._fitted = True
        return self

    def transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Transform data to lower dimension.

        Args:
            X: Data, shape (N, T, d) or (N*T, d)

        Returns:
            Transformed data, same number of dims as input but with m components
        """
        if not self._fitted:
            raise RuntimeError("PCAReducer must be fitted before transform")

        X = to_numpy(X)
        original_shape = X.shape

        if X.ndim == 3:
            N, T, d = X.shape
            X_flat = X.reshape(-1, d)
            X_reduced = self.pca.transform(X_flat)
            return X_reduced.reshape(N, T, -1)
        else:
            return self.pca.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Inverse transform from reduced space.

        From Appendix C.2.4:
            X̂ = F̂ @ P_{1:m}^T + R̂

        Args:
            X: Reduced data, shape (N, T, m)

        Returns:
            Reconstructed data, shape (N, T, d)
        """
        if not self._fitted:
            raise RuntimeError("PCAReducer must be fitted before inverse_transform")

        X = to_numpy(X)

        if X.ndim == 3:
            N, T, m = X.shape
            X_flat = X.reshape(-1, m)
            X_reconstructed = self.pca.inverse_transform(X_flat)
            return X_reconstructed.reshape(N, T, -1)
        else:
            return self.pca.inverse_transform(X)

    def fit_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio per component."""
        if not self._fitted:
            return None
        return self.pca.explained_variance_ratio_


class FactorClusterer:
    """
    K-means clustering of PCA factors.

    From Appendix C.2.4: "The extracted independent factors are subsequently
    grouped into 3 clusters using k-means clustering, under the assumption
    that factors within the same cluster share the same distribution."
    """

    def __init__(self, n_clusters: int = 3):
        """
        Args:
            n_clusters: Number of clusters (Appendix C.2.4: 3)
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self._fitted = False

    def fit(self, factors: Union[np.ndarray, Tensor]) -> "FactorClusterer":
        """
        Fit k-means on factor loadings or time series.

        Args:
            factors: Factor data, shape (n_factors, ...) or feature matrix

        Returns:
            self
        """
        factors = to_numpy(factors)

        if factors.ndim > 2:
            factors = factors.reshape(factors.shape[0], -1)

        self.kmeans.fit(factors)
        self._fitted = True
        return self

    def predict(self, factors: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Assign factors to clusters.

        Args:
            factors: Factor data

        Returns:
            Cluster labels
        """
        if not self._fitted:
            raise RuntimeError("FactorClusterer must be fitted first")

        factors = to_numpy(factors)
        if factors.ndim > 2:
            factors = factors.reshape(factors.shape[0], -1)

        return self.kmeans.predict(factors)

    @property
    def labels(self) -> np.ndarray:
        """Get cluster labels from fitting."""
        if not self._fitted:
            return None
        return self.kmeans.labels_


class ResidualModeler:
    """
    Model residuals with Gaussian Mixture.

    From Appendix C.2.4: "The remaining idiosyncratic components are treated
    separately. Since these residuals exhibit heavy-tailed behavior, they are
    modeled independently across dimensions using a Gaussian mixture with
    two components."
    """

    def __init__(self, n_components: int = 2):
        """
        Args:
            n_components: Number of GMM components (Appendix C.2.4: 2)
        """
        self.n_components = n_components
        self.gmms = []
        self._fitted = False
        self._n_dims = None

    def fit(self, residuals: Union[np.ndarray, Tensor]) -> "ResidualModeler":
        """
        Fit GMM to each dimension independently.

        Args:
            residuals: Residual data, shape (N, d)

        Returns:
            self
        """
        residuals = to_numpy(residuals)

        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        self._n_dims = residuals.shape[-1]
        self.gmms = []

        for d in range(self._n_dims):
            gmm = GaussianMixture(n_components=self.n_components, random_state=42)
            gmm.fit(residuals[:, d : d + 1])
            self.gmms.append(gmm)

        self._fitted = True
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from the fitted GMMs.

        Args:
            n_samples: Number of samples

        Returns:
            Samples, shape (n_samples, n_dims)
        """
        if not self._fitted:
            raise RuntimeError("ResidualModeler must be fitted first")

        samples = np.zeros((n_samples, self._n_dims))
        for d, gmm in enumerate(self.gmms):
            samples[:, d] = gmm.sample(n_samples)[0].flatten()

        return samples


class PCAKMeansReducer:
    """
    Combined PCA + K-means dimensionality reduction.

    Full pipeline from Appendix C.2.4:
    1. PCA: R^d → R^m (m=16)
    2. K-means: group m factors into k clusters (k=3)
    3. Residual modeling with GMM

    Usage:
        reducer = PCAKMeansReducer(n_components=16, n_clusters=3)
        X_reduced = reducer.fit_transform(X_train)
        # Fit SBBTS on X_reduced
        # After generating X_synth_reduced:
        X_synth = reducer.inverse_transform(X_synth_reduced)
    """

    def __init__(
        self,
        n_components: Union[int, str] = 16,
        n_clusters: int = 3,
        model_residuals: bool = True,
    ):
        """
        Args:
            n_components: PCA components, or 'auto' for Marchenko-Pastur selection
            n_clusters: K-means clusters (k=3)
            model_residuals: Whether to model residuals with per-asset GMM
        """
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.model_residuals = model_residuals

        self.pca_reducer = PCAReducer(n_components)
        self.clusterer = FactorClusterer(n_clusters)
        self.residual_modeler = ResidualModeler() if model_residuals else None

        self._fitted = False
        self._residual_std = None

    def fit(self, X: Union[np.ndarray, Tensor]) -> "PCAKMeansReducer":
        """
        Fit the full reduction pipeline.

        Args:
            X: Input data, shape (N, T, d)

        Returns:
            self
        """
        X = to_numpy(X)

        self.pca_reducer.fit(X)
        X_reduced = self.pca_reducer.transform(X)

        factor_features = X_reduced.reshape(X_reduced.shape[0], -1).T
        n_components = self.pca_reducer._n_components_ or self.n_components

        # Clamp n_clusters so k-means never sees fewer samples than clusters.
        # This can happen when 'auto' Marchenko-Pastur gives very few components.
        effective_n_clusters = min(self.n_clusters, n_components)
        if effective_n_clusters < self.n_clusters:
            import warnings as _w
            _w.warn(
                f"[PCAKMeansReducer] n_clusters={self.n_clusters} reduced to "
                f"{effective_n_clusters} because only {n_components} PCA components "
                f"were retained (n_components='auto' gave fewer components than clusters).",
                UserWarning,
                stacklevel=2,
            )
            self.clusterer.kmeans.n_clusters = effective_n_clusters

        self.clusterer.fit(factor_features[:n_components])

        if self.model_residuals and self.residual_modeler is not None:
            X_reconstructed = self.pca_reducer.inverse_transform(X_reduced)
            residuals = (X - X_reconstructed).reshape(-1, X.shape[-1])
            self.residual_modeler.fit(residuals)
            # Keep std as fallback for dimensions where GMM may underfit
            self._residual_std = np.std(residuals, axis=0)

        self._fitted = True
        return self

    def transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Transform to reduced space.

        Args:
            X: Input data, shape (N, T, d)

        Returns:
            Reduced data, shape (N, T, m)
        """
        return self.pca_reducer.transform(X)

    def inverse_transform(self, X_reduced: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Reconstruct from reduced space.

        From Appendix C.2.4:
            X̂ = F̂ @ P_{1:m}^T + R̂

        Args:
            X_reduced: Reduced data, shape (N, T, m)

        Returns:
            Reconstructed data, shape (N, T, d)
        """
        X_reduced = to_numpy(X_reduced)

        X_factor = self.pca_reducer.inverse_transform(X_reduced)

        if (
            self.model_residuals
            and self.residual_modeler is not None
            and self.residual_modeler._fitted
        ):
            N, T, d = X_factor.shape
            # Sample per-asset GMM residuals (captures heavy tails and bimodality)
            residuals = self.residual_modeler.sample(N * T).reshape(N, T, d)
            X_factor = X_factor + residuals
        elif self.model_residuals and self._residual_std is not None:
            # Fallback: Gaussian residuals with empirical std
            N, T, d = X_factor.shape
            residuals = np.random.randn(N, T, d) * self._residual_std
            X_factor = X_factor + residuals

        return X_factor

    def fit_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(X)
        return self.transform(X)

    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster assignments for factors."""
        return self.clusterer.labels

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get PCA explained variance ratios."""
        return self.pca_reducer.explained_variance_ratio
