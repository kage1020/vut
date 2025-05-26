from numpy.typing import NDArray
from sklearn.manifold import TSNE


def compute_tsne(
    data: NDArray,
    n_components: int = 2,
    random_state: int | None = 42,
) -> NDArray:
    """Compute t-SNE dimensionality reduction.

    Args:
        data (NDArray): Input data matrix of shape (n_samples, n_features).
        n_components (int, optional): Dimension of the embedded space. Defaults to 2.
        random_state (int | None, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        NDArray: t-SNE embedding of the data with shape (n_samples, n_components).
    """
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
    )
    return tsne.fit_transform(data)
