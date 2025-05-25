from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vut.palette import ColorMapName, create_palette, template


def plot_palette(
    *,
    name: ColorMapName | None = None,
    palette: NDArray | list[tuple[float, float, float]] = None,
    n: int = 256,
    path: str | Path = "palette.png",
) -> None:
    """Plot a color palette.

    Args:
        name (ColorMapName | None, optional): The name of the colormap to use. Defaults to None.
        palette (NDArray | list[tuple[float, float, float]], optional): A custom color palette. Defaults to None.
        n (int, optional): The number of colors in the colormap. Defaults to 256.
        path (str | Path, optional): The file path to save the plot. Defaults to "palette.png".
    """
    assert name is not None or palette is not None, (
        "Either name or palette must be provided"
    )
    if name is not None:
        cmap = template(n, name)
    else:
        assert palette.ndim == 2, "Palette must be a 2D array"
        cmap = create_palette(palette)
    gradient = np.linspace(0, 1, n)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(
        np.vstack((gradient, gradient)),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_axis_off()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_image(
    data: NDArray,
    path: str | Path = "image.png",
    title: str = None,
    show_axis: bool = False,
    is_jupyter: bool = False,
    return_canvas: bool = False,
) -> NDArray | None:
    """Plot a 3D array as an image.

    Args:
        data (NDArray): The 3D array to plot.
        path (str | Path, optional): The file path to save the image. Defaults to "image.png".
        title (str, optional): The title of the plot. Defaults to None.
        show_axis (bool, optional): Whether to show the axis. Defaults to False.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.

    Returns:
        NDArray: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    assert data.ndim == 3, "Data must be a 3D array"

    fig, ax = plt.subplots()
    ax.imshow(data)
    if not show_axis:
        ax.axis("off")

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if return_canvas:
        fig.canvas.draw()
        canvas = np.array(fig.canvas.buffer_rgba())
        plt.close(fig)
        return canvas[:, :, :3]

    if is_jupyter:
        plt.show()
        plt.close(fig)
    else:
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
