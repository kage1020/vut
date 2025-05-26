from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vut.palette import RGB, ColorMapName, create_palette, template


def plot_palette(
    *,
    name: ColorMapName | None = None,
    palette: NDArray | list[RGB] = None,
    n: int = 256,
    path: str | Path = "palette.png",
) -> None:
    """Plot a color palette.

    Args:
        name (ColorMapName | None, optional): The name of the colormap to use. Defaults to None.
        palette (NDArray | list[RGB], optional): A custom color palette. Defaults to None.
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
    else:
        plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_images(
    data: list[NDArray],
    paths: list[str | Path] | None = None,
    titles: list[str] | None = None,
    show_axis: bool = False,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    ncols: int = None,
    nrows: int = None,
) -> list[NDArray] | None:
    """Plot a list of 3D arrays as images.

    Args:
        data (list[NDArray]): List of 3D arrays to plot.
        paths (list[str | Path] | None): List of file paths to save the images. Defaults to None.
        titles (list[str] | None, optional): List of titles for each plot. Defaults to None.
        show_axis (bool, optional): Whether to show the axis. Defaults to False.
        is_jupyter (bool, optional): Whether to display the plots in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvases as numpy arrays. Defaults to False.
        ncols (int, optional): Number of columns in the grid layout. Defaults to None.
        nrows (int, optional): Number of rows in the grid layout. Defaults to None.

    Returns:
        list[NDArray]: List of canvases as numpy arrays if return_canvas is True, otherwise None.
    """
    assert all(d.ndim == 3 for d in data), "All data must be 3D arrays"
    assert paths is None or len(paths) == len(data), (
        "Paths must be provided for each image if specified"
    )
    assert titles is None or len(titles) == len(data), (
        "Titles must be provided for each image if specified"
    )

    if paths is None:
        paths = [None] * len(data)
    if titles is None:
        titles = [None] * len(data)

    num_images = len(data)
    if ncols is None and nrows is None:
        ncols = int(np.ceil(np.sqrt(num_images)))
        nrows = int(np.ceil(num_images / ncols))
    elif ncols is None:
        ncols = int(np.ceil(num_images / nrows))
    elif nrows is None:
        nrows = int(np.ceil(num_images / ncols))
    assert ncols * nrows >= num_images, (
        "Number of columns and rows must accommodate all images"
    )

    canvases = []
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    axs = axs.flatten() if nrows > 1 or ncols > 1 else [axs]
    for i, (ax, img, path) in enumerate(zip(axs, data, paths)):
        ax.imshow(img)
        if not show_axis:
            ax.axis("off")
        if titles and titles[i]:
            ax.set_title(titles[i])
        if return_canvas:
            fig.canvas.draw()
            canvas = np.array(fig.canvas.buffer_rgba())
            canvases.append(canvas[:, :, :3])
        elif path:
            plt.savefig(path, bbox_inches="tight")
    plt.tight_layout()
    if return_canvas:
        plt.close(fig)
        return canvases
    if is_jupyter:
        plt.show()
        plt.close(fig)


def plot_feature(
    data: NDArray,
    path: str | Path = "feature.png",
    title: str = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    palette: ColorMapName | list[RGB] | None = "plasma",
) -> NDArray | None:
    """Plot a 2D feature map.

    Args:
        data (NDArray): The 2D array to plot.
        path (str | Path, optional): The file path to save the image. Defaults to "feature.png".
        title (str, optional): The title of the plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        palette (ColorMapName | list[RGB], optional): The colormap to use. Defaults to "plasma".

    Returns:
        NDArray: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    assert data.ndim == 2, "Data must be a 2D array"

    fig, ax = plt.subplots()
    cax = ax.imshow(
        data, cmap=create_palette(palette) if isinstance(palette, list) else palette
    )
    if title:
        ax.set_title(title)
    fig.colorbar(cax)

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


def plot_features(
    data: list[NDArray],
    paths: list[str | Path] | None = None,
    titles: list[str] | None = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    ncols: int = None,
    nrows: int = None,
    palette: ColorMapName | list[RGB] | None = "plasma",
) -> list[NDArray] | None:
    """Plot a list of 2D feature maps.

    Args:
        data (list[NDArray]): List of 2D arrays to plot.
        paths (list[str | Path] | None): List of file paths to save the images. Defaults to None.
        titles (list[str] | None, optional): List of titles for each plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plots in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvases as numpy arrays. Defaults to False.
        ncols (int, optional): Number of columns in the grid layout. Defaults to None.
        nrows (int, optional): Number of rows in the grid layout. Defaults to None.
        palette (ColorMapName | list[RGB], optional): The colormap to use. Defaults to "plasma".

    Returns:
        list[NDArray]: List of canvases as numpy arrays if return_canvas is True, otherwise None.
    """
    assert all(d.ndim == 2 for d in data), "All data must be 2D arrays"
    assert paths is None or len(paths) == len(data), (
        "Paths must be provided for each image if specified"
    )
    assert titles is None or len(titles) == len(data), (
        "Titles must be provided for each image if specified"
    )

    if paths is None:
        paths = [None] * len(data)
    if titles is None:
        titles = [None] * len(data)

    num_features = len(data)
    if ncols is None and nrows is None:
        ncols = int(np.ceil(np.sqrt(num_features)))
        nrows = int(np.ceil(num_features / ncols))
    elif ncols is None:
        ncols = int(np.ceil(num_features / nrows))
    elif nrows is None:
        nrows = int(np.ceil(num_features / ncols))

    canvases = []
    fig, axs = plt.subplots(
        nrows=nrows if nrows is not None else 1,
        ncols=ncols if ncols is not None else len(data),
        figsize=(ncols * 3, nrows * 3) if nrows and ncols else (len(data) * 3, 3),
    )
    axs = axs.flatten() if nrows > 1 or ncols > 1 else [axs]
    for i, (ax, img, path) in enumerate(zip(axs, data, paths)):
        cax = ax.imshow(
            img,
            cmap=create_palette(palette) if isinstance(palette, list) else palette,
        )
        if titles and titles[i]:
            ax.set_title(titles[i])
        fig.colorbar(cax, ax=ax)
        if return_canvas:
            fig.canvas.draw()
            canvas = np.array(fig.canvas.buffer_rgba())
            canvases.append(canvas[:, :, :3])
        elif path:
            plt.savefig(path, bbox_inches="tight")
    plt.tight_layout()
    if return_canvas:
        plt.close(fig)
        return canvases
    if is_jupyter:
        plt.show()
        plt.close(fig)


def plot_scatter(
    data: NDArray,
    labels: list[str] | None = None,
    path: str | Path = "scatter.png",
    title: str | None = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    palette: ColorMapName | list[RGB] | None = "plasma",
) -> NDArray | None:
    """Plot a 2D scatter plot of the data.

    Args:
        data (NDArray): 2D array of shape (n_samples, 2) representing the data points.
        labels (list[str] | None, optional): List of labels for each data point. Defaults to None.
        path (str | Path, optional): File path to save the plot. Defaults to "scatter.png".
        title (str | None, optional): Title of the plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        palette (ColorMapName | list[RGB] | None, optional): Colormap to use for the plot. Defaults to "plasma".

    Returns:
        NDArray | None: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    assert data.ndim == 2, "Data must be a 2D array"
    assert data.shape[1] == 2, "Data must be 2D embedding (n_samples, 2)"

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = create_palette(palette) if isinstance(palette, list) else palette
    scatter = ax.scatter(
        data[:, 0],
        data[:, 1],
        c=labels if labels is not None else "blue",
        cmap=cmap if labels is not None else None,
        s=50,
        alpha=0.7,
    )

    if labels is not None:
        assert len(labels) == data.shape[0], "Labels must match number of samples"
        legend = ax.legend(*scatter.legend_elements())
        ax.add_artist(legend)

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
    else:
        plt.savefig(path, bbox_inches="tight")

    plt.close(fig)


def plot_metrics(
    metrics: dict[str, NDArray | list[int] | list[float]],
    path: str | Path = "metrics.png",
    title: str | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "Value",
    is_jupyter: bool = False,
    return_canvas: bool = False,
    figsize: tuple[int, int] = (10, 6),
) -> NDArray | None:
    """Plot machine learning metrics over time.

    Args:
        metrics (dict[str, NDArray | list[int] | list[float]]): Dictionary where keys are metric names and values are metric data.
        path (str | Path, optional): File path to save the plot. Defaults to "metrics.png".
        title (str | None, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to "Epoch".
        ylabel (str, optional): Label for the y-axis. Defaults to "Value".
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (10, 6).

    Returns:
        NDArray | None: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in metrics.items():
        data = np.array(data)

        assert data.ndim == 1, f"Metric data for '{name}' must be 1D array or list"

        x = np.arange(len(data))
        ax.plot(x, data, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

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
    else:
        plt.savefig(path, bbox_inches="tight")

    plt.close(fig)


def plot_action_segmentation(
    ground_truth: NDArray | list[int],
    prediction: NDArray | list[int] | None = None,
    confidences: NDArray | list[float] | None = None,
    path: str | Path = "action_segmentation.png",
    title: str | None = None,
    labels: list[str] | None = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    figsize: tuple[int, int] = (16, 8),
    palette: ColorMapName | list[RGB] | None = "plasma",
    legend_ncol: int = 5,
) -> NDArray | None:
    """Plot action segmentation results with ground truth, predictions, and optional confidences.

    Args:
        ground_truth (NDArray | list[int]): Ground truth action labels.
        prediction (NDArray | list[int] | None, optional): Predicted action labels. Defaults to None.
        confidences (NDArray | list[float] | None, optional): Confidence scores for predictions. Defaults to None.
        path (str | Path, optional): File path to save the plot. Defaults to "action_segmentation.png".
        title (str | None, optional): Title of the plot. Defaults to None.
        labels (list[str] | None, optional): Names of action classes. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (16, 8).
        palette (ColorMapName | list[RGB] | None, optional): Colormap to use for the plot. Defaults to "plasma".
        legend_ncol (int, optional): Number of columns in the legend. Defaults to 5.

    Returns:
        NDArray | None: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    ground_truth = np.array(ground_truth)

    assert ground_truth.ndim == 1, "Ground truth must be a 1D array"

    if prediction is not None:
        prediction = np.array(prediction)
        assert prediction.ndim == 1, "Prediction must be a 1D array"
        assert len(ground_truth) == len(prediction), (
            "Ground truth and prediction must have the same length"
        )

    if confidences is not None:
        confidences = np.array(confidences)
        if prediction is not None:
            assert len(confidences) == len(prediction), (
                "Confidences and prediction must have the same length"
            )
        else:
            assert len(confidences) == len(ground_truth), (
                "Confidences and ground truth must have the same length"
            )

    n_frames = len(ground_truth)
    x = np.arange(n_frames)

    if prediction is not None:
        unique_classes = np.unique(np.concatenate([ground_truth, prediction]))
    else:
        unique_classes = np.unique(ground_truth)
    n_classes = len(unique_classes)

    if isinstance(palette, list):
        cmap = create_palette(palette)
    else:
        cmap = plt.get_cmap(palette)

    colors = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    n_subplots = 1
    if prediction is not None:
        n_subplots += 1
    if confidences is not None:
        n_subplots += 1

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]

    ax_gt = axes[0]
    for i in range(n_frames):
        ax_gt.bar(x[i], 1, color=class_to_color[ground_truth[i]], width=1.0, alpha=0.8)
    ax_gt.set_xlim(0, n_frames - 1)
    ax_gt.set_ylabel("Ground Truth")
    ax_gt.set_ylim(0, 1)
    ax_gt.set_yticks([])

    current_axis_idx = 1
    if prediction is not None:
        ax_pred = axes[current_axis_idx]
        for i in range(n_frames):
            ax_pred.bar(
                x[i], 1, color=class_to_color[prediction[i]], width=1.0, alpha=0.8
            )
        ax_pred.set_xlim(0, n_frames - 1)
        ax_pred.set_ylabel("Prediction")
        ax_pred.set_ylim(0, 1)
        ax_pred.set_yticks([])
        current_axis_idx += 1

    if confidences is not None:
        ax_conf = axes[current_axis_idx]
        confidences = np.array(confidences)

        if confidences.ndim == 1:
            ax_conf.plot(x, confidences)
        elif confidences.ndim == 2:
            for cls_idx, cls in enumerate(unique_classes):
                if cls_idx < confidences.shape[1]:
                    ax_conf.plot(
                        x,
                        confidences[:, cls_idx],
                        color=class_to_color[cls],
                        label=labels[cls]
                        if labels is not None and cls < len(labels)
                        else f"Class {cls}",
                    )

        ax_conf.set_xlim(0, n_frames - 1)
        ax_conf.set_ylabel("Confidence")
        ax_conf.set_ylim(0, 1)
        ax_conf.grid(True, alpha=0.3)

    legend_elements = []
    for cls in unique_classes:
        if labels is not None and cls < len(labels):
            label = labels[cls]
        else:
            label = cls
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=class_to_color[cls], label=label)
        )

    ncol = min(len(legend_elements), legend_ncol)
    nrow = int(np.ceil(len(legend_elements) / ncol))

    grid = []
    for row in range(nrow):
        row_elements = []
        for col in range(ncol):
            idx = row * ncol + col
            if idx < len(legend_elements):
                row_elements.append(legend_elements[idx])
            else:
                row_elements.append(None)
        grid.append(row_elements)

    reordered_elements = []
    for col in range(ncol):
        for row in range(nrow):
            if grid[row][col] is not None:
                reordered_elements.append(grid[row][col])

    fig.legend(
        handles=reordered_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=ncol,
    )

    axes[-1].set_xlabel("Frame")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if return_canvas:
        fig.canvas.draw()
        canvas = np.array(fig.canvas.buffer_rgba())
        plt.close(fig)
        return canvas[:, :, :3]

    if is_jupyter:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight")

    plt.close(fig)
