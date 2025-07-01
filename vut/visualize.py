import warnings
from pathlib import Path

from numpy.typing import NDArray

from vut import visualization
from vut.palette import RGB, ColorMapName


def plot_palette(
    *,
    name: ColorMapName | None = None,
    palette: NDArray | list[RGB] = None,
    n: int = 256,
    path: str | Path = "palette.png",
    figsize: tuple[int, int] = (8, 2),
) -> None:
    """Plot a color palette.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_palette` instead.

    Args:
        name (ColorMapName | None, optional): The name of the colormap to use. Defaults to None.
        palette (NDArray | list[RGB], optional): A custom color palette. Defaults to None.
        n (int, optional): The number of colors in the colormap. Defaults to 256.
        path (str | Path, optional): The file path to save the plot. Defaults to "palette.png".
        figsize (tuple[int, int], optional): The size of the figure. Defaults to (8, 2).
    """
    warnings.warn(
        "vut.visualize.plot_palette is deprecated. Use vut.visualization.plot_palette instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_palette(
        name=name, palette=palette, n=n, path=path, figsize=figsize
    )


def plot_image(
    data: NDArray,
    path: str | Path = "image.png",
    title: str = None,
    show_axis: bool = False,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    figsize: tuple[int, int] = (8, 6),
) -> NDArray | None:
    """Plot a 3D array as an image.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_image` instead.

    Args:
        data (NDArray): The 3D array to plot.
        path (str | Path, optional): The file path to save the image. Defaults to "image.png".
        title (str, optional): The title of the plot. Defaults to None.
        show_axis (bool, optional): Whether to show the axis. Defaults to False.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        figsize (tuple[int, int], optional): The size of the figure. Defaults to (8, 6).

    Returns:
        NDArray: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_image is deprecated. Use vut.visualization.plot_image instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_image(
        data=data,
        path=path,
        title=title,
        show_axis=show_axis,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        figsize=figsize,
    )


def plot_images(
    data: list[NDArray],
    paths: list[str | Path] | None = None,
    titles: list[str] | None = None,
    show_axis: bool = False,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    ncols: int = None,
    nrows: int = None,
    figsize: tuple[int, int] | None = None,
) -> list[NDArray] | None:
    """Plot a list of 3D arrays as images.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_images` instead.

    Args:
        data (list[NDArray]): List of 3D arrays to plot.
        paths (list[str | Path] | None): List of file paths to save the images. Defaults to None.
        titles (list[str] | None, optional): List of titles for each plot. Defaults to None.
        show_axis (bool, optional): Whether to show the axis. Defaults to False.
        is_jupyter (bool, optional): Whether to display the plots in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvases as numpy arrays. Defaults to False.
        ncols (int, optional): Number of columns in the grid layout. Defaults to None.
        nrows (int, optional): Number of rows in the grid layout. Defaults to None.
        figsize (tuple[int, int] | None, optional): The size of the figure. If None, defaults to (ncols * 5, nrows * 5). Defaults to None.

    Returns:
        list[NDArray]: List of canvases as numpy arrays if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_images is deprecated. Use vut.visualization.plot_images instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_images(
        data=data,
        paths=paths,
        titles=titles,
        show_axis=show_axis,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
    )


def plot_feature(
    data: NDArray,
    path: str | Path = "feature.png",
    title: str = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    palette: ColorMapName | list[RGB] | None = "plasma",
    figsize: tuple[int, int] = (8, 6),
) -> NDArray | None:
    """Plot a 2D feature map.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_feature` instead.

    Args:
        data (NDArray): The 2D array to plot.
        path (str | Path, optional): The file path to save the image. Defaults to "feature.png".
        title (str, optional): The title of the plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        palette (ColorMapName | list[RGB], optional): The colormap to use. Defaults to "plasma".
        figsize (tuple[int, int], optional): The size of the figure. Defaults to (8, 6).

    Returns:
        NDArray: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_feature is deprecated. Use vut.visualization.plot_feature instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_feature(
        data=data,
        path=path,
        title=title,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        palette=palette,
        figsize=figsize,
    )


def plot_features(
    data: list[NDArray],
    paths: list[str | Path] | None = None,
    titles: list[str] | None = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    ncols: int = None,
    nrows: int = None,
    palette: ColorMapName | list[RGB] | None = "plasma",
    figsize: tuple[int, int] | None = None,
) -> list[NDArray] | None:
    """Plot a list of 2D feature maps.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_features` instead.

    Args:
        data (list[NDArray]): List of 2D arrays to plot.
        paths (list[str | Path] | None): List of file paths to save the images. Defaults to None.
        titles (list[str] | None, optional): List of titles for each plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plots in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvases as numpy arrays. Defaults to False.
        ncols (int, optional): Number of columns in the grid layout. Defaults to None.
        nrows (int, optional): Number of rows in the grid layout. Defaults to None.
        palette (ColorMapName | list[RGB], optional): The colormap to use. Defaults to "plasma".
        figsize (tuple[int, int] | None, optional): The size of the figure. If None, defaults to (ncols * 5, nrows * 5). Defaults to None.

    Returns:
        list[NDArray]: List of canvases as numpy arrays if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_features is deprecated. Use vut.visualization.plot_features instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_features(
        data=data,
        paths=paths,
        titles=titles,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        ncols=ncols,
        nrows=nrows,
        palette=palette,
        figsize=figsize,
    )


def plot_scatter(
    data: NDArray,
    labels: list[str] | None = None,
    path: str | Path = "scatter.png",
    title: str | None = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    palette: ColorMapName | list[RGB] | None = "plasma",
    figsize: tuple[int, int] = (10, 8),
) -> NDArray | None:
    """Plot a 2D scatter plot of the data.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_scatter` instead.

    Args:
        data (NDArray): 2D array of shape (n_samples, 2) representing the data points.
        labels (list[str] | None, optional): List of labels for each data point. Defaults to None.
        path (str | Path, optional): File path to save the plot. Defaults to "scatter.png".
        title (str | None, optional): Title of the plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        palette (ColorMapName | list[RGB] | None, optional): Colormap to use for the plot. Defaults to "plasma".
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (10, 8).

    Returns:
        NDArray | None: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_scatter is deprecated. Use vut.visualization.plot_scatter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_scatter(
        data=data,
        labels=labels,
        path=path,
        title=title,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        palette=palette,
        figsize=figsize,
    )


def plot_metrics(
    metrics: dict[str, NDArray | list[int] | list[float]],
    path: str | Path = "metrics.png",
    title: str | None = None,
    x_label: str = "Epoch",
    y_label: str = "Value",
    is_jupyter: bool = False,
    return_canvas: bool = False,
    figsize: tuple[int, int] = (10, 6),
) -> NDArray | None:
    """Plot machine learning metrics over time.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_metrics` instead.

    Args:
        metrics (dict[str, NDArray | list[int] | list[float]]): Dictionary where keys are metric names and values are metric data.
        path (str | Path, optional): File path to save the plot. Defaults to "metrics.png".
        title (str | None, optional): Title of the plot. Defaults to None.
        x_label (str, optional): Label for the x-axis. Defaults to "Epoch".
        y_label (str, optional): Label for the y-axis. Defaults to "Value".
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (10, 6).

    Returns:
        NDArray | None: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_metrics is deprecated. Use vut.visualization.plot_metrics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_metrics(
        metrics=metrics,
        path=path,
        title=title,
        x_label=x_label,
        y_label=y_label,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        figsize=figsize,
    )


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

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_action_segmentation` instead.

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
    warnings.warn(
        "vut.visualize.plot_action_segmentation is deprecated. Use vut.visualization.plot_action_segmentation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_action_segmentation(
        ground_truth=ground_truth,
        prediction=prediction,
        confidences=confidences,
        path=path,
        title=title,
        labels=labels,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        figsize=figsize,
        palette=palette,
        legend_ncol=legend_ncol,
    )


def plot_roc_curve(
    ground_truth: NDArray | list[int],
    prediction: NDArray | list[float],
    path: str | Path = "roc_curve.png",
    title: str | None = None,
    is_jupyter: bool = False,
    return_canvas: bool = False,
    figsize: tuple[int, int] = (8, 6),
) -> NDArray | None:
    """Plot ROC (Receiver Operating Characteristic) curve.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.plot_roc_curve` instead.

    Args:
        ground_truth (NDArray | list[int]): Ground truth binary labels (0 or 1).
        prediction (NDArray | list[float]): Prediction scores or probabilities.
        path (str | Path, optional): File path to save the plot. Defaults to "roc_curve.png".
        title (str | None, optional): Title of the plot. Defaults to None.
        is_jupyter (bool, optional): Whether to display the plot in a Jupyter notebook. Defaults to False.
        return_canvas (bool, optional): Whether to return the canvas as a numpy array. Defaults to False.
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (8, 6).

    Returns:
        NDArray | None: The canvas as a numpy array if return_canvas is True, otherwise None.
    """
    warnings.warn(
        "vut.visualize.plot_roc_curve is deprecated. Use vut.visualization.plot_roc_curve instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.plot_roc_curve(
        ground_truth=ground_truth,
        prediction=prediction,
        path=path,
        title=title,
        is_jupyter=is_jupyter,
        return_canvas=return_canvas,
        figsize=figsize,
    )


def make_video(
    image_dir: str | Path | None = None,
    image_paths: list[str | Path] | None = None,
    ground_truth: NDArray | list[int] | None = None,
    prediction: NDArray | list[int] | None = None,
    confidences: NDArray | list[float] | None = None,
    path: str | Path = "video.mp4",
    title: str | None = None,
    labels: list[str] | None = None,
    fps: int = 30,
    figsize: tuple[int, int] = (16, 9),
    palette: ColorMapName | list[RGB] | None = "plasma",
    legend_ncol: int = 5,
    show_segmentation: bool = True,
    show_confidence: bool = True,
) -> None:
    """Create a video from images with optional action segmentation and confidence overlays.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use `vut.visualization.make_video` instead.

    Args:
        image_dir (str | Path | None, optional): Directory containing images. Defaults to None.
        image_paths (list[str | Path] | None, optional): List of image file paths. Defaults to None.
        ground_truth (NDArray | list[int] | None, optional): Ground truth action labels. Defaults to None.
        prediction (NDArray | list[int] | None, optional): Predicted action labels. Defaults to None.
        confidences (NDArray | list[float] | None, optional): Confidence scores. Defaults to None.
        path (str | Path, optional): Output video file path. Defaults to "video.mp4".
        title (str | None, optional): Title of the video. Defaults to None.
        labels (list[str] | None, optional): Names of action classes. Defaults to None.
        fps (int, optional): Frames per second. Defaults to 30.
        figsize (tuple[int, int], optional): Figure size (width, height). Defaults to (16, 9).
        palette (ColorMapName | list[RGB] | None, optional): Colormap to use. Defaults to "plasma".
        legend_ncol (int, optional): Number of columns in the legend. Defaults to 5.
        show_segmentation (bool, optional): Whether to show segmentation overlay. Defaults to True.
        show_confidence (bool, optional): Whether to show confidence plot. Defaults to True.
    """
    warnings.warn(
        "vut.visualize.make_video is deprecated. Use vut.visualization.make_video instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return visualization.make_video(
        image_dir=image_dir,
        image_paths=image_paths,
        ground_truth=ground_truth,
        prediction=prediction,
        confidences=confidences,
        path=path,
        title=title,
        labels=labels,
        fps=fps,
        figsize=figsize,
        palette=palette,
        legend_ncol=legend_ncol,
        show_segmentation=show_segmentation,
        show_confidence=show_confidence,
    )
