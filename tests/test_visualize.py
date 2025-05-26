import os
import tempfile

import numpy as np
import pytest
from pytest_mock import MockerFixture

from vut.visualize import (
    plot_action_segmentation,
    plot_feature,
    plot_features,
    plot_image,
    plot_images,
    plot_metrics,
    plot_palette,
    plot_scatter,
)


@pytest.fixture
def img():
    return np.ceil(np.random.rand(100, 100, 3) * 255).astype(np.uint8)


def test_plot_palette__save_as_file(mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.get_cmap", return_value="viridis")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_palette(name="viridis", path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_palette__with_palette():
    palette = np.random.rand(10, 3)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_palette(palette=palette, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_image__save_as_file(img):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_image(img, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_image__show_in_jupyter(img, mocker: MockerFixture):
    mock = mocker.patch("matplotlib.pyplot.show")
    plot_image(img, is_jupyter=True)
    mock.assert_called_once()


def test_plot_image__return_canvas(img):
    canvas = plot_image(img, return_canvas=True)
    assert canvas.shape == (480, 640, 3)


def test_plot_images__save_as_file(img):
    images = [img, img]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path1 = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path2 = tmp_file.name
    plot_images(images, paths=[path1, path2])
    assert os.path.exists(path1)
    assert os.path.exists(path2)
    os.remove(path1)
    os.remove(path2)


def test_plot_images__show_in_jupyter(img, mocker: MockerFixture):
    images = [img, img]
    mock = mocker.patch("matplotlib.pyplot.show")
    plot_images(images, is_jupyter=True)
    mock.assert_called_once()


def test_plot_images__return_canvas(img):
    images = [img, img]
    canvas = plot_images(images, return_canvas=True)
    assert len(canvas) == 2


def test_plot_feature__save_as_file():
    feature = np.random.rand(10, 10)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_feature(feature, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_feature__show_in_jupyter(mocker: MockerFixture):
    feature = np.random.rand(10, 10)
    mock = mocker.patch("matplotlib.pyplot.show")
    plot_feature(feature, is_jupyter=True)
    mock.assert_called_once()


def test_plot_feature__return_canvas():
    feature = np.random.rand(10, 10)
    canvas = plot_feature(feature, return_canvas=True)
    assert canvas.shape == (480, 640, 3)


def test_plot_features__save_as_file():
    features = [np.random.rand(10, 10), np.random.rand(10, 10)]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path1 = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path2 = tmp_file.name
    plot_features(features, paths=[path1, path2])
    assert os.path.exists(path1)
    assert os.path.exists(path2)
    os.remove(path1)
    os.remove(path2)


def test_plot_features__show_in_jupyter(mocker: MockerFixture):
    features = [np.random.rand(10, 10), np.random.rand(10, 10)]
    mock = mocker.patch("matplotlib.pyplot.show")
    plot_features(features, is_jupyter=True)
    mock.assert_called_once()


def test_plot_features__return_canvas():
    features = [np.random.rand(10, 10), np.random.rand(10, 10)]
    canvas = plot_features(features, return_canvas=True)
    assert len(canvas) == 2


def test_plot_scatter__save_as_file():
    tsne_result = np.random.rand(20, 2) * 10
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_scatter(tsne_result, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_scatter__with_labels():
    tsne_result = np.random.rand(20, 2) * 10
    labels = [0, 1, 0, 1] * 5
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_scatter(tsne_result, labels=labels, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_scatter__show_in_jupyter(mocker: MockerFixture):
    tsne_result = np.random.rand(20, 2) * 10
    mock = mocker.patch("matplotlib.pyplot.show")
    plot_scatter(tsne_result, is_jupyter=True)
    mock.assert_called_once()


def test_plot_scatter__return_canvas():
    tsne_result = np.random.rand(20, 2) * 10
    canvas = plot_scatter(tsne_result, return_canvas=True)
    assert canvas.shape == (800, 1000, 3)


def test_plot_metrics__save_as_file():
    metrics = {
        "train_loss": [1.0, 0.8, 0.6, 0.4, 0.2],
        "val_loss": [1.2, 0.9, 0.7, 0.5, 0.3],
        "accuracy": [0.5, 0.6, 0.7, 0.8, 0.9],
    }
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_metrics(metrics, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_metrics__show_in_jupyter(mocker: MockerFixture):
    metrics = {
        "train_loss": [1.0, 0.8, 0.6, 0.4, 0.2],
        "val_loss": [1.2, 0.9, 0.7, 0.5, 0.3],
    }
    mock = mocker.patch("matplotlib.pyplot.show")
    plot_metrics(metrics, is_jupyter=True)
    mock.assert_called_once()


def test_plot_metrics__return_canvas():
    metrics = {
        "train_loss": [1.0, 0.8, 0.6, 0.4, 0.2],
        "val_loss": [1.2, 0.9, 0.7, 0.5, 0.3],
    }
    canvas = plot_metrics(metrics, return_canvas=True)
    assert canvas.shape == (600, 1000, 3)


def test_plot_metrics__invalid_metric_data():
    metrics = {"invalid_metric": [[1, 2], [3, 4]]}
    with pytest.raises(
        AssertionError,
        match="Metric data for 'invalid_metric' must be 1D array or list",
    ):
        plot_metrics(metrics)


def test_plot_action_segmentation__save_as_file():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]
    confidences = [0.9, 0.7, 0.8, 0.9, 0.95, 0.85, 0.9, 0.88]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, prediction, confidences, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__show_in_jupyter(mocker: MockerFixture):
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]
    confidences = [0.9, 0.7, 0.8, 0.9, 0.95, 0.85, 0.9, 0.88]

    mock = mocker.patch("matplotlib.pyplot.show")
    plot_action_segmentation(ground_truth, prediction, confidences, is_jupyter=True)
    mock.assert_called_once()


def test_plot_action_segmentation__return_canvas():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]
    confidences = [0.9, 0.7, 0.8, 0.9, 0.95, 0.85, 0.9, 0.88]

    canvas = plot_action_segmentation(
        ground_truth, prediction, confidences, return_canvas=True
    )
    assert canvas.shape == (800, 1600, 3)


def test_plot_action_segmentation__without_prediction():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    confidences = [0.9, 0.7, 0.8, 0.9, 0.95, 0.85, 0.9, 0.88]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, None, confidences, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__without_confidences():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, prediction, None, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__ground_truth_only():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__with_labels():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]
    labels = ["Background", "Action1", "Action2"]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, prediction, labels=labels, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__custom_palette():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]
    palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, prediction, palette=palette, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__custom_legend_ncol():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 0]
    prediction = [0, 1, 1, 1, 2, 2, 1, 0]

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, prediction, legend_ncol=3, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__invalid_ground_truth_dimension():
    ground_truth = [[0, 1], [2, 3]]
    prediction = [0, 1, 2, 3]

    with pytest.raises(AssertionError, match="Ground truth must be a 1D array"):
        plot_action_segmentation(ground_truth, prediction)


def test_plot_action_segmentation__invalid_prediction_dimension():
    ground_truth = [0, 1, 2, 3]
    prediction = [[0, 1], [2, 3]]

    with pytest.raises(AssertionError, match="Prediction must be a 1D array"):
        plot_action_segmentation(ground_truth, prediction)


def test_plot_action_segmentation__length_mismatch():
    ground_truth = [0, 1, 2]
    prediction = [0, 1, 2, 3]

    with pytest.raises(
        AssertionError, match="Ground truth and prediction must have the same length"
    ):
        plot_action_segmentation(ground_truth, prediction)


def test_plot_action_segmentation__confidences_length_mismatch_with_prediction():
    ground_truth = [0, 1, 2, 3]
    prediction = [0, 1, 2, 3]
    confidences = [0.1, 0.2, 0.3]  # Different length

    with pytest.raises(
        AssertionError, match="Confidences and prediction must have the same length"
    ):
        plot_action_segmentation(ground_truth, prediction, confidences)
