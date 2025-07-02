import os
import tempfile

import numpy as np
import pytest
from pytest_mock import MockerFixture

from vut.visualization import (
    make_video,
    plot_action_segmentation,
    plot_feature,
    plot_features,
    plot_image,
    plot_images,
    plot_metrics,
    plot_palette,
    plot_roc_curve,
    plot_scatter,
)


@pytest.fixture
def img():
    return np.ceil(np.random.rand(100, 100, 3) * 255).astype(np.uint8)


@pytest.fixture
def mock_ffmpeg(mocker: MockerFixture):
    """Mock ffmpeg to avoid dependency on ffmpeg installation"""
    # Mock the VideoWriter class instead of ffmpeg directly
    mock_writer = mocker.Mock()
    mock_writer.__enter__ = mocker.Mock(return_value=mock_writer)
    mock_writer.__exit__ = mocker.Mock(return_value=None)
    mock_writer.update = mocker.Mock()
    
    mock_writer_class = mocker.patch("vut.visualization.VideoWriter")
    mock_writer_class.return_value = mock_writer
    
    return mock_writer_class


def test_plot_feature__save_as_file():
    feature = np.random.rand(10, 10)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_feature(feature, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_feature__show_in_jupyter(mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    feature = np.random.rand(10, 10)
    plot_feature(feature, is_jupyter=True)


def test_plot_feature__return_canvas():
    feature = np.random.rand(10, 10)
    canvas = plot_feature(feature, return_canvas=True)
    assert canvas is not None
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3


def test_plot_features__save_as_file():
    features = [np.random.rand(10, 10), np.random.rand(10, 10)]
    paths = []
    for i in range(len(features)):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            paths.append(tmp_file.name)
    plot_features(features, paths=paths)
    for path in paths:
        assert os.path.exists(path)
        os.remove(path)


def test_plot_features__show_in_jupyter(mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    features = [np.random.rand(10, 10), np.random.rand(10, 10)]
    plot_features(features, is_jupyter=True)


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
    mocker.patch("matplotlib.pyplot.show")
    tsne_result = np.random.rand(20, 2) * 10
    plot_scatter(tsne_result, is_jupyter=True)


def test_plot_scatter__return_canvas():
    tsne_result = np.random.rand(20, 2) * 10
    canvas = plot_scatter(tsne_result, return_canvas=True)
    assert canvas is not None
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3


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
    mocker.patch("matplotlib.pyplot.show")
    metrics = {
        "train_loss": [1.0, 0.8, 0.6, 0.4, 0.2],
        "val_loss": [1.2, 0.9, 0.7, 0.5, 0.3],
        "accuracy": [0.5, 0.6, 0.7, 0.8, 0.9],
    }
    plot_metrics(metrics, is_jupyter=True)


def test_plot_metrics__return_canvas():
    metrics = {
        "train_loss": [1.0, 0.8, 0.6, 0.4, 0.2],
        "val_loss": [1.2, 0.9, 0.7, 0.5, 0.3],
        "accuracy": [0.5, 0.6, 0.7, 0.8, 0.9],
    }
    canvas = plot_metrics(metrics, return_canvas=True)
    assert canvas is not None
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3


def test_plot_metrics__invalid_metric_data():
    metrics = {
        "train_loss": np.array([[1.0, 0.8], [0.6, 0.4]]),  # Invalid 2D data
    }
    with pytest.raises(AssertionError):
        plot_metrics(metrics)


def test_plot_action_segmentation__save_as_file():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 1, 0, 0]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_action_segmentation(ground_truth, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_action_segmentation__show_in_jupyter(mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 1, 0, 0]
    plot_action_segmentation(ground_truth, is_jupyter=True)


def test_plot_action_segmentation__return_canvas():
    ground_truth = [0, 0, 1, 1, 2, 2, 1, 1, 0, 0]
    canvas = plot_action_segmentation(ground_truth, return_canvas=True)
    assert canvas is not None
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3


def test_plot_roc_curve__save_as_file():
    ground_truth = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    prediction = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.1, 0.2, 0.9]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        path = tmp_file.name
    plot_roc_curve(ground_truth, prediction, path=path)
    assert os.path.exists(path)
    os.remove(path)


def test_plot_roc_curve__show_in_jupyter(mocker: MockerFixture):
    mocker.patch("matplotlib.pyplot.show")
    ground_truth = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    prediction = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.1, 0.2, 0.9]
    plot_roc_curve(ground_truth, prediction, is_jupyter=True)


def test_plot_roc_curve__return_canvas():
    ground_truth = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
    prediction = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.1, 0.2, 0.9]
    canvas = plot_roc_curve(ground_truth, prediction, return_canvas=True)
    assert canvas is not None
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3


def test_plot_roc_curve__length_mismatch():
    ground_truth = [0, 0, 1, 1]
    prediction = [0.1, 0.2, 0.8]  # Length mismatch
    with pytest.raises(AssertionError):
        plot_roc_curve(ground_truth, prediction)


def test_plot_roc_curve__invalid_ground_truth_labels():
    ground_truth = [0, 0, 1, 2]  # Invalid label "2"
    prediction = [0.1, 0.2, 0.8, 0.9]
    with pytest.raises(AssertionError):
        plot_roc_curve(ground_truth, prediction)


def test_make_video__with_image_paths(mock_ffmpeg):
    # Create temporary image files
    image_paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Create a simple test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(tmp_file.name)
            image_paths.append(tmp_file.name)
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            video_path = tmp_video.name
        
        make_video(image_paths=image_paths, path=video_path)
        
        # Verify that ffmpeg was called
        mock_ffmpeg.assert_called()
        
    finally:
        # Clean up
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(video_path):
            os.remove(video_path)


def test_make_video__with_labels_and_data(mock_ffmpeg):
    # Create temporary image files
    image_paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Create a simple test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(tmp_file.name)
            image_paths.append(tmp_file.name)
    
    ground_truth = [0, 1, 0]
    prediction = [0, 1, 1]
    labels = ["action_A", "action_B"]
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            video_path = tmp_video.name
        
        make_video(
            image_paths=image_paths,
            ground_truth=ground_truth,
            prediction=prediction,
            labels=labels,
            path=video_path
        )
        
        # Verify that ffmpeg was called
        mock_ffmpeg.assert_called()
        
    finally:
        # Clean up
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(video_path):
            os.remove(video_path)


def test_make_video__no_segmentation_data(mock_ffmpeg):
    # Create temporary image files
    image_paths = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Create a simple test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(tmp_file.name)
            image_paths.append(tmp_file.name)
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            video_path = tmp_video.name
        
        make_video(
            image_paths=image_paths,
            path=video_path,
            show_segmentation=False,
            show_confidence=False
        )
        
        # Verify that ffmpeg was called
        mock_ffmpeg.assert_called()
        
    finally:
        # Clean up
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(video_path):
            os.remove(video_path)


def test_make_video__invalid_inputs():
    # Test with no image sources
    with pytest.raises(AssertionError):
        make_video()