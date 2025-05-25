import os
import tempfile

import numpy as np
import pytest
from pytest_mock import MockerFixture

from vut.visualize import plot_image, plot_palette


@pytest.fixture
def img():
    return np.ceil(np.random.rand(100, 100, 3) * 255).astype(np.uint8)


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
