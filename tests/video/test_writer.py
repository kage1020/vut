import os
import tempfile

import numpy as np
from PIL import Image

from vut.video.writer import VideoWriter


def test_video_writer__init():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name

    try:
        with VideoWriter(filename, framerate=30, size=(640, 480)) as writer:
            assert writer.filename == filename
            assert writer.framerate == 30
            assert writer.width == 640
            assert writer.height == 480
            assert writer.maxsize == 640
            assert writer.quality == 28
            assert writer.pix_fmt_in is None
            assert writer.pix_fmt_out == "yuv420p"
            assert writer.out is None

        assert os.path.exists(filename)
    finally:
        os.remove(filename)


def test_video_writer__resize():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name

    try:
        with VideoWriter(filename, framerate=30, size=(640, 480)) as writer:
            image = Image.new("RGB", (800, 600), color="blue")
            resized_image = writer._resize(image, max_size=640, image_size=image.size)

            assert resized_image.shape == (640, 480, 3)
            assert np.mean(resized_image) > 0
    finally:
        os.remove(filename)


def test_video_writer__update():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name

    try:
        with VideoWriter(filename, framerate=30, size=(640, 480)) as writer:
            image = Image.new("RGB", (640, 480), color="red")
            writer.update(image)

            assert os.path.exists(filename)
    finally:
        os.remove(filename)
