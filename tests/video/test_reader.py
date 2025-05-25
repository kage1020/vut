import os
import tempfile

import cv2
import numpy as np
import pytest

from vut.video.reader import VideoReader


@pytest.fixture
def video_file():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        filename = temp_file.name
    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (640, 480),
    )
    for i in range(10):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    yield filename
    os.remove(filename)


@pytest.fixture
def image_dir():
    dir_path = tempfile.mkdtemp()
    for i in range(10):
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(dir_path, f"image_{i:03d}.jpg"), img)
    yield dir_path
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file))
    os.rmdir(dir_path)


def test_video_reader__init_with_video(video_file):
    with VideoReader(video_path=video_file) as reader:
        assert len(reader) == 10
        assert reader.image_size == (640, 480)
        assert reader.index == 0


def test_video_reader__init_with_images(image_dir):
    with VideoReader(image_dir=image_dir) as reader:
        assert len(reader) == 10
        assert reader.image_size == (640, 480)
        assert reader.index == 0


def test_video_reader__next_with_video(video_file):
    with VideoReader(video_path=video_file) as reader:
        first_frame = next(reader)
        assert first_frame.shape == (480, 640, 3)
        assert reader.index == 1

        second_frame = next(reader)
        assert second_frame.shape == (480, 640, 3)
        assert reader.index == 2


def test_video_reader__next_with_images(image_dir):
    with VideoReader(image_dir=image_dir) as reader:
        first_image = next(reader)
        assert first_image.shape == (480, 640, 3)
        assert reader.index == 1

        second_image = next(reader)
        assert second_image.shape == (480, 640, 3)
        assert reader.index == 2
