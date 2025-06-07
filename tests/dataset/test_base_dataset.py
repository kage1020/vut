import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vut.config import Config
from vut.dataset.base import BaseDataLoader, BaseDataset


@pytest.fixture
def class_mapping_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("index,class\n0,action1\n1,action2\n2,action3\n")
        file_path = f.name
    yield file_path
    os.remove(file_path)


@pytest.fixture
def split_file(feature_files):
    feature_data, feature_paths = feature_files
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for path in feature_paths:
            f.write(f"{path}\n")
        file_path = f.name
    yield file_path
    os.remove(file_path)


@pytest.fixture
def feature_files():
    temp_dir = tempfile.mkdtemp()
    feature_data = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[7, 8, 9], [10, 11, 12]]),
        np.array([[13, 14, 15], [16, 17, 18]]),
    ]

    feature_paths = []
    for i, data in enumerate(feature_data):
        feature_path = Path(temp_dir) / f"feature{i + 1}.npy"
        np.save(feature_path, data)
        feature_paths.append(feature_path)

    yield feature_data, feature_paths

    for path in feature_paths:
        if path.exists():
            path.unlink()
    os.rmdir(temp_dir)


@pytest.fixture
def gt_dir():
    temp_dir = tempfile.mkdtemp()
    gt_data = [
        ["action1", "action2", "action1"],
        ["action2", "action3", "action2"],
        ["action3", "action1", "action3"],
    ]

    gt_paths = []
    for i, data in enumerate(gt_data):
        gt_path = Path(temp_dir) / f"feature{i + 1}.txt"
        with open(gt_path, "w") as f:
            f.write("\n".join(data))
        gt_paths.append(gt_path)

    yield temp_dir

    for path in gt_paths:
        if path.exists():
            path.unlink()
    os.rmdir(temp_dir)


def test_base_dataset(class_mapping_file, feature_files, gt_dir, split_file):
    feature_data, feature_paths = feature_files
    cfg = Config.from_dict(
        {
            "dataset": {
                "name": "test_dataset",
                "class_mapping_path": class_mapping_file,
                "class_mapping_has_header": True,
                "class_mapping_separator": ",",
                "split_dir": str(Path(split_file).parent),
                "split_file_name": Path(split_file).name,
                "gt_dir": gt_dir,
            }
        }
    )

    dataset = BaseDataset(cfg=cfg)

    assert len(dataset) == 3
    assert dataset.text_to_index == {"action1": 0, "action2": 1, "action3": 2}
    assert dataset.index_to_text == {0: "action1", 1: "action2", 2: "action3"}

    feature, gt = dataset[0]

    assert isinstance(feature, torch.Tensor)
    assert isinstance(gt, torch.Tensor)
    assert torch.equal(feature, torch.from_numpy(feature_data[0]))
    assert torch.equal(gt, torch.tensor([0, 1, 0]))


def test_base_dataloader(class_mapping_file, gt_dir, split_file):
    cfg = Config.from_dict(
        {
            "dataset": {
                "name": "test_dataset",
                "class_mapping_path": class_mapping_file,
                "class_mapping_has_header": True,
                "class_mapping_separator": ",",
                "split_dir": str(Path(split_file).parent),
                "split_file_name": Path(split_file).name,
                "gt_dir": gt_dir,
            },
            "training": {
                "batch_size": 2,
                "shuffle": False,
            },
        }
    )

    dataset = BaseDataset(cfg=cfg)
    dataloader = BaseDataLoader(cfg=cfg, dataset=dataset)

    assert dataloader.batch_size == 2
    assert dataloader.dataset == dataset

    batch = next(iter(dataloader))
    features, gts = batch

    assert len(features) == 2
    assert len(gts) == 2
    assert isinstance(features, torch.Tensor)
    assert isinstance(gts, torch.Tensor)
