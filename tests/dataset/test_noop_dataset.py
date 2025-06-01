import os
import tempfile

import pytest
import torch

from vut.config import Config
from vut.dataset.noop import NoopDataLoader, NoopDataset


@pytest.fixture
def class_mapping_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("index,class\n0,action1\n1,action2\n2,action3\n")
        file_path = f.name
    yield file_path
    os.remove(file_path)


def test_noop_dataset__init(class_mapping_file):
    cfg = Config.from_dict(
        {
            "dataset": {
                "name": "test_dataset",
                "class_mapping_path": class_mapping_file,
                "class_mapping_has_header": True,
                "class_mapping_separator": ",",
            }
        }
    )

    dataset = NoopDataset(cfg=cfg)

    assert dataset.text_to_index == {"action1": 0, "action2": 1, "action3": 2}
    assert dataset.index_to_text == {0: "action1", 1: "action2", 2: "action3"}
    assert len(dataset) == 0


def test_noop_dataset__getitem():
    cfg = Config.from_dict(
        {
            "dataset": {
                "name": "test_dataset",
            }
        }
    )

    dataset = NoopDataset(cfg=cfg)

    result = dataset[0]
    expected = torch.tensor([])

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, expected)

    assert torch.equal(dataset[5], expected)
    assert torch.equal(dataset[100], expected)


def test_noop_dataloader():
    cfg = Config.from_dict(
        {
            "dataset": {
                "name": "test_dataset",
            },
            "training": {
                "batch_size": 4,
                "shuffle": False,
            },
        }
    )

    dataset = NoopDataset(cfg=cfg)
    dataloader = NoopDataLoader(cfg=cfg, dataset=dataset)

    assert dataloader.batch_size == 4
    assert dataloader.dataset == dataset

    items = list(dataloader)
    assert len(items) == 0


def test_noop_dataloader__shuffle_error():
    """Test that shuffle=True raises an error with empty dataset"""
    cfg = Config.from_dict(
        {
            "dataset": {
                "name": "test_dataset",
            },
            "training": {
                "batch_size": 4,
                "shuffle": True,
            },
        }
    )

    dataset = NoopDataset(cfg=cfg)

    with pytest.raises(ValueError, match="num_samples should be a positive integer"):
        NoopDataLoader(cfg=cfg, dataset=dataset)
