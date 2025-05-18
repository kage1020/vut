import torch

from vut.cuda import get_device
from vut.models.i3d import I3D


def test_i3d__forwarding():
    device = get_device()
    model = I3D()
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    output = model(input_tensor)

    assert output.shape == (1, 400, 1)


def test_i3d__extract_features():
    device = get_device()
    model = I3D()
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    features = model.extract_features(input_tensor)

    assert features.shape == (1, 1024, 1, 1, 1)
