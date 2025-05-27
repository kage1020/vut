import pytest
import torch

from vut.cuda import get_device
from vut.models.i3d import I3D, InceptionModule, MaxPool3dSamePadding, Unit3D


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


def test_i3d__custom_num_classes():
    device = get_device()
    model = I3D(num_classes=10)
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    output = model(input_tensor)

    assert output.shape == (1, 10, 1)


def test_i3d__spatial_squeeze_false():
    device = get_device()
    model = I3D(spatial_squeeze=False)
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    output = model(input_tensor)

    assert output.shape == (1, 400, 1, 1, 1)


def test_i3d__different_endpoints():
    endpoints = ["Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Mixed_3b"]

    for endpoint in endpoints:
        model = I3D(final_endpoint=endpoint)

        assert model._final_endpoint == endpoint
        assert endpoint in model.end_points


def test_i3d__invalid_endpoint():
    with pytest.raises(AssertionError, match="Unknown final endpoint"):
        I3D(final_endpoint="invalid_endpoint")


def test_unit3d():
    device = get_device()
    unit = Unit3D(in_channels=3, out_channels=64, kernel_shape=(3, 3, 3))
    unit = unit.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    output = unit(input_tensor)

    assert output.shape == (1, 64, 16, 224, 224)


def test_unit3d__no_batch_norm():
    device = get_device()
    unit = Unit3D(
        in_channels=3, out_channels=64, kernel_shape=(3, 3, 3), use_batch_norm=False
    )
    unit = unit.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    output = unit(input_tensor)

    assert output.shape == (1, 64, 16, 224, 224)


def test_unit3d__no_activation():
    device = get_device()
    unit = Unit3D(
        in_channels=3, out_channels=64, kernel_shape=(3, 3, 3), activation_fn=None
    )
    unit = unit.to(device)
    input_tensor = torch.randn(1, 3, 16, 224, 224).to(device)

    output = unit(input_tensor)

    assert output.shape == (1, 64, 16, 224, 224)


def test_maxpool3d_same_padding():
    device = get_device()
    pool = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
    pool = pool.to(device)
    input_tensor = torch.randn(1, 64, 16, 224, 224).to(device)

    output = pool(input_tensor)

    assert output.shape == (1, 64, 8, 112, 112)


def test_maxpool3d_same_padding__different_sizes():
    device = get_device()
    pool = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0)
    pool = pool.to(device)

    input_tensor = torch.randn(1, 64, 15, 223, 223).to(device)
    output = pool(input_tensor)
    assert output.shape == (1, 64, 15, 223, 223)

    input_tensor = torch.randn(1, 64, 16, 224, 224).to(device)
    output = pool(input_tensor)
    assert output.shape == (1, 64, 16, 224, 224)


def test_inception_module():
    device = get_device()
    inception = InceptionModule(in_channels=192, out_channels=[64, 96, 128, 16, 32, 32])
    inception = inception.to(device)
    input_tensor = torch.randn(1, 192, 16, 224, 224).to(device)

    output = inception(input_tensor)

    expected_channels = 64 + 128 + 32 + 32
    assert output.shape == (1, expected_channels, 16, 224, 224)


def test_inception_module__different_channels():
    device = get_device()
    inception = InceptionModule(
        in_channels=480, out_channels=[192, 96, 208, 16, 48, 64]
    )
    inception = inception.to(device)
    input_tensor = torch.randn(1, 480, 8, 112, 112).to(device)

    output = inception(input_tensor)

    expected_channels = 192 + 208 + 48 + 64
    assert output.shape == (1, expected_channels, 8, 112, 112)
