# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore model.

Covers the configurable feature pooling that controls how much local context is
averaged into each patch embedding, and the configurable anomaly map blur.
"""

import pytest
import torch
from torch import nn

from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def _model(**kwargs) -> PatchcoreModel:
    return PatchcoreModel(backbone="resnet18", layers=["layer1"], pre_trained=False, **kwargs)


def test_default_feature_pooler() -> None:
    """Default configuration keeps the 3x3 average pooling."""
    model = _model()
    assert isinstance(model.feature_pooler, nn.AvgPool2d)
    assert model.feature_pooler.kernel_size == 3


def test_feature_pool_size_one_disables_pooling() -> None:
    """A pool size of one replaces the pooling with an identity."""
    model = _model(feature_pool_size=1)
    assert isinstance(model.feature_pooler, nn.Identity)


@pytest.mark.parametrize("feature_pool_size", [5, 7])
def test_feature_pool_size_sets_kernel_and_padding(feature_pool_size: int) -> None:
    """Larger pool sizes keep the feature map resolution unchanged."""
    model = _model(feature_pool_size=feature_pool_size)
    assert model.feature_pooler.kernel_size == feature_pool_size
    features = torch.rand(1, 8, 16, 16)
    assert model.feature_pooler(features).shape == features.shape


@pytest.mark.parametrize("feature_pool_size", [0, -1, 2, 4])
def test_invalid_feature_pool_size(feature_pool_size: int) -> None:
    """Non-positive and even pool sizes are rejected."""
    with pytest.raises(ValueError, match="feature_pool_size"):
        _model(feature_pool_size=feature_pool_size)


def test_unpooled_embedding_preserves_local_detail() -> None:
    """Disabling pooling keeps single-cell feature spikes in the embedding."""
    pooled = _model()
    unpooled = _model(feature_pool_size=1)
    features = {"layer1": torch.zeros(1, 4, 8, 8)}
    features["layer1"][0, :, 4, 4] = 1.0

    def embed(model: PatchcoreModel) -> torch.Tensor:
        maps = {layer: model.feature_pooler(feature) for layer, feature in features.items()}
        return model.reshape_embedding(model.generate_embedding(maps))

    assert embed(unpooled).amax() == pytest.approx(1.0)
    assert embed(pooled).amax() == pytest.approx(1.0 / 9)


def test_lightning_model_forwards_feature_pool_size() -> None:
    """The Lightning module passes the pool size to the torch model."""
    model = Patchcore(backbone="resnet18", layers=["layer1"], pre_trained=False, feature_pool_size=1)
    assert isinstance(model.model.feature_pooler, nn.Identity)


def test_default_blur_sigma() -> None:
    """Default configuration keeps the sigma of four used by the paper implementation."""
    model = _model()
    assert model.blur_sigma == 4
    assert model.anomaly_map_generator.blur.kernel.shape[-2:] == (33, 33)


@pytest.mark.parametrize("blur_sigma", [1, 2])
def test_blur_sigma_sets_kernel_size(blur_sigma: int) -> None:
    """The blur kernel is derived from the requested sigma."""
    model = _model(blur_sigma=blur_sigma)
    expected = 2 * int(4.0 * blur_sigma + 0.5) + 1
    assert model.anomaly_map_generator.blur.kernel.shape[-2:] == (expected, expected)


@pytest.mark.parametrize("blur_sigma", [0, -1])
def test_invalid_blur_sigma(blur_sigma: int) -> None:
    """Non-positive sigmas are rejected."""
    with pytest.raises(ValueError, match="blur_sigma"):
        _model(blur_sigma=blur_sigma)


def test_small_blur_sigma_keeps_peak() -> None:
    """A small sigma retains more of a single-pixel anomaly peak than the default."""
    patch_scores = torch.zeros(1, 1, 32, 32)
    patch_scores[0, 0, 16, 16] = 1.0
    sharp = _model(blur_sigma=1).anomaly_map_generator(patch_scores)
    smooth = _model().anomaly_map_generator(patch_scores)
    assert sharp.amax() > smooth.amax()


def test_lightning_model_forwards_blur_sigma() -> None:
    """The Lightning module passes the blur sigma to the torch model."""
    model = Patchcore(backbone="resnet18", layers=["layer1"], pre_trained=False, blur_sigma=1)
    assert model.model.blur_sigma == 1
    assert model.model.anomaly_map_generator.blur.kernel.shape[-2:] == (9, 9)
