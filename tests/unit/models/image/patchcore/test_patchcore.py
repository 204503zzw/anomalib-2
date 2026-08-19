# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore model.

Covers the configurable feature pooling that controls how much local context is
averaged into each patch embedding.
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
