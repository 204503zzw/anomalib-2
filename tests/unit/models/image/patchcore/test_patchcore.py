# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore model.

Covers the configurable feature pooling that controls how much local context is
averaged into each patch embedding, the configurable anomaly map blur, and the
optional FB-CLIP-inspired foreground-background disentanglement.
"""

import pytest
import torch
from torch import nn

from anomalib.models import Patchcore
from anomalib.models.image.patchcore.foreground import (
    BACKGROUND_VALUE,
    FOREGROUND_VALUE,
    compute_background_prototype,
    compute_soft_foreground_mask,
    enhance_semantic,
    enhance_spatial,
)
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


def _tokens(batch_size: int = 2, height: int = 8, width: int = 8, channels: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch_size, height * width, channels, generator=generator)


def test_soft_foreground_mask_values_and_shape() -> None:
    """The soft mask is binary over {0.5, 1.0} and matches the token layout."""
    tokens = _tokens()
    mask = compute_soft_foreground_mask(tokens, (8, 8))
    assert mask.shape == (2, 64)
    assert set(mask.unique().tolist()) <= {BACKGROUND_VALUE, FOREGROUND_VALUE}


def test_enhancement_views_preserve_shape() -> None:
    """SEM and SPA views keep the token tensor shape unchanged."""
    tokens = _tokens()
    mask = compute_soft_foreground_mask(tokens, (8, 8))
    assert enhance_semantic(tokens, mask).shape == tokens.shape
    assert enhance_spatial(tokens, mask, (8, 8)).shape == tokens.shape


def test_enhancement_views_handle_uniform_mask() -> None:
    """All-background and all-foreground masks produce finite outputs."""
    tokens = _tokens()
    for value in (BACKGROUND_VALUE, FOREGROUND_VALUE):
        mask = torch.full(tokens.shape[:2], value)
        assert torch.isfinite(enhance_semantic(tokens, mask)).all()
        assert torch.isfinite(enhance_spatial(tokens, mask, (8, 8))).all()


def test_background_prototype_shape() -> None:
    """The prototype is a single vector combining mean and max pooling."""
    background = torch.rand(10, 16)
    prototype = compute_background_prototype(background)
    assert prototype.shape == (16,)
    expected = 0.5 * background.mean(dim=0) + 0.5 * background.amax(dim=0)
    assert torch.allclose(prototype, expected)


def test_default_options_disable_disentanglement() -> None:
    """All FB-CLIP-inspired options are off by default."""
    model = _model()
    assert model.foreground_mask is False
    assert model.enhancement_views == ()
    assert model.background_suppression is False
    assert not model.uses_foreground_disentanglement
    assert "background_prototype" not in dict(model.named_buffers())


def test_invalid_enhancement_views() -> None:
    """Unknown view names are rejected."""
    with pytest.raises(ValueError, match="enhancement_views"):
        _model(enhancement_views=["id"])


def test_apply_enhancement_views_identity_only() -> None:
    """Without enabled views the tokens pass through unchanged."""
    model = _model()
    tokens = _tokens()
    mask = compute_soft_foreground_mask(tokens, (8, 8))
    assert model.apply_enhancement_views(tokens, mask, (8, 8)) is tokens


def test_apply_enhancement_views_averages_with_identity() -> None:
    """Enabled views are averaged together with the identity view."""
    model = _model(enhancement_views=["sem", "spa"])
    tokens = _tokens()
    mask = compute_soft_foreground_mask(tokens, (8, 8))
    fused = model.apply_enhancement_views(tokens, mask, (8, 8))
    expected = (tokens + enhance_semantic(tokens, mask) + enhance_spatial(tokens, mask, (8, 8))) / 3
    assert torch.allclose(fused, expected)


def test_forward_with_options_matches_default_shapes() -> None:
    """Enabled options keep the inference output shapes of the default model."""
    torch.manual_seed(0)
    images = torch.rand(2, 3, 64, 64)
    outputs = {}
    for name, kwargs in {
        "default": {},
        "fb": {"foreground_mask": True, "enhancement_views": ["sem", "spa"], "background_suppression": True},
    }.items():
        model = _model(**kwargs)
        model.train()
        model(images)
        model.subsample_embedding(sampling_ratio=0.5)
        model.eval()
        outputs[name] = model(images)
    assert outputs["fb"].pred_score.shape == outputs["default"].pred_score.shape
    assert outputs["fb"].anomaly_map.shape == outputs["default"].anomaly_map.shape
    assert torch.isfinite(outputs["fb"].anomaly_map).all()


def test_background_prototype_built_during_fit() -> None:
    """Fitting with background suppression populates the prototype buffer."""
    torch.manual_seed(0)
    model = _model(background_suppression=True)
    model.train()
    model(torch.rand(2, 3, 64, 64))
    model.subsample_embedding(sampling_ratio=0.5)
    assert model.background_prototype.numel() > 0
    assert "background_prototype" in dict(model.named_buffers())


def test_foreground_mask_downweights_background_scores() -> None:
    """Background patches are scaled by the soft mask at inference."""
    torch.manual_seed(0)
    images = torch.rand(2, 3, 64, 64)
    masked = _model(foreground_mask=True)
    masked.train()
    masked(images)
    masked.subsample_embedding(sampling_ratio=0.5)
    masked.eval()
    plain = _model()
    plain.memory_bank = masked.memory_bank
    plain.eval()
    masked_map = masked(images).anomaly_map
    plain_map = plain(images).anomaly_map
    assert masked_map.amax() <= plain_map.amax() + 1e-5


def test_lightning_model_forwards_fb_options() -> None:
    """The Lightning module passes the FB-CLIP-inspired options to the torch model."""
    model = Patchcore(
        backbone="resnet18",
        layers=["layer1"],
        pre_trained=False,
        foreground_mask=True,
        enhancement_views=["sem"],
        background_suppression=True,
    )
    assert model.model.foreground_mask is True
    assert model.model.enhancement_views == ("sem",)
    assert model.model.background_suppression is True
