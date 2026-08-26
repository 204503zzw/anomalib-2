# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training-free foreground-background disentanglement utilities for PatchCore.

This module adapts the training-free components of FB-CLIP (Hu et al., CVPR 2026,
"FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background
Disentanglement") to PatchCore's memory-bank pipeline:

- a soft foreground mask computed from local saliency, center distance, and
  inconsistency with the global feature;
- a semantic (SEM) enhancement view that aggregates foreground tokens by
  information richness and background tokens by stability;
- a spatial (SPA) enhancement view that aggregates 5x5 neighborhoods with
  foreground/background-aware weights;
- a background prototype that summarizes background patches for score
  suppression at inference time.

All operations are deterministic and contain no learnable parameters, so the
memory bank and query embeddings stay in the same feature space as long as the
same options are used during fitting and inference.
"""

import torch
from torch.nn import functional as F  # noqa: N812

FOREGROUND_VALUE = 1.0
"""Soft-mask value assigned to high-confidence foreground tokens."""

BACKGROUND_VALUE = 0.5
"""Soft-mask value assigned to uncertain/background tokens."""

SEM_RESIDUAL_ALPHA = 0.6
"""Residual blend factor of the semantic view (fraction of the enhanced tokens)."""

SPA_KERNEL_SIZE = 5
"""Neighborhood size of the spatial view."""

BACKGROUND_SUPPRESSION_BLEND = 0.5
"""Fraction of the patch score that is scaled by background dissimilarity."""


def _min_max_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize values to ``[0, 1]`` per sample along the last dimension."""
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    return (values - minimum) / (maximum - minimum + eps)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax over ``dim`` restricted to ``mask``; fully masked slices become zeros."""
    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim)
    return torch.nan_to_num(weights, nan=0.0)


def compute_soft_foreground_mask(tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Compute a soft foreground mask from training-free cues.

    Three complementary cues are combined with equal weights: local saliency
    (deviation from the 3x3 neighborhood mean), center distance (industrial
    images tend to center the object), and inconsistency with the global mean
    feature (the CLS-token substitute for CNN backbones). Each cue is min-max
    normalized per image and the average is binarized into a soft mask with
    values ``{0.5, 1.0}`` so that background tokens are down-weighted rather
    than discarded.

    Args:
        tokens (torch.Tensor): Patch embeddings of shape ``(B, L, C)`` where
            ``L == grid_size[0] * grid_size[1]``.
        grid_size (tuple[int, int]): Spatial layout ``(height, width)`` of the tokens.

    Returns:
        torch.Tensor: Soft foreground mask of shape ``(B, L)`` with values
            ``1.0`` (foreground) or ``0.5`` (uncertain/background).
    """
    batch_size, num_tokens, channels = tokens.shape
    height, width = grid_size

    global_feature = tokens.mean(dim=1, keepdim=True)
    inconsistency = 1 - F.cosine_similarity(tokens, global_feature, dim=-1)

    grid = tokens.permute(0, 2, 1).reshape(batch_size, channels, height, width)
    local_mean = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
    saliency = (1 - F.cosine_similarity(grid, local_mean, dim=1)).reshape(batch_size, num_tokens)

    ys = torch.linspace(-1.0, 1.0, height, device=tokens.device)
    xs = torch.linspace(-1.0, 1.0, width, device=tokens.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    center_closeness = 1 - torch.sqrt(grid_y**2 + grid_x**2).flatten() / (2**0.5)
    center_closeness = center_closeness.unsqueeze(0).expand(batch_size, -1)

    score = (
        _min_max_normalize(saliency) + _min_max_normalize(center_closeness) + _min_max_normalize(inconsistency)
    ) / 3
    return torch.where(
        score > 0.5,
        torch.full_like(score, FOREGROUND_VALUE),
        torch.full_like(score, BACKGROUND_VALUE),
    )


def enhance_semantic(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply the semantic (SEM) enhancement view.

    Foreground tokens attend to other foreground tokens weighted by their
    information richness (cosine deviation from the global feature), while
    background tokens attend to other background tokens weighted by their
    stability (cosine similarity to the global feature). The aggregated
    features are blended with the original tokens through a fixed residual.

    Args:
        tokens (torch.Tensor): Patch embeddings of shape ``(B, L, C)``.
        mask (torch.Tensor): Soft foreground mask of shape ``(B, L)``.

    Returns:
        torch.Tensor: Enhanced embeddings of shape ``(B, L, C)``.
    """
    global_feature = tokens.mean(dim=1, keepdim=True)
    similarity = F.cosine_similarity(tokens, global_feature, dim=-1)
    richness = 1 - similarity

    foreground_prob = mask
    background_prob = 1 - mask

    foreground_weights = foreground_prob.unsqueeze(2) * foreground_prob.unsqueeze(1)
    background_weights = background_prob.unsqueeze(2) * background_prob.unsqueeze(1)

    foreground_attention = _masked_softmax(
        richness.unsqueeze(1) * foreground_weights,
        foreground_weights > 0,
        dim=-1,
    )
    background_attention = _masked_softmax(
        similarity.unsqueeze(1) * background_weights,
        background_weights > 0,
        dim=-1,
    )

    foreground_agg = foreground_attention @ tokens
    background_agg = background_attention @ tokens
    mixed = foreground_prob.unsqueeze(-1) * foreground_agg + background_prob.unsqueeze(-1) * background_agg
    return SEM_RESIDUAL_ALPHA * mixed + (1 - SEM_RESIDUAL_ALPHA) * tokens


def enhance_spatial(tokens: torch.Tensor, mask: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Apply the spatial (SPA) enhancement view.

    Each location aggregates its 5x5 neighborhood. Foreground neighbors are
    weighted by information richness (deviation from the global feature times
    deviation from the neighborhood mean) and background neighbors by stability
    (similarity to the global feature times feature magnitude). The two branch
    outputs are combined proportionally to the number of valid neighbors of
    each type so the feature magnitude stays comparable to the input.

    Args:
        tokens (torch.Tensor): Patch embeddings of shape ``(B, L, C)``.
        mask (torch.Tensor): Soft foreground mask of shape ``(B, L)``.
        grid_size (tuple[int, int]): Spatial layout ``(height, width)`` of the tokens.

    Returns:
        torch.Tensor: Enhanced embeddings of shape ``(B, L, C)``.
    """
    batch_size, num_tokens, channels = tokens.shape
    height, width = grid_size
    padding = SPA_KERNEL_SIZE // 2
    window = SPA_KERNEL_SIZE**2

    grid = tokens.permute(0, 2, 1).reshape(batch_size, channels, height, width)
    patches = F.unfold(grid, kernel_size=SPA_KERNEL_SIZE, padding=padding)
    patches = patches.reshape(batch_size, channels, window, num_tokens)

    mask_grid = mask.reshape(batch_size, 1, height, width)
    mask_patches = F.unfold(mask_grid, kernel_size=SPA_KERNEL_SIZE, padding=padding)
    mask_patches = mask_patches.reshape(batch_size, window, num_tokens)
    valid = mask_patches > 0

    global_feature = tokens.mean(dim=1).reshape(batch_size, channels, 1, 1)
    similarity = F.cosine_similarity(patches, global_feature, dim=1)
    neighborhood_mean = patches.mean(dim=2, keepdim=True)

    richness = (1 - similarity) * (patches - neighborhood_mean).norm(dim=1)
    stability = similarity * patches.norm(dim=1)

    is_foreground = valid & (mask_patches == FOREGROUND_VALUE)
    is_background = valid & (mask_patches == BACKGROUND_VALUE)

    foreground_weights = _masked_softmax(richness, is_foreground, dim=1)
    background_weights = _masked_softmax(stability, is_background, dim=1)

    foreground_agg = (patches * foreground_weights.unsqueeze(1)).sum(dim=2)
    background_agg = (patches * background_weights.unsqueeze(1)).sum(dim=2)

    valid_counts = valid.sum(dim=1, keepdim=False).clamp_min(1)
    foreground_fraction = (is_foreground.sum(dim=1) / valid_counts).unsqueeze(1)
    background_fraction = (is_background.sum(dim=1) / valid_counts).unsqueeze(1)

    aggregated = foreground_fraction * foreground_agg + background_fraction * background_agg
    return aggregated.permute(0, 2, 1)


def compute_background_prototype(background_tokens: torch.Tensor) -> torch.Tensor:
    """Summarize background tokens into a single prototype vector.

    Combines global averaging with max-pooling so that both the common
    background appearance and its most salient characteristics are preserved.

    Args:
        background_tokens (torch.Tensor): Background candidate embeddings of
            shape ``(N, C)``.

    Returns:
        torch.Tensor: Background prototype of shape ``(C,)``.
    """
    return 0.5 * background_tokens.mean(dim=0) + 0.5 * background_tokens.amax(dim=0)
