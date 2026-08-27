# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the region-level adaptive threshold metric."""

import logging

import pytest
import torch

from anomalib.metrics.threshold.region_f1_adaptive_threshold import _RegionF1AdaptiveThreshold


def make_maps(num_maps: int = 2, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Create anomaly maps with one large and one small anomalous region.

    The background contains low-level noise, the large region has a high anomaly
    score and the small region a moderate one.

    Args:
        num_maps (int): Number of maps in the batch. Defaults to 2.
        size (int): Spatial size of the maps. Defaults to 32.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Anomaly maps and ground-truth masks of shape ``(num_maps, size, size)``.
    """
    generator = torch.Generator().manual_seed(0)
    anomaly_maps = torch.rand(num_maps, size, size, generator=generator) * 0.1
    gt_masks = torch.zeros(num_maps, size, size)
    anomaly_maps[0, 4:20, 4:20] = 0.9
    gt_masks[0, 4:20, 4:20] = 1
    anomaly_maps[1, 2:5, 2:5] = 0.5
    gt_masks[1, 2:5, 2:5] = 1
    return anomaly_maps, gt_masks


class TestRegionF1AdaptiveThreshold:
    """Test the region-level F1 adaptive threshold."""

    @staticmethod
    @pytest.mark.parametrize("channel_dim", [False, True])
    def test_threshold_separates_all_regions(channel_dim: bool) -> None:
        """Test that the threshold suppresses noise while keeping both regions."""
        anomaly_maps, gt_masks = make_maps()
        if channel_dim:
            anomaly_maps = anomaly_maps.unsqueeze(1)
            gt_masks = gt_masks.unsqueeze(1)

        metric = _RegionF1AdaptiveThreshold(num_thresholds=20)
        metric.update(anomaly_maps, gt_masks)
        threshold = metric.compute()

        # noise stays below the threshold, both anomalous regions above it
        assert threshold.item() > 0.1
        assert threshold.item() < 0.5

    @staticmethod
    def test_small_region_is_not_ignored() -> None:
        """Test that region-level scoring does not sacrifice a small region for pixel accuracy."""
        anomaly_maps, gt_masks = make_maps()
        metric = _RegionF1AdaptiveThreshold(num_thresholds=20)
        metric.update(anomaly_maps, gt_masks)
        threshold = metric.compute()

        pred_masks = anomaly_maps > threshold
        assert bool((pred_masks[1] & (gt_masks[1] > 0.5)).any())

    @staticmethod
    def test_overlap_ratio_penalizes_over_segmentation() -> None:
        """Test that the default overlap ratio rejects a blob covering the whole image."""
        anomaly_maps, gt_masks = make_maps()

        strict = _RegionF1AdaptiveThreshold(num_thresholds=20, overlap_ratio=0.5)
        strict.update(anomaly_maps, gt_masks)
        lenient = _RegionF1AdaptiveThreshold(num_thresholds=20, overlap_ratio=0.0)
        lenient.update(anomaly_maps, gt_masks)

        # with any-overlap matching, the noise-dominated blob at the lowest candidate
        # already reaches perfect region scores, so the threshold collapses to the minimum
        assert lenient.compute().item() < 0.1
        assert strict.compute().item() > 0.1

    @staticmethod
    def test_no_anomalous_regions_warning(caplog: pytest.LogCaptureFixture) -> None:
        """Test that the maximum score is used when no anomalous region is present."""
        anomaly_maps = torch.rand(2, 8, 8)
        gt_masks = torch.zeros(2, 8, 8)

        metric = _RegionF1AdaptiveThreshold(num_thresholds=5)
        metric.update(anomaly_maps, gt_masks)

        with caplog.at_level(logging.WARNING):
            threshold = metric.compute()

        assert "does not contain any anomalous regions" in caplog.text
        assert threshold.item() == pytest.approx(anomaly_maps.max().item())

    @staticmethod
    @pytest.mark.parametrize(
        ("num_thresholds", "overlap_ratio"),
        [(1, 0.0), (10, 1.0), (10, -0.1)],
    )
    def test_invalid_arguments(num_thresholds: int, overlap_ratio: float) -> None:
        """Test that invalid constructor arguments are rejected."""
        with pytest.raises(ValueError, match=r"num_thresholds|overlap_ratio"):
            _RegionF1AdaptiveThreshold(num_thresholds=num_thresholds, overlap_ratio=overlap_ratio)
