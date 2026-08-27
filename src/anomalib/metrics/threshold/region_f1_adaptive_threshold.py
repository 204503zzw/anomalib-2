# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Region-level F1 adaptive threshold metric for anomaly detection.

This module provides the ``RegionF1AdaptiveThreshold`` class, which selects the
threshold that maximizes a *region-level* (connected-component) F1 score instead
of the pixel-level F1 score used by
:class:`~anomalib.metrics.threshold.F1AdaptiveThreshold`.

For every candidate threshold the anomaly maps are binarized and:

- a ground-truth region counts as detected (region-level true positive) when the
  fraction of its pixels covered by the prediction exceeds ``overlap_ratio``,
  which defines region-level recall;
- a predicted region counts as correct when the fraction of its pixels falling
  inside any ground-truth region exceeds ``overlap_ratio``, which defines
  region-level precision. This penalizes over-segmentation: a single blob that
  covers the whole image overlaps every ground-truth region but is mostly
  background, so it is not counted as correct.

Region-level scoring weights every defect equally, regardless of its size, so
the resulting threshold is not dominated by a few large anomalies and penalizes
each spurious blob once rather than per pixel.

Example:
    >>> import torch
    >>> from anomalib.metrics.threshold.region_f1_adaptive_threshold import (
    ...     _RegionF1AdaptiveThreshold,
    ... )
    >>> anomaly_map = torch.zeros(1, 8, 8)
    >>> anomaly_map[0, 2:4, 2:4] = 0.9
    >>> gt_mask = torch.zeros(1, 8, 8)
    >>> gt_mask[0, 2:4, 2:4] = 1
    >>> threshold = _RegionF1AdaptiveThreshold(num_thresholds=10)
    >>> threshold.update(anomaly_map, gt_mask)
    >>> bool(threshold.compute() < 0.9)
    True

Note:
    Computing the region-level curve requires a connected-component labeling per
    candidate threshold, so it is considerably slower than the pixel-level
    variant. Use ``num_thresholds`` to trade granularity for speed.
"""

import logging

import cv2
import numpy as np
import torch
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.metrics.base import AnomalibMetric

from .base import Threshold

logger = logging.getLogger(__name__)


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label the connected components of a binary mask.

    Args:
        mask (np.ndarray): Binary mask of shape ``(H, W)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Integer label map of shape ``(H, W)`` in which the background is ``0`` and the
            regions are labeled from ``1`` to ``N``, and the pixel area of each label as an array of length ``N + 1``
            (index ``0`` holds the background area).
    """
    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    areas = np.bincount(labels.ravel())
    return labels, areas


def _count_matched_regions(
    labels: np.ndarray,
    areas: np.ndarray,
    other_mask: np.ndarray,
    overlap_ratio: float,
) -> int:
    """Count the regions of a label map that are sufficiently covered by another mask.

    Args:
        labels (np.ndarray): Integer label map of shape ``(H, W)`` with background ``0``.
        areas (np.ndarray): Pixel area of each label, as returned by :func:`_label_regions`.
        other_mask (np.ndarray): Binary mask of shape ``(H, W)`` to intersect with the labeled regions.
        overlap_ratio (float): Minimum fraction of a region's pixels that must be covered by ``other_mask`` for the
            region to count as matched. A value of ``0.0`` means any overlap suffices.

    Returns:
        int: Number of matched regions.
    """
    num_regions = len(areas) - 1
    if num_regions < 1:
        return 0
    overlap = np.bincount(labels[other_mask].ravel(), minlength=len(areas))[1:]
    return int((overlap > overlap_ratio * areas[1:]).sum())


class _RegionF1AdaptiveThreshold(Threshold):
    """Adaptive threshold that maximizes the region-level F1 score.

    The metric accumulates anomaly maps and ground-truth masks during validation
    and sweeps ``num_thresholds`` candidate thresholds between the smallest and
    largest observed anomaly score. At each candidate it computes region-level
    precision and recall by matching predicted connected components against
    ground-truth connected components, and finally returns the threshold with the
    highest region-level F1 score.

    Args:
        num_thresholds (int): Number of candidate thresholds in the sweep. Defaults to ``50``.
        overlap_ratio (float): Minimum fraction of a region's pixels that must overlap with the other mask for the
            region to count as matched. ``0.0`` means any overlap suffices, which makes the metric blind to
            over-segmentation. Defaults to ``0.5``.
        **kwargs: Additional keyword arguments passed to the ``torchmetrics.Metric`` base class.

    Raises:
        ValueError: If ``num_thresholds`` is smaller than ``2`` or ``overlap_ratio`` is outside ``[0, 1)``.

    Example:
        >>> import torch
        >>> threshold = _RegionF1AdaptiveThreshold(num_thresholds=10)
        >>> anomaly_map = torch.rand(2, 16, 16)
        >>> gt_mask = torch.zeros(2, 16, 16)
        >>> gt_mask[0, 4:8, 4:8] = 1
        >>> threshold.update(anomaly_map, gt_mask)
        >>> value = threshold.compute()
    """

    preds: list[torch.Tensor]
    target: list[torch.Tensor]

    def __init__(self, num_thresholds: int = 50, overlap_ratio: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)

        if num_thresholds < 2:
            msg = f"num_thresholds must be at least 2, got {num_thresholds}."
            raise ValueError(msg)
        if not 0.0 <= overlap_ratio < 1.0:
            msg = f"overlap_ratio must be in the range [0, 1), got {overlap_ratio}."
            raise ValueError(msg)

        self.num_thresholds = num_thresholds
        self.overlap_ratio = overlap_ratio

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric state with a batch of anomaly maps and ground-truth masks.

        Args:
            preds (torch.Tensor): Anomaly maps of shape ``(B, H, W)`` or ``(B, 1, H, W)``.
            target (torch.Tensor): Ground-truth masks of shape ``(B, H, W)`` or ``(B, 1, H, W)``.
        """
        self.preds.append(preds.detach().flatten(end_dim=-3) if preds.dim() > 3 else preds.detach())
        self.target.append(target.detach().flatten(end_dim=-3) if target.dim() > 3 else target.detach())

    def compute(self) -> torch.Tensor:
        """Compute the threshold that maximizes the region-level F1 score.

        Returns:
            torch.Tensor: Optimal threshold value. When several candidates reach the same maximum F1 score, the
                smallest of them is returned.

        Warning:
            If the ground-truth masks do not contain any anomalous region, the threshold defaults to the highest
            observed anomaly score so that no pixel is flagged as anomalous.
        """
        preds = dim_zero_cat(self.preds).float().cpu()
        target = dim_zero_cat(self.target).cpu() > 0.5

        gt_regions = [_label_regions(mask.numpy()) for mask in target]
        num_gt_regions = sum(len(areas) - 1 for _, areas in gt_regions)

        if num_gt_regions == 0:
            msg = (
                "The validation set does not contain any anomalous regions. As a result, the region-level adaptive "
                "threshold will take the value of the highest anomaly score observed in the normal validation images, "
                "which may lead to poor predictions. For a more reliable adaptive threshold computation, please add "
                "some anomalous images to the validation set."
            )
            logger.warning(msg)
            return preds.max()

        candidates = torch.linspace(preds.min().item(), preds.max().item(), self.num_thresholds)
        f1_scores = torch.zeros(self.num_thresholds)
        preds_numpy = preds.numpy()

        for idx, candidate in enumerate(candidates):
            num_pred_regions = 0
            num_detected_gt = 0
            num_correct_pred = 0
            for anomaly_map, (gt_labels, gt_areas) in zip(preds_numpy, gt_regions, strict=True):
                pred_labels, pred_areas = _label_regions(anomaly_map > candidate.item())
                num_pred_regions += len(pred_areas) - 1
                if len(gt_areas) > 1:
                    num_detected_gt += _count_matched_regions(
                        gt_labels,
                        gt_areas,
                        pred_labels > 0,
                        self.overlap_ratio,
                    )
                num_correct_pred += _count_matched_regions(
                    pred_labels,
                    pred_areas,
                    gt_labels > 0,
                    self.overlap_ratio,
                )

            recall = num_detected_gt / num_gt_regions
            precision = num_correct_pred / num_pred_regions if num_pred_regions > 0 else 0.0
            f1_scores[idx] = 2 * precision * recall / (precision + recall + 1e-10)

        return candidates[int(torch.argmax(f1_scores))]


class RegionF1AdaptiveThreshold(AnomalibMetric, _RegionF1AdaptiveThreshold):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to the RegionF1AdaptiveThreshold metric."""
