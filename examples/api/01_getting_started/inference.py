# ruff: noqa
import sys
import os
from pathlib import Path
import io
import builtins

# 强制标准输出/输入使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 强制 open() 默认使用 UTF-8 编码（解决 GBK 解码问题）
original_open = builtins.open


def utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
    return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)


builtins.open = utf8_open  # type: ignore[method-assign]

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import importlib
import importlib.abc
import importlib.util
from types import ModuleType
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
from jsonargparse import ArgumentParser, ActionConfigFile
from torch.utils.data import DataLoader
from torchmetrics import Metric
from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import get_model


# ---------- 旧版权重兼容 ----------
class _LegacyMetricStub(Metric):
    """占位 metric：仅用于反序列化旧 ckpt 里已不存在的 metric 类，不参与任何计算。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def update(self, *args, **kwargs) -> None:
        return None

    def compute(self) -> torch.Tensor:
        return torch.tensor(float("nan"))


def _build_legacy_metric_module(fullname: str) -> ModuleType:
    """构造一个占位模块：属性优先取 anomalib.metrics 里的同名类，否则生成占位 metric。"""
    module = ModuleType(fullname)

    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        metrics = importlib.import_module("anomalib.metrics")
        attr = getattr(metrics, name, None)
        if not isinstance(attr, type):
            attr = type(name, (_LegacyMetricStub,), {"__module__": fullname})
            print(f">>> 兼容旧权重：{fullname}.{name} 已不存在，使用占位 metric")
        setattr(module, name, attr)
        return attr

    module.__getattr__ = __getattr__  # type: ignore[method-assign]
    return module


class _LegacyMetricFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """为已被重命名/删除的 anomalib.metrics 子模块提供占位实现。

    ckpt 会 pickle 训练时评估器里的 metric 类；换 anomalib 版本后这些模块可能已经消失
    （例如 anomalib.metrics.false_negatives_positives 现在叫 anomalib.metrics.pg_pb），
    torch.load 会因此抛 ModuleNotFoundError。该 finder 挂在 sys.meta_path 末尾，
    只有正常导入失败时才生效，因此不会影响真实模块的加载。
    """

    prefix = "anomalib.metrics."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        print(f">>> 兼容旧权重：模块 {spec.name} 已不存在，使用占位模块")
        return _build_legacy_metric_module(spec.name)

    def exec_module(self, module) -> None:
        return None


def install_legacy_metric_compat() -> None:
    """安装旧 ckpt 的 metric 模块兼容层（重复调用无副作用）。"""
    if any(isinstance(finder, _LegacyMetricFinder) for finder in sys.meta_path):
        return
    sys.meta_path.append(_LegacyMetricFinder())


# ---------- 工具函数 ----------
def safe_imread(path: str) -> np.ndarray:
    path = str(path)
    img_array = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def safe_imwrite(path: str, img: np.ndarray) -> bool:
    path = str(path)
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(path)
        return True
    return False


def save_heatmap(anomaly_map, save_path):
    amap = anomaly_map.squeeze().cpu().numpy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heatmap = (amap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    safe_imwrite(str(save_path), heatmap)


def save_overlay(image, anomaly_map, save_path):
    img = image.cpu().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    amap = anomaly_map.squeeze().cpu().numpy()
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    amap = (amap * 255).astype(np.uint8)
    if amap.shape[:2] != img.shape[:2]:
        amap = cv2.resize(amap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    safe_imwrite(str(save_path), overlay)


def build_post_processor(pixel_sensitivity=None, image_sensitivity=None):
    """按需构建 OneClassPostProcessor。

    sensitivity 作用在归一化后的 [0, 1] 尺度上，阈值 = 1 - sensitivity。
    影响 anomalib 自己产出的 pred_mask / pred_label（即四联图里的红圈）。
    """
    if pixel_sensitivity is None and image_sensitivity is None:
        return None
    # 新版 anomalib 叫 PostProcessor，旧版叫 OneClassPostProcessor，参数相同
    try:
        from anomalib.post_processing import PostProcessor as _PP
    except ImportError:
        from anomalib.post_processing import OneClassPostProcessor as _PP
    kwargs = {}
    if pixel_sensitivity is not None:
        kwargs["pixel_sensitivity"] = pixel_sensitivity
    if image_sensitivity is not None:
        kwargs["image_sensitivity"] = image_sensitivity
    print(f">>> 使用 post_processor sensitivity: {kwargs}")
    return _PP(**kwargs)


def build_model_from_config(model_cfg: dict, image_size=None, post_processor=None):
    """根据配置构建模型。

    image_size: 训练时的输入尺寸 [height, width]。必须与训练脚本一致，
    否则会退回模型默认的 Resize（PatchCore 为 256x256），推理指标会变差。
    post_processor: 可选的 OneClassPostProcessor，用于覆盖默认 0.5 阈值。
    """
    if "class_path" in model_cfg:
        import importlib

        class_path = model_cfg["class_path"]
        init_args = dict(model_cfg.get("init_args", {}))
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        if image_size is not None and "pre_processor" not in init_args:
            init_args["pre_processor"] = model_class.configure_pre_processor(
                image_size=tuple(image_size),
            )
            print(f">>> 使用输入尺寸 (h, w) = {tuple(image_size)}")
        if post_processor is not None and "post_processor" not in init_args:
            init_args["post_processor"] = post_processor
        return model_class(**init_args)
    else:
        model = get_model(model_cfg)
        if image_size is not None:
            model.pre_processor = type(model).configure_pre_processor(image_size=tuple(image_size))
            print(f">>> 使用输入尺寸 (h, w) = {tuple(image_size)}")
        if post_processor is not None:
            model.post_processor = post_processor
        return model


def build_transform(transform_cfg: dict):
    import importlib

    class_path = transform_cfg["class_path"]
    init_args = transform_cfg.get("init_args", {})
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if "transforms" in init_args:
        init_args["transforms"] = [build_transform(t) for t in init_args["transforms"]]
    if "size" in init_args and isinstance(init_args["size"], list):
        init_args["size"] = tuple(init_args["size"])
    return cls(**init_args)


# ---------- 像素级评估函数 ----------
def load_gt_mask(gt_path, target_shape=None):
    gt = safe_imread(str(gt_path))
    if gt is None:
        raise FileNotFoundError(f"无法读取GT: {gt_path}")
    if len(gt.shape) == 3:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
    mask = (binary > 0).astype(np.uint8)
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def compute_pixel_metrics(pred_mask, gt_mask, eps=1e-8):
    tp = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    fp = np.logical_and(pred_mask > 0, gt_mask == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_mask > 0).sum()
    tn = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
    miss_rate = fn / (fn + tp + eps)
    false_alarm = fp / (fp + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "miss_rate": miss_rate,
        "false_alarm": false_alarm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def get_batch_gt_label(pred, index):
    """从预测批次里取出图像级 GT 标签，没有则回退到 GT 掩膜，都没有返回 None。"""
    gt_label = getattr(pred, "gt_label", None)
    if gt_label is not None:
        value = gt_label[index] if gt_label.dim() > 0 else gt_label
        return int(value.cpu().item())
    mask = get_batch_gt_mask(pred, index)
    return None if mask is None else int(mask.any())


def get_batch_gt_mask(pred, index):
    """从预测批次里取出 GT 掩膜（测试模式下由数据集提供），没有则返回 None。"""
    gt_mask = getattr(pred, "gt_mask", None)
    if gt_mask is None:
        return None
    mask = gt_mask[index] if gt_mask.dim() > 2 else gt_mask
    return mask.squeeze().cpu().numpy().astype(np.uint8)


def get_batch_pred_mask(pred, index):
    """从预测批次里取出 anomalib 生成的预测掩膜，没有则返回 None。"""
    pred_mask = getattr(pred, "pred_mask", None)
    if pred_mask is None:
        return None
    mask = pred_mask[index] if pred_mask.dim() > 2 else pred_mask
    return mask.squeeze().cpu().numpy().astype(np.uint8)


def get_eval_pred_mask(pred, index, anomaly_map, pixel_threshold):
    """优先使用 anomalib 生成的预测掩膜，否则按回退阈值二值化 anomaly_map。"""
    pred_mask = get_batch_pred_mask(pred, index)
    if pred_mask is not None:
        return pred_mask
    if anomaly_map is None:
        return None
    amap_np = anomaly_map.squeeze().cpu().numpy()
    return (amap_np > pixel_threshold).astype(np.uint8)


# ---------- 区域级评估函数 ----------
def _filter_small_regions(num_labels, labels, min_area):
    """返回面积不小于 min_area 的连通域编号列表。"""
    return [i for i in range(1, num_labels) if (labels == i).sum() >= min_area]


def _region_bbox(region):
    """返回连通域的外接框 (x, y, w, h)。"""
    ys, xs = np.nonzero(region)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def compute_region_metrics(pred_mask, gt_mask, coverage_threshold=0.6, min_area=0, overlap_ratio_threshold=0.01):
    """区域级（每个缺陷）评估：按 GT 缺陷被预测覆盖的像素比例判定。

    覆盖率 = 单个预测区域与该 GT 区域的交集像素数 / GT 区域面积（多个预测区域不累加，
    取覆盖率最高的那个），>= coverage_threshold（默认 60%）记为检出，否则记为漏检；
    单独把某个 GT 覆盖到该比例的预测区域算命中，其余预测区域记为误报。
    IoU 仅作为明细里的参考值。

    覆盖率不足的 GT 区域按覆盖率再拆分：>= overlap_ratio_threshold（默认 1%，即
    1%~60%）记为「漏检-与预测有交集」，< overlap_ratio_threshold（0%~1%）记为
    「漏检-与预测无交集」。

    误报区域按落入 GT 的像素比例拆分：>= overlap_ratio_threshold 的计入
    ``region_fp_overlap``（与 GT 有交集，但对应 GT 覆盖率不足所以不算命中），
    < overlap_ratio_threshold（含交集为 0）的计入 ``region_fp_isolated``。

    除汇总计数外，还逐个区域给出明细：
    - ``gt_regions``: 每个 GT 缺陷的面积、外接框、覆盖率、最佳 IoU、匹配到的预测区域、
      检出/漏检状态，以及漏检时的 ``miss_type``（``overlap`` / ``isolated``）；
    - ``pred_regions``: 每个预测区域的面积、外接框、与 GT 的最佳 IoU、状态
      （``matched`` / ``false_alarm_overlap`` / ``false_alarm_isolated``）、与 GT 的交集像素数、
      该预测区域落在 GT 内的像素比例，以及被它覆盖得最好的那个 GT 缺陷的 id、面积与交集像素
      （``main_gt_id`` / ``main_gt_area`` / ``main_gt_inter``，与任何 GT 都不相交时 id 为 ``None``）、
      对该 GT 的覆盖率 ``main_gt_coverage``（交集像素 / 该 GT 面积，无交集时为 0）。
    """
    n_gt, gt_labels = cv2.connectedComponents(gt_mask.astype(np.uint8))
    n_pred, pred_labels = cv2.connectedComponents(pred_mask.astype(np.uint8))
    gt_ids = _filter_small_regions(n_gt, gt_labels, min_area)
    pred_ids = _filter_small_regions(n_pred, pred_labels, min_area)
    pred_masks = {p: pred_labels == p for p in pred_ids}

    detected, missed, matched_pred, ious = 0, 0, set(), []
    missed_overlap = missed_isolated = 0
    gt_details = []
    pred_best_iou = {p: 0.0 for p in pred_ids}
    pred_inter = {p: 0 for p in pred_ids}
    # 每个预测区域覆盖得最好的那个 GT 缺陷（覆盖率、id、面积、交集像素），供覆盖率分档使用
    pred_main_gt = {p: (0.0, None, 0, 0) for p in pred_ids}
    for g in gt_ids:
        gt_region = gt_labels == g
        gt_area = int(gt_region.sum())
        best_iou, best_pred, best_inter = 0.0, None, 0
        hit_preds = []
        for p in pred_ids:
            intersection = int(np.logical_and(gt_region, pred_masks[p]).sum())
            if intersection == 0:
                continue
            pred_inter[p] += intersection
            ratio = intersection / (gt_area + 1e-8)
            if ratio > pred_main_gt[p][0]:
                pred_main_gt[p] = (ratio, g, gt_area, intersection)
            if ratio >= coverage_threshold:
                hit_preds.append(p)
            iou = intersection / np.logical_or(gt_region, pred_masks[p]).sum()
            pred_best_iou[p] = max(pred_best_iou[p], iou)
            if intersection > best_inter:
                best_pred, best_inter = p, intersection
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
        covered_ratio = best_inter / (gt_area + 1e-8)
        hit = covered_ratio >= coverage_threshold
        miss_type = None
        if hit:
            detected += 1
            matched_pred.update(hit_preds)
        else:
            missed += 1
            if covered_ratio >= overlap_ratio_threshold:
                miss_type = "overlap"
                missed_overlap += 1
            else:
                miss_type = "isolated"
                missed_isolated += 1
        x, y, w, h = _region_bbox(gt_region)
        gt_details.append({
            "gt_id": g,
            "area": gt_area,
            "bbox": (x, y, w, h),
            "best_iou": best_iou,
            "matched_pred_id": best_pred if hit else None,
            "matched_inter": best_inter if hit else 0,
            "covered_ratio": covered_ratio,
            "status": "detected" if hit else "missed",
            "miss_type": miss_type,
        })

    pred_details = []
    fp_overlap = fp_isolated = 0
    for p in pred_ids:
        pred_area = int(pred_masks[p].sum())
        x, y, w, h = _region_bbox(pred_masks[p])
        overlap_ratio = pred_inter[p] / (pred_area + 1e-8)
        if p in matched_pred:
            status = "matched"
        elif overlap_ratio >= overlap_ratio_threshold:
            status = "false_alarm_overlap"
            fp_overlap += 1
        else:
            status = "false_alarm_isolated"
            fp_isolated += 1
        main_gt_coverage, main_gt_id, main_gt_area, main_gt_inter = pred_main_gt[p]
        pred_details.append({
            "pred_id": p,
            "area": pred_area,
            "bbox": (x, y, w, h),
            "best_iou": pred_best_iou[p],
            "gt_inter": pred_inter[p],
            "gt_overlap_ratio": overlap_ratio,
            "status": status,
            "main_gt_id": main_gt_id,
            "main_gt_area": main_gt_area if main_gt_id is not None else None,
            "main_gt_inter": main_gt_inter,
            "main_gt_coverage": main_gt_coverage,
        })

    false_alarm_regions = len(pred_ids) - len(matched_pred)
    return {
        "region_gt": len(gt_ids),
        "region_detected": detected,
        "region_missed": missed,
        "region_missed_overlap": missed_overlap,
        "region_missed_isolated": missed_isolated,
        "region_fp": false_alarm_regions,
        "region_fp_overlap": fp_overlap,
        "region_fp_isolated": fp_isolated,
        "region_pred": len(pred_ids),
        "region_best_iou": max(ious) if ious else 0.0,
        "gt_regions": gt_details,
        "pred_regions": pred_details,
    }


DEFAULT_AREA_SPLIT = 300


def collect_gt_records(image_path, region_metrics):
    """收集每个 GT 缺陷的面积与覆盖率，供区域级汇总表统计使用。"""
    image_name = Path(image_path).name
    return [
        {
            "image": image_name,
            "gt_id": r["gt_id"],
            "area": r["area"],
            "covered_ratio": r["covered_ratio"],
            "status": r["status"],
        }
        for r in region_metrics["gt_regions"]
    ]


def summarize_gt_area_table(gt_records, split=DEFAULT_AREA_SPLIT, coverage_threshold=0.6):
    """GT 缺陷按面积分成 < split / >= split 两桶，各桶给出缺陷数与漏检数（fn）。"""
    small = [r for r in gt_records if r["area"] < split]
    large = [r for r in gt_records if r["area"] >= split]

    def fn(items):
        return sum(1 for r in items if r["covered_ratio"] < coverage_threshold)

    return {
        "total": len(gt_records),
        "small": len(small),
        "small_fn": fn(small),
        "large": len(large),
        "large_fn": fn(large),
    }


def collect_pred_records(image_path, region_metrics):
    """收集每个预测区域对它交集最大那个 GT 缺陷的覆盖率，供区域级汇总表统计使用。"""
    image_name = Path(image_path).name
    return [
        {
            "image": image_name,
            "pred_id": r["pred_id"],
            "area": r["area"],
            "main_gt_coverage": r["main_gt_coverage"],
            "status": r["status"],
        }
        for r in region_metrics["pred_regions"]
    ]


def summarize_pred_overlap_table(pred_records, coverage_threshold=0.6, overlap_ratio_threshold=0.01):
    """预测区域按它对 GT 的覆盖率分成 >=60% / 1%-60% / 0%-1% 三档。

    覆盖率 = 该预测区域与 GT 的交集像素 / 该 GT 面积（取交集最大的那个 GT，
    与任何 GT 都不相交时记 0），因此三档之和等于预测区域总数。
    """
    high = sum(1 for r in pred_records if r["main_gt_coverage"] >= coverage_threshold)
    mid = sum(1 for r in pred_records if overlap_ratio_threshold <= r["main_gt_coverage"] < coverage_threshold)
    low = sum(1 for r in pred_records if r["main_gt_coverage"] < overlap_ratio_threshold)
    return {"total": len(pred_records), "high": high, "mid": mid, "low": low}


def format_region_summary_tables(
    gt_records,
    pred_records,
    total_fp,
    split=DEFAULT_AREA_SPLIT,
    coverage_threshold=0.6,
    overlap_ratio_threshold=0.01,
    split_side=None,
):
    """把区域级结果格式化成两张汇总表。

    表1 按 GT 缺陷面积分桶：``total | < split | fn | >= split | fn | total_fp``，
    fn 为该面积桶里覆盖率不足 coverage_threshold 的缺陷数，total_fp 为全部误报区域数。
    传入 split_side 时表头按 ``N x N`` 展示边长（分界面积为 N*N）。
    表2 按预测区域对 GT 的覆盖率分档：``total | >= coverage | overlap~coverage | < overlap``，
    total 为预测区域总数，三档之和等于 total。
    """
    if not gt_records and not pred_records:
        return []
    area_row = summarize_gt_area_table(gt_records, split, coverage_threshold)
    overlap_row = summarize_pred_overlap_table(pred_records, coverage_threshold, overlap_ratio_threshold)
    if split_side:
        split_label = f"{split_side}x{split_side}={split} 像素"
    else:
        side = int(round(split**0.5))
        split_label = f"{split} 像素（约 {side}x{side}）"
    high_label = f"与GT交集(>={coverage_threshold:.0%})"
    mid_label = f"与GT交集({overlap_ratio_threshold:.0%}-{coverage_threshold:.0%})"
    low_label = f"与GT交集(0%-{overlap_ratio_threshold:.0%})"
    return [
        f"【GT缺陷面积分桶】分界={split_label}；fn=覆盖率 < {coverage_threshold:.0%} 的缺陷数",
        f"{'total':>8} {f'< {split}':>10} {'fn':>8} {f'> {split}':>10} {'fn':>8} {'total_fp':>10}",
        f"{area_row['total']:>8} {area_row['small']:>10} {area_row['small_fn']:>8} "
        f"{area_row['large']:>10} {area_row['large_fn']:>8} {total_fp:>10}",
        "【预测区域与 GT 的交集分档】total=预测区域总数；覆盖率 = 该区域覆盖的 GT 像素 / 该 GT 面积",
        f"{'total':>8} {high_label:>18} {mid_label:>18} {low_label:>18}",
        f"{overlap_row['total']:>8} {overlap_row['high']:>18} {overlap_row['mid']:>18} {overlap_row['low']:>18}",
    ]


DEFAULT_AREA_BINS = [16, 64, 256, 1024, 4096]


def collect_gt_area_records(image_path, region_metrics):
    """收集每个 GT 缺陷的面积与检出状态，供面积分箱统计使用。"""
    image_name = Path(image_path).name
    return [
        {
            "image": image_name,
            "gt_id": r["gt_id"],
            "area": r["area"],
            "status": r["status"],
            "covered_ratio": r["covered_ratio"],
        }
        for r in region_metrics["gt_regions"]
    ]


def parse_area_bins(edges):
    """把分箱边界归一化成升序正整数列表。

    兼容 CLI 传入的字符串元素、``"4,8,16"`` / ``"[4, 8, 16]"`` 这样的整串写法，
    以及中文全角逗号。
    """
    if edges is None:
        return []
    if isinstance(edges, str):
        edges = edges.replace("，", ",").strip().strip("[]").split(",")
    values = set()
    for e in edges:
        text = str(e).strip().strip("[]")
        if not text:
            continue
        value = int(float(text))
        if value > 0:
            values.add(value)
    return sorted(values)


def resolve_area_bin_edges(edges, sides=None):
    """解析面积分箱边界：给了边长列表就按 N*N 换算，否则直接用面积值。

    两个参数都接受 ``[4, 8]`` / ``"4,8"`` / ``"[4, 8]"`` 等写法。
    """
    sides = parse_area_bins(sides)
    if sides:
        return sorted({s**2 for s in sides})
    return parse_area_bins(edges)


def summarize_area_bins(records, edges):
    """按缺陷面积分箱统计检出/漏检数量。

    edges 为升序的面积分界值（像素数），生成 (0, e1]、(e1, e2] ... (en, inf) 各区间。
    """
    edges = parse_area_bins(edges)
    bounds = [(0, edges[0])] if edges else []
    bounds += [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    bounds.append((edges[-1] if edges else 0, float("inf")))

    rows = []
    for low, high in bounds:
        in_bin = [r for r in records if low < r["area"] <= high]
        missed = sum(1 for r in in_bin if r["status"] == "missed")
        total = len(in_bin)
        label = f"({low}, {high}]" if high != float("inf") else f"> {low}"
        missed_items = sorted(
            ((r["image"], r["area"]) for r in in_bin if r["status"] == "missed"), key=lambda t: (t[0], t[1])
        )
        rows.append({
            "range": label,
            "total": total,
            "detected": total - missed,
            "missed": missed,
            "miss_rate": missed / (total + 1e-8),
            "missed_items": missed_items,
        })
    return rows


def format_fp_image_summary(records, max_examples=0):
    """把有误报的图片按误报类型列成清单。

    records 每项为 ``{'image', 'fp_overlap', 'fp_isolated'}``，只列出有误报的图片。
    max_examples 控制每类最多列多少张：0 表示全部列出，正数表示最多 N 条，负数表示不列。
    """
    if max_examples < 0:
        return []
    groups = [
        ("与GT区域有交集的误报图片", "fp_overlap"),
        ("与GT区域无交集的误报图片", "fp_isolated"),
    ]
    lines = []
    for title, key in groups:
        items = sorted(((r["image"], r[key]) for r in records if r[key] > 0), key=lambda t: (-t[1], t[0]))
        if not items:
            lines.append(f"{title}: 无")
            continue
        shown = items if max_examples == 0 else items[:max_examples]
        text = ", ".join(f"{name}({count})" for name, count in shown)
        if len(items) > len(shown):
            text += f" ... 共 {len(items)} 张"
        lines.append(f"{title}（共 {len(items)} 张，格式为 图片名(误报区域数)）: {text}")
    return lines


def summarize_missed_areas(records):
    """漏检缺陷与检出缺陷的面积分布对比。"""
    missed = np.array([r["area"] for r in records if r["status"] == "missed"], dtype=float)
    detected = np.array([r["area"] for r in records if r["status"] == "detected"], dtype=float)

    def stats(areas):
        if areas.size == 0:
            return None
        return {
            "count": int(areas.size),
            "min": float(areas.min()),
            "p25": float(np.percentile(areas, 25)),
            "median": float(np.median(areas)),
            "mean": float(areas.mean()),
            "p75": float(np.percentile(areas, 75)),
            "max": float(areas.max()),
        }

    return {"missed": stats(missed), "detected": stats(detected)}


def format_area_summary(records, edges, max_examples=0):
    """把面积分箱统计与漏检面积分布格式化成文本行。

    max_examples 控制每个面积区间列出多少张漏检图片名：0 表示全部列出，
    正数表示最多列 N 条，负数表示不列。
    """
    if not records:
        return []
    lines = [
        "【漏检缺陷面积统计】面积单位为像素，基于评估分辨率下的 GT 连通域",
        f"{'面积区间':>16} {'缺陷数':>8} {'检出':>8} {'漏检':>8} {'漏检率':>10}",
    ]
    rows = summarize_area_bins(records, edges)
    for row in rows:
        lines.append(
            f"{row['range']:>16} {row['total']:>8} {row['detected']:>8} {row['missed']:>8} {row['miss_rate']:>9.2%}"
        )

    if max_examples >= 0 and any(row["missed"] for row in rows):
        lines.append("【各面积区间的漏检图片】格式为 图片名(缺陷面积)")
        for row in rows:
            if not row["missed"]:
                continue
            shown = row["missed_items"] if max_examples == 0 else row["missed_items"][:max_examples]
            text = ", ".join(f"{name}({area})" for name, area in shown)
            if len(row["missed_items"]) > len(shown):
                text += f" ... 共 {len(row['missed_items'])} 个"
            lines.append(f"  {row['range']}: {text}")

    dist = summarize_missed_areas(records)
    for key, title in (("missed", "漏检缺陷面积"), ("detected", "检出缺陷面积")):
        s = dist[key]
        if s is None:
            lines.append(f"{title}: 无")
            continue
        lines.append(
            f"{title}: n={s['count']} min={s['min']:.0f} p25={s['p25']:.0f} "
            f"中位数={s['median']:.0f} 均值={s['mean']:.1f} p75={s['p75']:.0f} max={s['max']:.0f}"
        )
    return lines


def collect_region_rows(image_path, region_metrics):
    """把单张图的区域级明细展开成每个区域一行，供 CSV 输出。"""
    image_name = Path(image_path).name
    rows = []
    for r in region_metrics["gt_regions"]:
        x, y, w, h = r["bbox"]
        rows.append({
            "image": image_name,
            "region_type": "gt",
            "region_id": r["gt_id"],
            "status": r["status"],
            "area": r["area"],
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "best_iou": round(r["best_iou"], 4),
            "matched_id": "" if r["matched_pred_id"] is None else r["matched_pred_id"],
            "overlap_ratio": round(r["covered_ratio"], 4),
            "miss_type": r["miss_type"] or "",
        })
    for r in region_metrics["pred_regions"]:
        x, y, w, h = r["bbox"]
        rows.append({
            "image": image_name,
            "region_type": "pred",
            "region_id": r["pred_id"],
            "status": r["status"],
            "area": r["area"],
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "best_iou": round(r["best_iou"], 4),
            "matched_id": "" if r["main_gt_id"] is None else r["main_gt_id"],
            "overlap_ratio": round(r["gt_overlap_ratio"], 4),
            "gt_inter": r["gt_inter"],
            "main_gt_area": "" if r["main_gt_area"] is None else r["main_gt_area"],
            "main_gt_inter": r["main_gt_inter"],
            "main_gt_coverage": round(r["main_gt_coverage"], 4),
        })
    return rows


def format_region_details(region_metrics):
    """把每个 GT 缺陷的检出/漏检情况以及误报的预测区域格式化成文本行。"""
    lines = []
    for r in region_metrics["gt_regions"]:
        x, y, w, h = r["bbox"]
        if r["status"] == "detected":
            state = "检出"
        else:
            state = "漏检(与预测有交集)" if r["miss_type"] == "overlap" else "漏检(与预测无交集)"
        matched = f" 主匹配预测#{r['matched_pred_id']}" if r["matched_pred_id"] is not None else ""
        lines.append(
            f"GT#{r['gt_id']} {state} 面积={r['area']} 框=({x},{y},{w},{h}) "
            f"覆盖率={r['covered_ratio']:.2%} IoU={r['best_iou']:.4f}{matched}"
        )
    for r in region_metrics["pred_regions"]:
        if r["status"] == "matched":
            continue
        x, y, w, h = r["bbox"]
        kind = "误报(与GT有交集)" if r["status"] == "false_alarm_overlap" else "误报(与GT无交集)"
        main_gt = "无" if r["main_gt_id"] is None else f"#{r['main_gt_id']}(面积={r['main_gt_area']})"
        lines.append(
            f"预测#{r['pred_id']} {kind} 面积={r['area']} 框=({x},{y},{w},{h}) "
            f"IoU={r['best_iou']:.4f} 与GT交集={r['gt_inter']} "
            f"落入GT比例={r['gt_overlap_ratio']:.2%} 主相交GT={main_gt} "
            f"对主相交GT覆盖率={r['main_gt_coverage']:.2%}"
        )
    return lines


def print_region_details(region_metrics):
    """打印每个 GT 缺陷的检出/漏检情况，以及误报的预测区域。"""
    for line in format_region_details(region_metrics):
        print(f"     {line}")


# ---------- 图像级评估函数 ----------
def get_image_level_gt(gt_path):
    """判断图像级别的GT标签（有缺陷=1，无缺陷=0）"""
    gt_mask = load_gt_mask(gt_path)
    return 1 if gt_mask.sum() > 0 else 0


def compute_image_metrics(predictions, gt_files, score_threshold=0.5, no_gt_as_normal=False):
    """
    计算图像级别的混淆矩阵和评估指标

    Args:
        predictions: 模型预测结果列表
        gt_files: GT文件字典 {stem: path}
        score_threshold: 异常分数阈值
        no_gt_as_normal: gt_files 中缺少 GT 时是否按正常图计入统计

    Returns:
        dict: 包含TP, FP, TN, FN、各项指标，以及误检(fp_images)/漏检(fn_images)的图片名
    """
    img_tp = img_fp = img_tn = img_fn = 0
    fp_images, fn_images = [], []

    for idx, pred in enumerate(predictions):
        image_paths = pred.image_path if isinstance(pred.image_path, list) else [pred.image_path]

        for i, img_path in enumerate(image_paths):
            name = Path(img_path).stem

            # 获取预测分数和标签
            score = pred.pred_score[i].cpu().item() if pred.pred_score.dim() > 0 else pred.pred_score.cpu().item()
            pred_label = 1 if score >= score_threshold else 0

            # 获取GT标签：优先用数据集自带的（测试模式），否则按文件名查 gt_dir
            gt_label = get_batch_gt_label(pred, i)
            if gt_label is None and name in gt_files:
                gt_label = get_image_level_gt(gt_files[name])
            elif gt_label is None and no_gt_as_normal:
                gt_label = 0
            if gt_label is not None:
                # 统计混淆矩阵
                if pred_label == 1 and gt_label == 1:
                    img_tp += 1
                elif pred_label == 1 and gt_label == 0:
                    img_fp += 1
                    fp_images.append((Path(img_path).name, score))
                elif pred_label == 0 and gt_label == 0:
                    img_tn += 1
                elif pred_label == 0 and gt_label == 1:
                    img_fn += 1
                    fn_images.append((Path(img_path).name, score))

    # 计算指标
    eps = 1e-8
    total_positive = img_tp + img_fn
    total_negative = img_fp + img_tn

    miss_rate = img_fn / (total_positive + eps)  # 漏检率 = FN / (TP + FN)
    false_alarm = img_fp / (total_negative + eps)  # 误检率 = FP / (FP + TN)
    precision = img_tp / (img_tp + img_fp + eps)
    recall = img_tp / (img_tp + img_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (img_tp + img_tn) / (img_tp + img_fp + img_tn + img_fn + eps)

    return {
        "tp": img_tp,
        "fp": img_fp,
        "tn": img_tn,
        "fn": img_fn,
        "miss_rate": miss_rate,
        "false_alarm": false_alarm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fp_images": sorted(fp_images, key=lambda t: (-t[1], t[0])),
        "fn_images": sorted(fn_images, key=lambda t: (t[1], t[0])),
    }


def format_image_error_lists(img_metrics):
    """把图像级误检/漏检的图片名列成文本行，格式为 图片名(异常分数)。"""
    lines = []
    for title, key in (("误检图片(好图判为NG)", "fp_images"), ("漏检图片(缺陷图判为OK)", "fn_images")):
        items = img_metrics.get(key, [])
        if not items:
            lines.append(f"{title}: 无")
            continue
        text = ", ".join(f"{name}({score:.4f})" for name, score in items)
        lines.append(f"{title}（共 {len(items)} 张，格式为 图片名(异常分数)）: {text}")
    return lines


def collect_eval_pairs(predictions, gt_dir=None):
    """逐张收集回退阈值扫描所需的 (anomaly_map, gt_mask) 对。"""
    gt_files = {}
    if gt_dir:
        gt_files = {f.stem: f for f in Path(gt_dir).iterdir() if f.is_file()}

    pairs = []
    for pred in predictions:
        if getattr(pred, "anomaly_map", None) is None:
            continue
        image_paths = pred.image_path if isinstance(pred.image_path, list) else [pred.image_path]
        for i, img_path in enumerate(image_paths):
            amap = pred.anomaly_map[i] if pred.anomaly_map.dim() >= 3 else pred.anomaly_map
            amap = amap.squeeze().cpu().numpy()
            gt_mask = get_batch_gt_mask(pred, i)
            if gt_mask is None:
                gt_path = gt_files.get(Path(img_path).stem)
                if gt_path is None:
                    continue
                gt_mask = load_gt_mask(gt_path)
            if gt_mask.shape != amap.shape:
                gt_mask = cv2.resize(gt_mask, (amap.shape[1], amap.shape[0]), interpolation=cv2.INTER_NEAREST)
            pairs.append((amap, gt_mask))
    return pairs


def scan_best_threshold(predictions, gt_dir=None, num_steps=200, level="pixel", no_gt_as_normal=False):
    """扫描 anomaly_map 二值化回退路径的最佳阈值。

    Args:
        predictions: 预测结果
        gt_dir: GT目录（测试模式下可为 None，GT 从数据集取）
        num_steps: 扫描步数
        level: 'pixel' 或 'image'
        no_gt_as_normal: 图像级扫描时，gt_files 中缺少 GT 是否按正常图计入统计
    """
    thresholds = np.linspace(0.0, 1.0, num_steps)
    best_f1, best_th = 0.0, 0.5

    if level == "image":
        gt_files = {f.stem: f for f in Path(gt_dir).iterdir() if f.is_file()} if gt_dir else {}
        for th in thresholds:
            metrics = compute_image_metrics(
                predictions,
                gt_files,
                score_threshold=th,
                no_gt_as_normal=no_gt_as_normal,
            )
            if metrics["f1"] > best_f1:
                best_f1, best_th = metrics["f1"], th
    else:
        pairs = collect_eval_pairs(predictions, gt_dir)
        if not pairs:
            print("警告：没有可用的GT掩膜，使用默认阈值 0.5")
            return 0.5
        for th in thresholds:
            total_tp = total_fp = total_fn = 0
            for amap, gt_mask in pairs:
                pred_mask = amap > th
                gt_bool = gt_mask > 0
                total_tp += np.logical_and(pred_mask, gt_bool).sum()
                total_fp += np.logical_and(pred_mask, ~gt_bool).sum()
                total_fn += np.logical_and(~pred_mask, gt_bool).sum()
            precision = total_tp / (total_tp + total_fp + 1e-8)
            recall = total_tp / (total_tp + total_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1, best_th = f1, th

    print(f"扫描完成（{level}级）：最佳阈值 = {best_th:.4f}，对应 F1 = {best_f1:.4f}")
    return best_th


# ---------- 主推理函数 ----------
def infer(args):
    install_legacy_metric_compat()
    output_dir = Path(args.output)
    heatmap_dir = output_dir / "heatmap"
    overlay_dir = output_dir / "overlay"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(default_root_dir=str(output_dir), devices=1)
    print(">>> 加载模型...")
    model = build_model_from_config(
        args.model,
        args.image_size,
        build_post_processor(args.pixel_sensitivity, args.image_sensitivity),
    )

    print(">>> 加载数据集...")
    data_cfg = dict(args.data)
    if "image_size" in data_cfg and isinstance(data_cfg["image_size"], list):
        data_cfg["image_size"] = tuple(data_cfg["image_size"])
    if "transform" in data_cfg and isinstance(data_cfg["transform"], dict):
        data_cfg["transform"] = build_transform(data_cfg["transform"])

    if args.mode == "test":
        # 测试模式：Folder 数据集自带 gt_label / gt_mask，先跑 anomalib 官方指标表
        # 默认 val_split_mode=from_test 会抽走一半测试图，评估时改成 same_as_test 保证测试集完整
        data_cfg.setdefault("val_split_mode", "same_as_test")
        datamodule = Folder(**data_cfg)
        datamodule.setup("test")
        print(f">>> 测试集共 {len(datamodule.test_data)} 张图片")
        print(f">>> 使用权重: {args.ckpt_path}")
        engine.test(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        dataloader = datamodule.test_dataloader()
    else:
        dataset = PredictDataset(**data_cfg)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
        print(f">>> 开始推理，共 {len(dataset)} 张图片")
        print(f">>> 使用权重: {args.ckpt_path}")

    predictions = engine.predict(model=model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)
    if predictions is None or len(predictions) == 0:
        print("!!! 未获得任何预测结果")
        return

    # 准备GT文件索引
    gt_files = {}
    if args.gt_dir:
        gt_dir_path = Path(args.gt_dir)
        for f in gt_dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                gt_files[f.stem] = f
    image_no_gt_as_normal = args.no_gt_as_normal and args.mode == "predict"

    # 阈值处理（像素级）
    if args.scan_threshold and (args.gt_dir or args.mode == "test"):
        print(">>> 正在扫描最佳阈值（像素级）...")
        pixel_threshold = scan_best_threshold(predictions, args.gt_dir, level="pixel")
    else:
        pixel_threshold = args.threshold
    print(f">>> 使用像素级回退阈值: {pixel_threshold:.4f}（仅在缺少 anomalib pred_mask 时，对 anomaly_map 二值化使用）")

    pred_mask_available = any(getattr(pred, "pred_mask", None) is not None for pred in predictions)
    fallback_available = any(
        getattr(pred, "pred_mask", None) is None and getattr(pred, "anomaly_map", None) is not None
        for pred in predictions
    )
    if pred_mask_available and fallback_available:
        pixel_mask_source = "优先使用 anomalib pred_mask；缺失时回退到 anomaly_map > 回退阈值"
    elif pred_mask_available:
        pixel_mask_source = "anomalib pred_mask"
    else:
        pixel_mask_source = "anomaly_map > 回退阈值"
    print(f">>> 像素级/区域级评估掩膜来源: {pixel_mask_source}")

    # 图像级阈值
    if args.scan_image_threshold and (args.gt_dir or args.mode == "test"):
        print(">>> 正在扫描最佳阈值（图像级）...")
        image_threshold = scan_best_threshold(
            predictions,
            args.gt_dir,
            level="image",
            no_gt_as_normal=image_no_gt_as_normal,
        )
    else:
        image_threshold = args.image_threshold
    print(f">>> 使用图像级阈值: {image_threshold:.4f}")

    # 面积分桶分界：给了边长就用 N*N，否则直接用面积值
    area_split = args.area_split_side**2 if args.area_split_side else args.area_split
    area_bin_edges = resolve_area_bin_edges(args.area_bins, args.area_bin_sides)

    results = []
    region_rows = []
    gt_area_records = []
    gt_records = []
    pred_records = []
    region_detail_blocks = []
    total_tp = total_fp = total_fn = total_tn = 0
    total_gt_regions = total_detected = total_missed = total_region_fp = 0
    total_region_fp_overlap = total_region_fp_isolated = 0
    images_with_fp = images_with_fp_isolated = 0
    fp_image_records = []

    for idx, pred in enumerate(predictions):
        image_paths = pred.image_path if isinstance(pred.image_path, list) else [pred.image_path]
        for i, img_path in enumerate(image_paths):
            score = pred.pred_score[i].cpu().item() if pred.pred_score.dim() > 0 else pred.pred_score.cpu().item()
            label = pred.pred_label[i].cpu().item() if pred.pred_label.dim() > 0 else pred.pred_label.cpu().item()
            try:
                print(f"[{idx + 1}] {img_path} | score={score:.4f} | label={label}")
            except UnicodeEncodeError:
                print(f"[{idx + 1}] <路径含特殊字符> | score={score:.4f} | label={label}")

            name = Path(img_path).stem

            # 保存热力图和叠加图
            anomaly_map = getattr(pred, "anomaly_map", None)
            amap = None
            if anomaly_map is not None:
                amap = anomaly_map[i] if anomaly_map.dim() >= 3 else anomaly_map
                img = pred.image[i] if pred.image.dim() == 4 else pred.image
                save_heatmap(amap, heatmap_dir / f"{name}_heatmap.jpg")
                save_overlay(img, amap, overlay_dir / f"{name}_overlay.jpg")

            # 像素级评估：优先使用 anomalib 的 pred_mask，否则按回退阈值二值化 anomaly_map
            pixel_metrics = None
            pred_mask = gt_mask = None
            img_gt_label = get_batch_gt_label(pred, i)
            batch_gt_mask = get_batch_gt_mask(pred, i)
            eval_pred_mask = get_eval_pred_mask(pred, i, amap, pixel_threshold)
            if batch_gt_mask is not None:
                if eval_pred_mask is not None:
                    pred_mask = eval_pred_mask
                    gt_mask = batch_gt_mask
                    if gt_mask.shape != pred_mask.shape:
                        gt_mask = cv2.resize(
                            gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                    pixel_metrics = compute_pixel_metrics(pred_mask, gt_mask)
                    total_tp += pixel_metrics["tp"]
                    total_fp += pixel_metrics["fp"]
                    total_fn += pixel_metrics["fn"]
                    total_tn += pixel_metrics["tn"]
                    print(
                        f"   [像素级] 漏检率={pixel_metrics['miss_rate']:.2%} 误检率={pixel_metrics['false_alarm']:.2%} F1={pixel_metrics['f1']:.4f}"
                    )
            elif args.gt_dir and name in gt_files:
                gt_path = gt_files[name]
                img_gt_label = get_image_level_gt(gt_path)

                if eval_pred_mask is not None:
                    pred_mask = eval_pred_mask
                    gt_mask = load_gt_mask(gt_path, target_shape=pred_mask.shape)
                    pixel_metrics = compute_pixel_metrics(pred_mask, gt_mask)
                    total_tp += pixel_metrics["tp"]
                    total_fp += pixel_metrics["fp"]
                    total_fn += pixel_metrics["fn"]
                    total_tn += pixel_metrics["tn"]
                    print(
                        f"   [像素级] 漏检率={pixel_metrics['miss_rate']:.2%} 误检率={pixel_metrics['false_alarm']:.2%} F1={pixel_metrics['f1']:.4f}"
                    )
                else:
                    print(f"   警告：{name} 无 anomaly_map，跳过像素级评估")
            elif args.no_gt_as_normal and eval_pred_mask is not None:
                # 没有 GT 文件的图当成正常图，用全零 mask 计入（贡献 TN/FP，与 anomalib 口径一致）
                pred_mask = eval_pred_mask
                gt_mask = np.zeros_like(pred_mask)
                img_gt_label = 0 if img_gt_label is None else img_gt_label
                pixel_metrics = compute_pixel_metrics(pred_mask, gt_mask)
                total_tp += pixel_metrics["tp"]
                total_fp += pixel_metrics["fp"]
                total_fn += pixel_metrics["fn"]
                total_tn += pixel_metrics["tn"]
                print(f"   [像素级无GT→视为正常] 误检率={pixel_metrics['false_alarm']:.2%}")
            else:
                print(f"   无GT掩码，跳过像素级评估")

            # 区域级评估：按 GT 缺陷被预测覆盖的比例判定检出/漏检
            if pixel_metrics is not None and pred_mask is not None and gt_mask is not None:
                region_metrics = compute_region_metrics(
                    pred_mask,
                    gt_mask,
                    args.coverage_threshold,
                    args.min_region_area,
                    args.overlap_ratio_threshold,
                )
                pixel_metrics.update(region_metrics)
                total_gt_regions += region_metrics["region_gt"]
                total_detected += region_metrics["region_detected"]
                total_missed += region_metrics["region_missed"]
                total_region_fp += region_metrics["region_fp"]
                total_region_fp_overlap += region_metrics["region_fp_overlap"]
                total_region_fp_isolated += region_metrics["region_fp_isolated"]
                images_with_fp += int(region_metrics["region_fp"] > 0)
                images_with_fp_isolated += int(region_metrics["region_fp_isolated"] > 0)
                if region_metrics["region_fp"] > 0:
                    fp_image_records.append({
                        "image": Path(img_path).name,
                        "fp_overlap": region_metrics["region_fp_overlap"],
                        "fp_isolated": region_metrics["region_fp_isolated"],
                    })
                print(
                    f"   [区域级] 缺陷{region_metrics['region_gt']}个 检出{region_metrics['region_detected']}个 "
                    f"漏检{region_metrics['region_missed']}个 误报{region_metrics['region_fp']}个"
                    f"(与GT有交集{region_metrics['region_fp_overlap']}个/无交集"
                    f"{region_metrics['region_fp_isolated']}个) "
                    f"最大IoU={region_metrics['region_best_iou']:.4f}"
                )
                region_rows += collect_region_rows(img_path, region_metrics)
                gt_area_records += collect_gt_area_records(img_path, region_metrics)
                gt_records += collect_gt_records(img_path, region_metrics)
                pred_records += collect_pred_records(img_path, region_metrics)
                print_region_details(region_metrics)
                region_detail_blocks.append({
                    "image": Path(img_path).name,
                    "summary": (
                        f"缺陷={region_metrics['region_gt']} 检出={region_metrics['region_detected']} "
                        f"漏检={region_metrics['region_missed']} 误报={region_metrics['region_fp']}"
                        f"(有交集={region_metrics['region_fp_overlap']} "
                        f"无交集={region_metrics['region_fp_isolated']})"
                    ),
                    "lines": format_region_details(region_metrics),
                })

            # 构建结果行
            row = {
                "image": str(img_path),
                "score": score,
                "pred_label": 1 if score >= image_threshold else 0,
                "gt_label": img_gt_label if img_gt_label is not None else "N/A",
            }
            if pixel_metrics:
                row.update({
                    "pixel_miss_rate": pixel_metrics["miss_rate"],
                    "pixel_false_alarm": pixel_metrics["false_alarm"],
                    "pixel_f1": pixel_metrics["f1"],
                    "pixel_iou": pixel_metrics["iou"],
                    "pixel_tp": pixel_metrics["tp"],
                    "pixel_fp": pixel_metrics["fp"],
                    "pixel_fn": pixel_metrics["fn"],
                    "pixel_tn": pixel_metrics["tn"],
                    "region_gt": pixel_metrics.get("region_gt", ""),
                    "region_detected": pixel_metrics.get("region_detected", ""),
                    "region_missed": pixel_metrics.get("region_missed", ""),
                    "region_fp": pixel_metrics.get("region_fp", ""),
                    "region_fp_overlap": pixel_metrics.get("region_fp_overlap", ""),
                    "region_fp_isolated": pixel_metrics.get("region_fp_isolated", ""),
                    "region_best_iou": round(pixel_metrics.get("region_best_iou", 0.0), 4),
                })
            results.append(row)

    # ========== 像素级全局汇总 ==========
    if total_tp + total_fn + total_fp > 0:
        global_miss = total_fn / (total_fn + total_tp + 1e-8)
        global_fa = total_fp / (total_fp + total_tn + 1e-8)
        global_precision = total_tp / (total_tp + total_fp + 1e-8)
        global_recall = total_tp / (total_tp + total_fn + 1e-8)
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall + 1e-8)
        global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

        print("\n" + "=" * 60)
        print("【像素级全局评估结果】")
        print(f"漏检率 (Miss Rate):   {global_miss:.2%}")
        print(f"误检率 (False Alarm): {global_fa:.2%}")
        print(f"F1-Score:             {global_f1:.4f}")
        print(f"IoU:                  {global_iou:.4f}")
        print(f"掩膜来源:             {pixel_mask_source}")
        print(f"回退阈值:             {pixel_threshold:.4f}")
        print(f"TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}")
        print("-" * 60)
        region_miss = total_missed / (total_gt_regions + 1e-8)
        print(f"【区域级】覆盖率阈值 {args.coverage_threshold:.2f}")
        print(f"缺陷总数:             {total_gt_regions}")
        print(f"检出:                 {total_detected}")
        print(f"漏检:                 {total_missed}  (漏检率 {region_miss:.2%})")
        print(f"误报区域:             {total_region_fp}")
        print(f"  与GT区域有交集:       {total_region_fp_overlap}")
        print(f"  与GT区域无交集:       {total_region_fp_isolated}")
        print(f"含误报的图片数:       {images_with_fp}（其中含无交集误报 {images_with_fp_isolated} 张）")
        for line in format_fp_image_summary(fp_image_records, args.fp_image_examples):
            print(line)
        area_summary_lines = format_area_summary(gt_area_records, area_bin_edges, args.area_bin_examples)
        if area_summary_lines:
            print("-" * 60)
            for line in area_summary_lines:
                print(line)
        summary_table_lines = format_region_summary_tables(
            gt_records,
            pred_records,
            total_region_fp,
            area_split,
            args.coverage_threshold,
            args.overlap_ratio_threshold,
            args.area_split_side,
        )
        if summary_table_lines:
            print("-" * 60)
            for line in summary_table_lines:
                print(line)
        print("=" * 60)

    # ========== 图像级全局汇总 ==========
    if args.mode == "test" or (args.gt_dir and len(gt_files) > 0):
        img_metrics = compute_image_metrics(
            predictions,
            gt_files,
            score_threshold=image_threshold,
            no_gt_as_normal=image_no_gt_as_normal,
        )

        print("\n" + "=" * 60)
        print("【图像级全局评估结果】")
        print(f"漏检率 (Miss Rate):   {img_metrics['miss_rate']:.2%}")
        print(f"误检率 (False Alarm): {img_metrics['false_alarm']:.2%}")
        print(f"准确率 (Accuracy):    {img_metrics['accuracy']:.2%}")
        print(f"精确率 (Precision):   {img_metrics['precision']:.4f}")
        print(f"召回率 (Recall):      {img_metrics['recall']:.4f}")
        print(f"F1-Score:             {img_metrics['f1']:.4f}")
        print(f"使用阈值:             {image_threshold:.4f}")
        print(f"TP={img_metrics['tp']} FP={img_metrics['fp']} TN={img_metrics['tn']} FN={img_metrics['fn']}")
        print("=" * 60)

        # 保存汇总报告
        summary_path = output_dir / "evaluation_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("【像素级评估】\n")
            if total_tp + total_fn + total_fp > 0:
                f.write(f"漏检率: {global_miss:.4f}\n")
                f.write(f"误检率: {global_fa:.4f}\n")
                f.write(f"F1-Score: {global_f1:.4f}\n")
                f.write(f"IoU: {global_iou:.4f}\n")
                f.write(f"掩膜来源: {pixel_mask_source}\n")
                f.write(f"回退阈值: {pixel_threshold:.4f}\n")
                f.write(f"TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}\n")
                f.write(f"区域级覆盖率阈值: {args.coverage_threshold:.2f}\n")
                f.write(
                    f"区域级 缺陷总数={total_gt_regions} 检出={total_detected} "
                    f"漏检={total_missed} 误报={total_region_fp}\n"
                )
                f.write(f"区域级误报拆分: 与GT有交集={total_region_fp_overlap} 与GT无交集={total_region_fp_isolated}\n")
                f.write(f"含误报图片数={images_with_fp} 含无交集误报图片数={images_with_fp_isolated}\n")
                for line in format_fp_image_summary(fp_image_records, args.fp_image_examples):
                    f.write(line + "\n")
                f.write(f"区域级漏检率: {total_missed / (total_gt_regions + 1e-8):.4f}\n")
                for line in format_area_summary(gt_area_records, area_bin_edges, args.area_bin_examples):
                    f.write(line + "\n")
                for line in format_region_summary_tables(
                    gt_records,
                    pred_records,
                    total_region_fp,
                    area_split,
                    args.coverage_threshold,
                    args.overlap_ratio_threshold,
                    args.area_split_side,
                ):
                    f.write(line + "\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("【图像级评估】\n")
            f.write(f"漏检率: {img_metrics['miss_rate']:.4f}\n")
            f.write(f"误检率: {img_metrics['false_alarm']:.4f}\n")
            f.write(f"准确率: {img_metrics['accuracy']:.4f}\n")
            f.write(f"精确率: {img_metrics['precision']:.4f}\n")
            f.write(f"召回率: {img_metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {img_metrics['f1']:.4f}\n")
            f.write(f"阈值: {image_threshold:.4f}\n")
            f.write(f"TP={img_metrics['tp']} FP={img_metrics['fp']} TN={img_metrics['tn']} FN={img_metrics['fn']}\n")
            for line in format_image_error_lists(img_metrics):
                f.write(line + "\n")
            f.write("=" * 60 + "\n")

            if region_detail_blocks:
                f.write("\n" + "=" * 60 + "\n")
                f.write("【区域级逐图明细】\n")
                for block in region_detail_blocks:
                    f.write(f"\n{block['image']}  {block['summary']}\n")
                    for line in block["lines"]:
                        f.write(f"  {line}\n")
                f.write("=" * 60 + "\n")

        print(f"\n汇总报告已保存: {summary_path}")

    # 保存区域级明细CSV
    if region_rows:
        region_csv_path = output_dir / "region_detail.csv"
        pd.DataFrame(region_rows).to_csv(region_csv_path, index=False, encoding="utf-8-sig")
        print(f"区域级明细已保存: {region_csv_path}")

    # 保存CSV
    csv_path = output_dir / "result.csv"
    df = pd.DataFrame(results)
    if "pixel_miss_rate" in df.columns:
        df["pixel_miss_rate"] = df["pixel_miss_rate"].apply(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
        df["pixel_false_alarm"] = df["pixel_false_alarm"].apply(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n推理完成，结果保存到: {output_dir}")


# ---------- 命令行解析 ----------
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile, help="配置文件路径")
    parser.add_argument("--output", type=str, default="./inference_results")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型权重路径 .ckpt")
    parser.add_argument("--model", type=dict, required=True, help="模型配置")
    parser.add_argument(
        "--data", type=dict, required=True, help="数据配置：predict 模式为 PredictDataset 参数，test 模式为 Folder 参数"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "test"],
        help="predict：单目录推理；test：用 Folder 数据集跑 engine.test，额外输出 anomalib 官方指标表",
    )
    parser.add_argument("--show", type=bool, default=False, help="是否显示结果图像")
    parser.add_argument(
        "--image_size", type=list, default=None, help="训练时的输入尺寸 [height, width]，必须与训练脚本一致"
    )
    # 像素级评估参数
    parser.add_argument("--gt_dir", type=str, default=None, help="GT掩码目录（与测试图片同名）")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="像素级回退二值化阈值：仅在 pred_mask 不可用时用于 anomaly_map（默认0.5）",
    )
    parser.add_argument(
        "--scan_threshold", type=bool, default=True, help="是否自动扫描 anomaly_map 回退路径的最佳阈值（像素级）"
    )
    parser.add_argument(
        "--no_gt_as_normal",
        type=bool,
        default=False,
        help="predict 模式下 gt_dir 中找不到 GT 的图视为正常图，并计入像素级和图像级指标",
    )
    # 图像级评估参数
    parser.add_argument("--image_threshold", type=float, default=0.5, help="图像级异常分数阈值（默认0.5）")
    parser.add_argument("--scan_image_threshold", type=bool, default=True, help="是否自动扫描最佳阈值（图像级）")
    # anomalib 内部后处理阈值（影响可视化四联图里的 pred_mask 红圈），阈值 = 1 - sensitivity
    parser.add_argument(
        "--pixel_sensitivity",
        type=float,
        default=None,
        help="anomalib 像素级灵敏度，阈值=1-该值，默认0.5；调大则红圈变大",
    )
    parser.add_argument(
        "--image_sensitivity", type=float, default=None, help="anomalib 图像级灵敏度，阈值=1-该值，默认0.5"
    )
    # 区域级评估参数
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.6,
        help="区域级判定阈值：单个预测区域覆盖GT缺陷的像素比例>=该值算检出，否则算漏检",
    )
    parser.add_argument(
        "--overlap_ratio_threshold",
        type=float,
        default=0.01,
        help="交集阈值：覆盖率低于该值视为与 GT 无实质交集，用于拆分漏检/误报与覆盖率分档",
    )
    parser.add_argument(
        "--area_split",
        type=int,
        default=DEFAULT_AREA_SPLIT,
        help=f"GT缺陷面积分桶分界（像素数），默认 {DEFAULT_AREA_SPLIT}",
    )
    parser.add_argument(
        "--area_split_side",
        type=int,
        default=None,
        help="按边长指定面积分桶分界：传 N 则分界=N*N 像素（优先于 --area_split）",
    )
    parser.add_argument(
        "--min_region_area", type=int, default=0, help="忽略面积小于该值的连通域（像素数），用于过滤噪点"
    )
    parser.add_argument(
        "--area_bins",
        type=Union[str, list[int]],
        default=DEFAULT_AREA_BINS,
        help="缺陷面积分箱边界（像素数），用于统计各面积区间的漏检率，如 [4,8,16,32] 或 4,8,16,32",
    )
    parser.add_argument(
        "--area_bin_sides",
        type=Union[str, list[int]],
        default=None,
        help="按边长指定面积分箱边界：传 2,4,8 则边界=4,16,64 像素（优先于 --area_bins）",
    )
    parser.add_argument(
        "--area_bin_examples",
        type=int,
        default=0,
        help="每个面积区间列出多少张漏检图片名：0=全部列出，N>0=最多 N 条，负数=不列",
    )
    parser.add_argument(
        "--fp_image_examples",
        type=int,
        default=0,
        help="每类误报列出多少张图片名：0=全部列出，N>0=最多 N 条，负数=不列",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    infer(args)
