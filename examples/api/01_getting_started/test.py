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


def get_batch_mask_path(pred, index):
    """从预测批次里取出 GT 掩膜文件路径，没有则返回 None。"""
    mask_path = getattr(pred, "mask_path", None)
    if mask_path is None:
        return None
    path = mask_path[index] if isinstance(mask_path, list) else mask_path
    return str(path) if path else None


def get_original_hw(image_path):
    """读取原图分辨率 (h, w)，读取失败返回 None。"""
    img = safe_imread(str(image_path))
    return None if img is None else img.shape[:2]


def resize_mask(mask, target_hw):
    """把掩膜最近邻缩放到 (h, w)，尺寸一致时原样返回。"""
    if mask.shape == tuple(target_hw):
        return mask
    return cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)


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

    覆盖率 = 该 GT 区域与所有预测区域的交集像素数 / GT 区域面积，
    >= coverage_threshold（默认 60%）记为检出，否则记为漏检；贡献了该覆盖的预测区域
      全部算命中，没命中任何已检出 GT 的预测区域记为误报。IoU 仅作为明细里的参考值。

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
      该预测区域落在 GT 内的像素比例，以及交集最大的那个 GT 缺陷的 id 与面积
      （``main_gt_id`` / ``main_gt_area``，与任何 GT 都不相交时为 ``None``）。
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
    # 每个预测区域交集像素最多的那个 GT 缺陷（面积与 id），供按 GT 面积分桶使用
    pred_main_gt = {p: (0, None, 0) for p in pred_ids}
    for g in gt_ids:
        gt_region = gt_labels == g
        gt_area = int(gt_region.sum())
        best_iou, best_pred, best_inter, covered = 0.0, None, 0, 0
        overlapping = []
        for p in pred_ids:
            intersection = int(np.logical_and(gt_region, pred_masks[p]).sum())
            if intersection == 0:
                continue
            covered += intersection
            pred_inter[p] += intersection
            overlapping.append(p)
            if intersection > pred_main_gt[p][0]:
                pred_main_gt[p] = (intersection, g, gt_area)
            iou = intersection / np.logical_or(gt_region, pred_masks[p]).sum()
            pred_best_iou[p] = max(pred_best_iou[p], iou)
            if intersection > best_inter:
                best_pred, best_inter = p, intersection
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
        covered_ratio = covered / (gt_area + 1e-8)
        hit = covered_ratio >= coverage_threshold
        miss_type = None
        if hit:
            detected += 1
            # 覆盖可能由多个预测区域共同贡献，它们都算命中
            matched_pred.update(overlapping)
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
        _, main_gt_id, main_gt_area = pred_main_gt[p]
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


def collect_gt_area_records(image_path, region_metrics):
    """收集每个 GT 缺陷的面积与检出状态，供面积分箱统计使用。"""
    image_name = Path(image_path).name
    return [
        {
            "image": image_name,
            "gt_id": r["gt_id"],
            "area": r["area"],
            "status": r["status"],
            "miss_type": r["miss_type"],
            "covered_ratio": r["covered_ratio"],
        }
        for r in region_metrics["gt_regions"]
    ]


def collect_pred_area_records(image_path, region_metrics):
    """收集每个预测区域的面积与命中/误报状态，供预测区域面积分桶统计使用。"""
    image_name = Path(image_path).name
    return [
        {
            "image": image_name,
            "pred_id": r["pred_id"],
            "area": r["area"],
            "status": r["status"],
            "gt_overlap_ratio": r["gt_overlap_ratio"],
            "main_gt_area": r["main_gt_area"],
        }
        for r in region_metrics["pred_regions"]
    ]


DEFAULT_AREA_SPLIT = 300


FALSE_ALARM_BUCKET = "误报(全部)"


def summarize_pred_area_split(pred_records, split=DEFAULT_AREA_SPLIT):
    """按「交集最大的那个 GT 缺陷面积」把命中的预测区域分桶，误报单独一行。

    所有误报区域（不论与 GT 有无交集）都归入 ``误报(全部)`` 行，不进面积桶。
    """
    matched_records = [r for r in pred_records if r["status"] == "matched"]
    buckets = [
        (f"< {split}", [r for r in matched_records if (r["main_gt_area"] or 0) < split]),
        (f">= {split}", [r for r in matched_records if (r["main_gt_area"] or 0) >= split]),
        (FALSE_ALARM_BUCKET, [r for r in pred_records if r["status"] != "matched"]),
    ]
    rows = []
    for label, items in buckets:
        rows.append({
            "range": label,
            "total": len(items),
            "matched": sum(1 for r in items if r["status"] == "matched"),
            "fp_overlap": sum(1 for r in items if r["status"] == "false_alarm_overlap"),
            "fp_isolated": sum(1 for r in items if r["status"] == "false_alarm_isolated"),
            "items": items,
        })
    return rows


def summarize_gt_area_split(gt_records, split=DEFAULT_AREA_SPLIT):
    """按面积把 GT 缺陷分成 < split 和 >= split 两桶，并统计检出/漏检构成。"""
    buckets = [
        (f"< {split}", [r for r in gt_records if r["area"] < split]),
        (f">= {split}", [r for r in gt_records if r["area"] >= split]),
    ]
    rows = []
    for label, items in buckets:
        missed = [r for r in items if r["status"] == "missed"]
        rows.append({
            "range": label,
            "total": len(items),
            "detected": len(items) - len(missed),
            "missed": len(missed),
            "missed_overlap": sum(1 for r in missed if r["miss_type"] == "overlap"),
            "missed_isolated": sum(1 for r in missed if r["miss_type"] == "isolated"),
            "miss_rate": len(missed) / (len(items) + 1e-8),
            "missed_items": missed,
        })
    return rows


def _format_area_split_examples(title, items, max_examples):
    """把区域记录列成 ``图片名(面积)`` 清单，返回 0 或 1 行文本。"""
    if max_examples < 0 or not items:
        return []
    pairs = sorted(((r["image"], r["area"]) for r in items), key=lambda t: (t[0], t[1]))
    shown = pairs if max_examples == 0 else pairs[:max_examples]
    text = ", ".join(f"{name}({area})" for name, area in shown)
    if len(pairs) > len(shown):
        text += f" ... 共 {len(pairs)} 个"
    return [f"  {title}（共 {len(pairs)} 个，格式为 图片名(区域面积)）: {text}"]


def format_area_split_summary(
    gt_records,
    pred_records,
    split=DEFAULT_AREA_SPLIT,
    coverage_threshold=0.6,
    overlap_ratio_threshold=0.01,
    max_examples=0,
):
    """把「预测区域 / GT 缺陷」按 GT 缺陷面积（split 像素为界）分桶的统计格式化成文本行。

    命中的预测区域按它交集最大的那个 GT 缺陷的面积分桶；「命中GT数」列取该面积桶里
    覆盖率 >= coverage_threshold 的 GT 缺陷数（与 GT 行的检出数一致），「预测区域数」
    则是预测连通域个数（一个 GT 可能被多个预测区域同时命中）；
    所有误报区域（不论与 GT 有无交集）单独列在 ``误报(全部)`` 行，并拆成
    与 GT 有交集 / 无交集两类；GT 侧两个面积桶另给出漏检（与预测有交集 / 无交集）的拆分。
    max_examples 控制每个桶里列出多少条区域（漏检、误报各自计数）：0 表示全部列出，
    正数表示最多 N 条，负数表示不列。
    """
    if not gt_records and not pred_records:
        return []
    lines = [
        f"【面积分桶统计】分界={split} 像素；检出判定：GT 覆盖率 >= {coverage_threshold:.0%}，"
        f"有交集判定：交集比例 >= {overlap_ratio_threshold:.0%}",
        f"{'预测区域(按命中GT面积)':>16} {'命中GT数':>9} {'预测区域数':>10} {'误报(有交集)':>14} {'误报(无交集)':>14}",
    ]
    pred_rows = summarize_pred_area_split(pred_records, split)
    gt_rows = summarize_gt_area_split(gt_records, split)
    # 命中 GT 数直接取该面积桶里覆盖率达到阈值的 GT 数，与 GT 表的检出数一致
    hit_gt_by_range = {r["range"]: r["detected"] for r in gt_rows}
    for row in pred_rows:
        # 误报行没有对应的命中 GT，置为 "-"
        hit_gt_text = "-" if row["range"] == FALSE_ALARM_BUCKET else str(hit_gt_by_range.get(row["range"], 0))
        lines.append(
            f"{row['range']:>16} {hit_gt_text:>9} {row['total']:>10} "
            f"{row['fp_overlap']:>14} {row['fp_isolated']:>14}"
        )
    lines.append(f"{'GT缺陷面积':>16} {'缺陷数':>8} {'检出':>8} {'漏检':>8} {'漏检率':>10}")
    for row in gt_rows:
        lines.append(
            f"{row['range']:>16} {row['total']:>8} {row['detected']:>8} {row['missed']:>8} {row['miss_rate']:>9.2%}"
        )

    for gt_row in gt_rows:
        label = gt_row["range"]
        lines.append(
            f"面积 {label} 的缺陷漏检: {gt_row['missed']} 个"
            f"（与预测有交集 {gt_row['missed_overlap']} 个 / 无交集 {gt_row['missed_isolated']} 个）"
        )
        lines += _format_area_split_examples(f"面积 {label} 的漏检缺陷", gt_row["missed_items"], max_examples)

    fa_row = next((r for r in pred_rows if r["range"] == FALSE_ALARM_BUCKET), None)
    if fa_row is not None:
        lines.append(
            f"误报预测区域合计: {fa_row['total']} 个"
            f"（与GT有交集 {fa_row['fp_overlap']} 个 / 无交集 {fa_row['fp_isolated']} 个）"
        )
        lines += _format_area_split_examples("误报预测区域", fa_row["items"], max_examples)
    return lines


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
            f"落入GT比例={r['gt_overlap_ratio']:.2%} 主相交GT={main_gt}"
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

    results = []
    region_rows = []
    gt_area_records = []
    pred_area_records = []
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
            # 评估统一在原图分辨率上进行：预测掩膜还原到原图尺寸，GT 优先读原始掩膜文件
            orig_hw = get_original_hw(img_path)
            if eval_pred_mask is not None and orig_hw is not None:
                eval_pred_mask = resize_mask(eval_pred_mask, orig_hw)
            if batch_gt_mask is not None:
                if eval_pred_mask is not None:
                    pred_mask = eval_pred_mask
                    mask_path = get_batch_mask_path(pred, i)
                    if mask_path and Path(mask_path).is_file():
                        gt_mask = load_gt_mask(mask_path, target_shape=pred_mask.shape)
                    else:
                        gt_mask = resize_mask(batch_gt_mask, pred_mask.shape)
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
                pred_area_records += collect_pred_area_records(img_path, region_metrics)
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
        area_split_lines = format_area_split_summary(
            gt_area_records,
            pred_area_records,
            args.area_split,
            args.coverage_threshold,
            args.overlap_ratio_threshold,
            args.area_split_examples,
        )
        if area_split_lines:
            print("-" * 60)
            for line in area_split_lines:
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
                for line in format_area_split_summary(
                    gt_area_records,
                    pred_area_records,
                    args.area_split,
                    args.coverage_threshold,
                    args.overlap_ratio_threshold,
                    args.area_split_examples,
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
        "--scan_threshold", type=bool, default=False, help="是否自动扫描 anomaly_map 回退路径的最佳阈值（像素级）"
    )
    parser.add_argument(
        "--no_gt_as_normal",
        type=bool,
        default=False,
        help="predict 模式下 gt_dir 中找不到 GT 的图视为正常图，并计入像素级和图像级指标",
    )
    # 图像级评估参数
    parser.add_argument("--image_threshold", type=float, default=0.5, help="图像级异常分数阈值（默认0.5）")
    parser.add_argument("--scan_image_threshold", type=bool, default=False, help="是否自动扫描最佳阈值（图像级）")
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
        help="区域级判定阈值：GT缺陷被预测覆盖的像素比例>=该值算检出，否则算漏检（默认0.6）",
    )
    parser.add_argument(
        "--overlap_ratio_threshold",
        type=float,
        default=0.01,
        help="交集判定阈值：交集比例>=该值算与对方区域有交集，否则算无交集（默认0.01）",
    )
    parser.add_argument(
        "--area_split",
        type=int,
        default=DEFAULT_AREA_SPLIT,
        help=(
            "面积分桶分界（像素数）：按 GT 缺陷面积分成小于/不小于该面积两桶，"
            f"预测区域按它交集最大的 GT 缺陷面积归桶（默认{DEFAULT_AREA_SPLIT}）"
        ),
    )
    parser.add_argument(
        "--area_split_examples",
        type=int,
        default=0,
        help="每个面积桶里列出多少条漏检缺陷/误报区域：0=全部列出，N>0=最多 N 条，负数=不列",
    )
    parser.add_argument(
        "--min_region_area", type=int, default=0, help="忽略面积小于该值的连通域（像素数），用于过滤噪点"
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
