# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# 中文注释/文档字符串会触发全角字符与句末标点检查，这里统一关闭。
# ruff: noqa: D415, RUF001, RUF002, RUF003

"""在原图尺寸上保存 anomalib 三联图（原图 | 原图+热力图 | 原图+预测掩膜轮廓）。

anomalib 自带的 ``ImageVisualizer`` 用 ``item.image``（模型输入张量）作底图，并把每一格
缩放到 ``field_size``（默认 256x256），所以导出的三联图分辨率很低、缺陷细节看不清。
``FullResImageVisualizer`` 改为从 ``item.image_path`` 读原图，把 anomaly_map / mask 插值
到原图尺寸后再叠加，因此输出图与原图同分辨率。

用法一：训练/测试脚本里替换 visualizer（例如 superadd.py）::

    from full_res_visualizer import FullResImageVisualizer

    model = SuperADD(..., visualizer=FullResImageVisualizer())
    engine.test(model=model, datamodule=datamodule)

用法二：直接对已有 ckpt 跑一遍并导出三联图::

    python full_res_visualizer.py --ckpt_path results/.../model.ckpt \
        --input /path/to/images --output results/full_res --model SuperADD \
        --image_size 2688 1856 --model_kwargs '{"patch_size": 640, "patch_overlap": 128}'

注意：``--image_size`` 和 ``--model_kwargs`` 必须与训练脚本（例如 superadd.py）完全一致，
否则推理结果会与训练时不同。

调阈值：``--heat_range VMIN VMAX`` 控制热力图的色彩区间（只影响看起来红不红），
``--pred_threshold`` 覆盖掩膜阈值（影响轮廓圈出多少）。

去噪：``--prefilter median|mean|gaussian`` 在图片送进模型之前做一次滤波（默认 ``none``
即不处理），用来压掉传感器噪点导致的零散高响应；``--prefilter_size`` 调核边长（默认 3）。
三联图的底图仍是未滤波的原图。
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from jsonargparse import ArgumentParser
from PIL import Image, ImageFilter
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader

from anomalib.data import ImageItem, PredictDataset
from anomalib.engine import Engine
from anomalib.models import _get_model_class_by_name
from anomalib.utils.path import generate_output_filename
from anomalib.visualization.image import ImageVisualizer
from anomalib.visualization.image.functional import (
    add_text_to_image,
    create_image_grid,
    overlay_images,
    visualize_anomaly_map,
    visualize_image,
    visualize_mask,
)

if TYPE_CHECKING:
    from lightning.pytorch import Trainer

    from anomalib.data import ImageBatch, NumpyImageBatch, NumpyImageItem
    from anomalib.models import AnomalibModule


def _to_pil_mask(
    mask: torch.Tensor,
    size: tuple[int, int],
    thickness: int,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """把掩膜张量转成原图尺寸的轮廓图，thickness 为轮廓线宽（像素）。"""
    contour = visualize_mask(mask.squeeze().to(torch.uint8) * 255, mode="contour", color=color)
    contour = contour.resize(size, Image.NEAREST)
    if thickness > 1:
        contour = contour.filter(ImageFilter.MaxFilter(thickness if thickness % 2 else thickness + 1))
    return contour


def _gaussian_kernel(size: int) -> torch.Tensor:
    """按 OpenCV 的 ``sigma = 0.3 * ((size - 1) / 2 - 1) + 0.8`` 生成 ``size x size`` 高斯核。"""
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    line = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = torch.outer(line, line)
    return kernel / kernel.sum()


class SpatialFilter(torch.nn.Module):
    """推理前的空间滤波，作用在原始分辨率的图片张量上。

    Args:
        mode: ``median``（去椒盐噪点，保边）、``mean``（均值模糊）或 ``gaussian``（高斯模糊）。
        size: 核边长，必须是 >= 3 的奇数；越大去噪越狠，也越容易抹掉小缺陷。
    """

    def __init__(self, mode: str = "median", size: int = 3) -> None:
        super().__init__()
        if mode not in {"median", "mean", "gaussian"}:
            msg = f"不支持的滤波方式: {mode}"
            raise ValueError(msg)
        if size < 3 or size % 2 == 0:
            msg = f"滤波核边长必须是 >= 3 的奇数，得到: {size}"
            raise ValueError(msg)
        self.mode = mode
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """对 ``(C, H, W)`` 或 ``(N, C, H, W)`` 的图片做滤波，返回同形状张量。"""
        batched = image.dim() == 4
        data = image if batched else image.unsqueeze(0)
        size = self.size
        pad = size // 2
        padded = F.pad(data.float(), (pad, pad, pad, pad), mode="reflect")
        if self.mode == "median":
            patches = padded.unfold(2, size, 1).unfold(3, size, 1).reshape(*data.shape, size * size)
            out = patches.median(dim=-1).values
        else:
            kernel = torch.full((size, size), 1.0 / (size * size)) if self.mode == "mean" else _gaussian_kernel(size)
            channels = data.shape[1]
            weight = kernel.to(padded).expand(channels, 1, size, size)
            out = F.conv2d(padded, weight, groups=channels)
        out = out.to(image.dtype)
        return out if batched else out.squeeze(0)


class FullResImageVisualizer(ImageVisualizer):
    """按原图分辨率导出三联图的 visualizer。

    Args:
        alpha: 热力图叠加权重（0~1），越大热力图越明显。
        max_size: 输出单格的最大边长；原图更大时按比例缩小，``None`` 表示不限制。
        contour_color: 预测掩膜轮廓颜色（RGB）。
        include_gt_mask: GT 掩膜存在时，是否额外输出一格「原图 + GT 轮廓」。
        text: 是否在每一格左上角写标题。
        heat_range: 热力图的显示区间 ``(vmin, vmax)``；区间越窄红色越多。``None`` 表示
            直接用 anomaly_map 的原始取值上色（模型输出一般已归一化到 0~1）。
        pred_threshold: 掩膜阈值；给定时用 ``anomaly_map >= pred_threshold`` 重新算轮廓，
            覆盖 ckpt 里的 pixel_threshold。``None`` 表示沿用模型自己的预测掩膜。
        output_dir: 输出目录，默认为 ``<trainer.default_root_dir>/images``。
        text_config: 标题文字样式，支持 ``font`` / ``size`` / ``color`` / ``background``。
    """

    def __init__(
        self,
        alpha: float = 0.5,
        max_size: int | None = None,
        contour_color: tuple[int, int, int] = (255, 0, 0),
        include_gt_mask: bool = True,
        text: bool = True,
        heat_range: tuple[float, float] | None = None,
        pred_threshold: float | None = None,
        output_dir: str | Path | None = None,
        text_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, text_config=text_config)
        self.output_dir: str | Path | None = output_dir
        self.alpha = alpha
        self.max_size = max_size
        self.contour_color = contour_color
        self.include_gt_mask = include_gt_mask
        self.text = text
        self.heat_range = heat_range
        self.pred_threshold = pred_threshold

    def _base_image(self, item: ImageItem) -> Image.Image | None:
        """读取底图：优先用磁盘上的原图，读不到时退回模型输入张量。"""
        path = Path(str(item.image_path)) if item.image_path else None
        if path is not None and path.is_file():
            image = Image.open(path).convert("RGB")
        elif item.image is not None:
            image = visualize_image(item.image).convert("RGB")
        else:
            return None
        if self.max_size and max(image.size) > self.max_size:
            scale = self.max_size / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        return image

    def _scale_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """按 ``heat_range`` 把 anomaly_map 线性拉伸到 0~1，用于控制热力图的色彩范围。"""
        scaled = anomaly_map.squeeze().float()
        if self.heat_range is None:
            return scaled
        vmin, vmax = self.heat_range
        return ((scaled - vmin) / max(vmax - vmin, 1e-12)).clamp(0.0, 1.0)

    def _pred_mask(self, item: ImageItem) -> torch.Tensor | None:
        """取预测掩膜；``pred_threshold`` 非空时按该阈值重新二值化 anomaly_map。"""
        if self.pred_threshold is None:
            return item.pred_mask
        if item.anomaly_map is None:
            return item.pred_mask
        return item.anomaly_map.squeeze() >= self.pred_threshold

    def visualize_full_res(self, item: ImageItem) -> Image.Image | None:
        """生成单张原图分辨率的三联图。"""
        base = self._base_image(item)
        if base is None:
            return None

        size = base.size
        thickness = max(1, round(min(size) / 500))
        panels: list[tuple[str, Image.Image]] = [("Image", base)]

        if item.anomaly_map is not None:
            heatmap = visualize_anomaly_map(self._scale_anomaly_map(item.anomaly_map), colormap=True, normalize=False)
            heatmap = heatmap.resize(size, Image.BICUBIC)
            panels.append(("Image + Anomaly Map", overlay_images(base, heatmap, alpha=self.alpha).convert("RGB")))

        if self.include_gt_mask and item.gt_mask is not None:
            contour = _to_pil_mask(item.gt_mask, size, thickness, (255, 255, 255))
            panels.append(("Image + GT Mask", overlay_images(base, contour).convert("RGB")))

        pred_mask = self._pred_mask(item)
        if pred_mask is not None:
            contour = _to_pil_mask(pred_mask, size, thickness, self.contour_color)
            panels.append(("Image + Pred Mask", overlay_images(base, contour).convert("RGB")))

        images = [image for _, image in panels]
        if self.text:
            text_kwargs = _text_kwargs(self.text_config)
            images = [add_text_to_image(image, title, **text_kwargs) for title, image in panels]

        return create_image_grid(images, nrow=len(images))

    def visualize(
        self,
        predictions: "ImageItem | NumpyImageItem | ImageBatch | NumpyImageBatch",
    ) -> Image.Image | list[Image.Image | None] | None:
        """对单个 item 走原图分辨率逻辑，其余情况交给父类。"""
        if isinstance(predictions, ImageItem):
            return self.visualize_full_res(predictions)
        return super().visualize(predictions)

    def on_test_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: "ImageBatch",
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """测试批次结束时按原图分辨率保存三联图。"""
        del pl_module, outputs, batch_idx, dataloader_idx
        self._save_batch(trainer, batch)

    def on_predict_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "AnomalibModule",
        outputs: "ImageBatch",
        batch: "ImageBatch",
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """推理批次结束时按原图分辨率保存三联图。"""
        del pl_module, outputs, batch_idx, dataloader_idx
        self._save_batch(trainer, batch)

    def _save_batch(self, trainer: "Trainer", batch: "ImageBatch") -> None:
        """把一个 batch 的三联图写到 ``output_dir``。"""
        output_dir = Path(self.output_dir) if self.output_dir else Path(trainer.default_root_dir) / "images"
        self.output_dir = output_dir

        datamodule = trainer.datamodule if hasattr(trainer, "datamodule") else None
        for item in batch:
            image = self.visualize_full_res(item)
            if image is None:
                continue
            filename = generate_output_filename(
                input_path=item.image_path or "",
                output_path=output_dir,
                dataset_name=datamodule.name if datamodule is not None else None,
                category=datamodule.category if datamodule is not None else None,
            )
            image.save(filename)


def _text_kwargs(text_config: dict[str, Any]) -> dict[str, Any]:
    """把 ``text_config`` 里 ``add_text_to_image`` 认识的字段挑出来。"""
    return {key: value for key, value in text_config.items() if key in {"font", "size", "color", "background"}}


def _get_parser() -> ArgumentParser:
    """构造命令行参数解析器。"""
    parser = ArgumentParser(description="用已有 ckpt 导出原图分辨率的三联图")
    parser.add_argument("--ckpt_path", type=str, required=True, help="模型权重 .ckpt")
    parser.add_argument("--input", type=str, required=True, help="待推理的图片目录或单张图片")
    parser.add_argument("--output", type=str, default="./full_res_visualization", help="输出目录")
    parser.add_argument("--model", type=str, default="SuperADD", help="anomalib 模型类名，如 SuperADD / Patchcore")
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="训练时的输入尺寸，写法 --image_size 2688 1856，必须与训练脚本一致",
    )
    parser.add_argument(
        "--model_kwargs",
        type=dict,
        default={},
        help='其余模型参数（JSON），必须与训练一致，如 \'{"patch_size": 640, "patch_overlap": 128}\'',
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="热力图叠加权重（0~1）")
    parser.add_argument(
        "--heat_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="热力图显示区间，如 --heat_range 0.3 0.8；区间越窄红色越多",
    )
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=None,
        help="掩膜阈值，覆盖 ckpt 里的 pixel_threshold；轮廓太多就调大，漏检就调小",
    )
    parser.add_argument(
        "--prefilter",
        type=str,
        default="none",
        choices=["none", "median", "mean", "gaussian"],
        help="推理前的滤波方式；默认 none 不处理，噪点多可用 median",
    )
    parser.add_argument(
        "--prefilter_size",
        type=int,
        default=3,
        help="滤波核边长（>=3 的奇数，如 3/5/7）；越大去噪越狠，也越容易抹掉小缺陷",
    )
    parser.add_argument("--max_size", type=int, default=None, help="输出单格最大边长，原图过大时按比例缩小")
    return parser


def main() -> None:
    """加载 ckpt 跑一遍推理，并把三联图按原图分辨率写到 --output。"""
    args = _get_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_class = _get_model_class_by_name(args.model)
    model_kwargs: dict[str, Any] = dict(args.model_kwargs)
    if args.image_size:
        height, width = args.image_size
        model_kwargs["pre_processor"] = model_class.configure_pre_processor(image_size=(height, width))
    model = model_class(
        visualizer=FullResImageVisualizer(
            alpha=args.alpha,
            max_size=args.max_size,
            heat_range=tuple(args.heat_range) if args.heat_range else None,
            pred_threshold=args.pred_threshold,
            output_dir=output_dir,
        ),
        **model_kwargs,
    )

    prefilter = SpatialFilter(args.prefilter, args.prefilter_size) if args.prefilter != "none" else None
    dataset = PredictDataset(path=args.input, transform=prefilter)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    print(f">>> 共 {len(dataset)} 张图片，权重: {args.ckpt_path}")

    engine = Engine(default_root_dir=str(output_dir), devices=1)
    engine.predict(model=model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)
    print(f">>> 三联图已保存到: {output_dir}")


if __name__ == "__main__":
    main()
