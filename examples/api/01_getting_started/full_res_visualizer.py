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
        --image_size 2688 1856
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from jsonargparse import ArgumentParser
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from anomalib.data import ImageItem, PredictDataset
from anomalib.engine import Engine
from anomalib.models import get_model
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


class FullResImageVisualizer(ImageVisualizer):
    """按原图分辨率导出三联图的 visualizer。

    Args:
        alpha: 热力图叠加权重（0~1），越大热力图越明显。
        max_size: 输出单格的最大边长；原图更大时按比例缩小，``None`` 表示不限制。
        contour_color: 预测掩膜轮廓颜色（RGB）。
        include_gt_mask: GT 掩膜存在时，是否额外输出一格「原图 + GT 轮廓」。
        text: 是否在每一格左上角写标题。
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

    def visualize_full_res(self, item: ImageItem) -> Image.Image | None:
        """生成单张原图分辨率的三联图。"""
        base = self._base_image(item)
        if base is None:
            return None

        size = base.size
        thickness = max(1, round(min(size) / 500))
        panels: list[tuple[str, Image.Image]] = [("Image", base)]

        if item.anomaly_map is not None:
            heatmap = visualize_anomaly_map(item.anomaly_map.squeeze(), colormap=True, normalize=False)
            heatmap = heatmap.resize(size, Image.BICUBIC)
            panels.append(("Image + Anomaly Map", overlay_images(base, heatmap, alpha=self.alpha).convert("RGB")))

        if self.include_gt_mask and item.gt_mask is not None:
            contour = _to_pil_mask(item.gt_mask, size, thickness, (255, 255, 255))
            panels.append(("Image + GT Mask", overlay_images(base, contour).convert("RGB")))

        if item.pred_mask is not None:
            contour = _to_pil_mask(item.pred_mask, size, thickness, self.contour_color)
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
        type=list[int],
        default=None,
        help="训练时的输入尺寸 [height, width]，必须与训练脚本一致",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="热力图叠加权重（0~1）")
    parser.add_argument("--max_size", type=int, default=None, help="输出单格最大边长，原图过大时按比例缩小")
    return parser


def main() -> None:
    """加载 ckpt 跑一遍推理，并把三联图按原图分辨率写到 --output。"""
    args = _get_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(
        args.model,
        visualizer=FullResImageVisualizer(alpha=args.alpha, max_size=args.max_size, output_dir=output_dir),
    )
    if args.image_size:
        height, width = args.image_size
        model.pre_processor = type(model).configure_pre_processor(image_size=(height, width))

    dataset = PredictDataset(path=args.input)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    print(f">>> 共 {len(dataset)} 张图片，权重: {args.ckpt_path}")

    engine = Engine(default_root_dir=str(output_dir), devices=1)
    engine.predict(model=model, dataloaders=[dataloader], ckpt_path=args.ckpt_path)
    print(f">>> 三联图已保存到: {output_dir}")


if __name__ == "__main__":
    main()
