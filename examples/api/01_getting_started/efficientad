# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""使用自定义数据集训练 EfficientAd 模型.

EfficientAd 是一个基于 student-teacher 架构的快速异常检测模型，
使用预训练的 EfficientNet 作为 teacher 网络，推理速度极快（毫秒级）。

注意：EfficientAd 训练时需要额外的 ImageNette 数据集作为 penalty 数据源，
      首次运行会自动从网络下载（约 1.5GB），也可以提前手动放到指定目录。

数据集目录结构（二选一）：

方式一：只有正常图（无监督，推荐）
    my_dataset/
    └── good/          # 正常图片（训练用）
        ├── img001.png
        ├── img002.png
        └── ...

方式二：有正常图 + 缺陷图（可选 mask）
    my_dataset/
    train
    good
    test 
    good --一部分没训练的
    ng --全部的ng图
    ├── good/          # 正常图片
    │   ├── img001.png
    │   └── ...
    ├── defect/        # 缺陷图片（测试用）
    │   ├── img003.png
    │   └── ...
    └── mask/          # 缺陷 mask（可选，与 defect 同名对应）
        ├── img003.png
        └── ...
"""

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# ============================================================
# 1. 配置数据集
# ============================================================
datamodule = Folder(
    name="my_dataset",                # 数据集名称（用于日志）
    root=r"/hy-tmp/unsupervised/data3/data_no",       # ← 改成你的数据集根目录
    normal_dir="train/good",                # 正常图片所在子目录
    abnormal_dir=r"test/ng",            # defect  缺陷图片子目录（没有就设为 None）
    mask_dir="ground_truth/ng",                    # mask 目录（像素级标注，没有就设为 None）
    normal_test_dir=r"test/good",             # 单独的正常测试集目录（没有就设为 None）
    extensions=None,                  # 图片后缀过滤，如 (".png", ".jpg")，None 表示全部
    train_batch_size=1,               # EfficientAd 论文推荐 batch_size=1
    eval_batch_size=1,
    num_workers=0,                    # Windows 下必须设为 0，Linux 可设 4~8
    test_split_mode="from_dir",       # "from_dir" 从目录划分 / "synthetic" 合成测试集
    test_split_ratio=1,             # 没有单独测试目录时，从训练集抽取的比例
    val_split_mode="from_test",       # 验证集从测试集中切分
    val_split_ratio=1,              # 验证集占测试集的比例
    seed=42,
)

# ============================================================
# 2. 配置模型
# ============================================================
model = EfficientAd(
    imagenet_dir="./datasets/imagenette",  # ImageNette 数据集路径（首次自动下载）
    teacher_out_channels=384,              # teacher 输出通道数
    model_size="small",                    # 模型大小: "small" 或 "medium"
    lr=1e-4,                               # 学习率
    weight_decay=1e-5,                     # 权重衰减
    padding=False,                         # 是否使用 padding
    pad_maps=True,                         # 是否 pad 输出异常图
)

# ============================================================
# 3. 配置引擎并训练
# ============================================================
engine = Engine(
    max_epochs=70,                    # EfficientAd 推荐 70 个 epoch
    accelerator="auto",               # 自动选择 GPU/CPU
    devices=1,                        # 使用设备数
    default_root_dir="results",       # 结果保存目录
)

if __name__ == "__main__":
    # 训练
    engine.fit(model=model, datamodule=datamodule)

    # 测试（输出 AUROC 等指标）
    engine.test(model=model, datamodule=datamodule)

    # ============================================================
    # 4.（可选）单张图片推理
    # ============================================================
    # predictions = engine.predict(
    #     model=model,
    #     datamodule=datamodule,
    # )

    # ============================================================
    # 5.（可选）导出模型
    # ============================================================
    # from anomalib.deploy import ExportType
    # engine.export(model=model, export_type=ExportType.OPENVINO)
    # engine.export(model=model, export_type=ExportType.ONNX)
