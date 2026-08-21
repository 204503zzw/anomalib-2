# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

# """Example showing how to use the SuperADD model.

# SuperADD is an unsupervised (one-class) memory-based model that extracts
# multi-layer token features from a pretrained DINOv3 backbone over overlapping
# image patches and detects anomalies by nearest-neighbor search. Only normal
# images are needed for training.
# """
# from anomalib.data import Folder
# from anomalib.data import MVTecAD2
# from anomalib.engine import Engine
# from anomalib.models import SuperADD

# # 1. Basic Usage
# # Initialize with default settings
# model = SuperADD()

# # 2. Custom Configuration
# # Configure model parameters
# model = SuperADD(
#     backbone="vit_huge_plus_patch16_dinov3",  # DINOv3 feature extraction backbone
#     patch_size=320,  # Side length of the overlapping patches
#     patch_overlap=16,  # Overlap between neighboring patches
#     precision="float16",  # Halves the memory footprint of the backbone
#     task="segmentation",
# )

# # 3. Training Pipeline
# # Set up the complete training pipeline
# datamodule = Folder(
#     #root=ROOT_PATH,
#     root=r"/hy-tmp/unsupervised/data3/data_no",       # ← 改成你的数据集根目录
#     normal_dir="train/good",                # 正常图片所在子目录
#     abnormal_dir=r"test/ng",            # defect  缺陷图片子目录（没有就设为 None）
#     mask_dir="ground_truth/ng",                    # mask 目录（像素级标注，没有就设为 None）
#     normal_test_dir=r"test/good",             # 单独的正常测试集目录（没有就设为 None）
#     #image_size=IMAGE_SIZE,
#     train_batch_size=16,
#     eval_batch_size=4,
#     num_workers=0,
#     # Padding 常数填充255，对应实验3
#     resize_mode="pad",
#     padding_value=255
# )

# # Initialize training engine with specific settings
# engine = Engine(
#     max_epochs=1,  # SuperADD needs only one pass over the normal images
#     accelerator="auto",  # Automatically detect GPU/CPU
#     devices=1,  # Only a single device is supported
# )

# if __name__ == "__main__":
#     # Build the memory bank from the normal training images
#     engine.fit(
#         model=model,
#         datamodule=datamodule,
#     )

#     # Predict anomaly maps and scores
#     predictions = engine.predict(
#         model=model,
#         datamodule=datamodule,
#     )
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import SuperADD
from anomalib.models.image.super_add.post_processor import SuperADDPostProcessor
from torchvision.transforms.v2 import ColorJitter

# ============================================================
# 1. 数据集
# ============================================================
datamodule = Folder(
    name="superadd_computer",
    root=r"/hy-tmp/unsupervised/anomalib-main-2/examples/api/01_getting_started/roi/crops16/roi",
    normal_dir="good",
    abnormal_dir="ng",
    normal_test_dir= None,#"test/good",
    # mask_dir="ground_truth/ng",          # 有像素级标注就打开
    train_batch_size=1,                    # 3072 必须用 1
    eval_batch_size=1,
    num_workers=4,
    test_split_mode="from_dir",
    test_split_ratio=0,
    val_split_mode="same_as_test",
    seed=42,
    train_augmentations=ColorJitter(brightness=(0.8, 1.2)),  # 论文推荐
)

# ============================================================
# 2. SuperADD（针对 3072 的推荐参数）
# ============================================================
model = SuperADD(
    patch_size=640,              # 论文官方推荐值（优先使用）
    patch_overlap=128,           # 论文官方推荐值
    # 备选（显存不足时按顺序降级）：
    # patch_size=512, patch_overlap=128
    # patch_size=448, patch_overlap=64
    # patch_size=320, patch_overlap=64

    pre_processor=SuperADD.configure_pre_processor(
        image_size=(2688, 1856)  # 保持原图分辨率，不要下采样
    ),
    post_processor=SuperADDPostProcessor(),  # 推荐显式使用
    gaussian_blur_sigma=3.0,     # 小目标（螺丝）建议比默认 4.0 稍低
    score_quantile=3e-3,         # 小目标可适当放大，让图像级分数更关注局部高响应
)

# ============================================================
# 3. Engine
# ============================================================
engine = Engine(
    max_epochs=1,
    accelerator="auto",
    devices=1,                   # SuperADD 目前只支持单卡
    default_root_dir="results",
)

# ============================================================
# 4. 训练 + 测试
# ============================================================
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
