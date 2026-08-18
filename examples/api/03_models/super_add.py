# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example showing how to use the SuperADD model.

SuperADD is an unsupervised (one-class) memory-based model that extracts
multi-layer token features from a pretrained DINOv3 backbone over overlapping
image patches and detects anomalies by nearest-neighbor search. Only normal
images are needed for training.
"""

from anomalib.data import MVTecAD2
from anomalib.engine import Engine
from anomalib.models import SuperADD

# 1. Basic Usage
# Initialize with default settings
model = SuperADD()

# 2. Custom Configuration
# Configure model parameters
model = SuperADD(
    backbone="vit_huge_plus_patch16_dinov3",  # DINOv3 feature extraction backbone
    patch_size=448,  # Side length of the overlapping patches
    patch_overlap=16,  # Overlap between neighboring patches
    precision="float16",  # Halves the memory footprint of the backbone
)

# 3. Training Pipeline
# Set up the complete training pipeline
datamodule = MVTecAD2(
    root="./datasets/MVTec_AD_2",
    category="sheet_metal",
    train_batch_size=1,  # High resolution features require little batching
    eval_batch_size=1,
)

# Initialize training engine with specific settings
engine = Engine(
    max_epochs=1,  # SuperADD needs only one pass over the normal images
    accelerator="auto",  # Automatically detect GPU/CPU
    devices=1,  # Only a single device is supported
)

if __name__ == "__main__":
    # Build the memory bank from the normal training images
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    # Predict anomaly maps and scores
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
    )
