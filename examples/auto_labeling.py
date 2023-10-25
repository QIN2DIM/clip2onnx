# -*- coding: utf-8 -*-
# Time       : 2023/10/24 5:39
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from PIL import Image
# pip install hcaptcha_challenger==0.9.0
from hcaptcha_challenger import (
    DataLake,
    install,
    ModelHub,
    ZeroShotImageClassifier,
    register_pipline,
)
from tqdm import tqdm

install(upgrade=True)

assets_dir = Path(__file__).parent.parent.joinpath("assets")
model_dir = Path(__file__).parent.parent.joinpath("model")
images_dir = assets_dir.joinpath("off_road_vehicle")


def flush_env():
    __formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y%m%d%H%M")
    now = datetime.strptime(str(datetime.now()), __formats[0]).strftime(__formats[1])
    yes_dir = images_dir.joinpath(now, "yes")
    bad_dir = images_dir.joinpath(now, "bad")
    for cd in [yes_dir, bad_dir]:
        shutil.rmtree(cd, ignore_errors=True)
        cd.mkdir(parents=True, exist_ok=True)

    return yes_dir, bad_dir


def auto_labeling(
        fmt: Literal["onnx", "transformers"] = None,
        visual_path: Path | None = None,
        textual_path: Path | None = None,
        **kwargs,
):
    """
    Demonstrates how to read onnx models and run them (without relying on torches)

    :param fmt:
        (Default to None)
        IF fmt in ["onnx"]:
            Load ONNX model and complete inference task based on CPU. The whole process does not rely on torch.
        ELSE:
            - If your environment has 'transformers' and 'torch' installed and the cuda graphics card is available,
            it will automatically switch to GPU mode.
            - But this situation is not within the scope of our demonstration, only for comparison.
            - At this time, the program will pull the preset pipline model from the huggingface to
            perform the zero-shot image classification task.

            That is, the `visual_path` and `textual_path` parameters do not take effect at this time.

    :param visual_path:
        (Default to None) Path to visual ONNX model.
        If not set, the program pulls the default ONNX model from the GitHub repository

    :param textual_path:
        (Default to None) Path to textual ONNX model.
        If not set, the program pulls the default ONNX model from the GitHub repository

    :return:
    """
    modelhub = ModelHub.from_github_repo()
    modelhub.parse_objects()

    # Refresh experiment environment
    yes_dir, bad_dir = flush_env()

    # Prompt: "Please click each image containing an off-road vehicle"
    data_lake = DataLake.from_prompts(
        positive_labels=["off-road vehicle"], negative_labels=["bicycle", "car"]
    )

    # Parse DataLake and build the model pipline
    tool = ZeroShotImageClassifier.from_datalake(data_lake)
    model = register_pipline(
        modelhub, fmt=fmt, visual_path=visual_path, textual_path=textual_path, **kwargs
    )

    total = len(os.listdir(images_dir))
    with tqdm(total=total, desc=f"Labeling | {images_dir.name}") as progress:
        for image_name in os.listdir(images_dir):
            image_path = images_dir.joinpath(image_name)
            if not image_path.is_file():
                progress.total -= 1
                continue

            # we're only dealing with binary classification tasks here
            # The label at position 0 is the highest scoring target
            results = tool(model, Image.open(image_path))
            if results[0]["label"] in data_lake.positive_labels:
                output_path = yes_dir.joinpath(image_name)
            else:
                output_path = bad_dir.joinpath(image_name)
            shutil.copyfile(image_path, output_path)

            progress.update(1)

    if "win32" in sys.platform:
        os.startfile(images_dir)


def run():
    # 1. Use the model you just converted to ONNX format
    # auto_labeling(
    #     fmt="onnx",
    #     textual_path=model_dir.joinpath("path/to/textual_model"),
    #     visual_path=model_dir.joinpath("path/to/visual_model")
    # )

    # 2. Using automatic mode, you can see the demo running effect first.
    auto_labeling()

    # 3. Using CUDA mode, You need to install `transformers`. For comparison only
    # pip install -U transformers
    # auto_labeling(fmt="transformers")


if __name__ == "__main__":
    run()
