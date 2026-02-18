# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch


# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None, help="Path to input RGBA image")
parser.add_argument("--output", type=str, default=None, help="Path to output GLB")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

if not args.image:
    print("No input image provided")
    exit(1)


# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image(args.image)
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)


# run model
output = inference(image, mask, seed=42)

print("here are the output types:\n", output.keys())

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
output["glb"].export("scene.glb")
print("Your reconstruction has been saved to splat.ply")
