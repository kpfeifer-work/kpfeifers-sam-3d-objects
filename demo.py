# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

print("HI THERE")

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

print("LOADING INFERENCE")

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)


print("LOADING IMAGE")

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

print("RUNNING INFERENCE")

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
output["glb"].export("splat.glb")
print("Your reconstruction has been saved to splat.ply")
