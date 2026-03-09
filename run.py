# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("2D to 3D test data/cab1.jpeg")
mask = None #load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

print("here are the output types:\n", output.keys())

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
output["glb"].export("scene.glb")
print("Your reconstruction has been saved to splat.ply")
