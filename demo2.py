## Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import argparse
import os

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask, load_mask

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="", help="Path to input RGBA image")
parser.add_argument("--output", type=str, default="", help="Path to output GLB")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# load model
tag = "hf"
inputFileName = "./fridge_frenchdoors_hires.jpg"
outputFileName = ""

# derive base filename (no directory, no extension) from inputFileName
base_name = os.path.splitext(os.path.basename(inputFileName))[0]

# default mask and output names derived from the input base name
maskFileName = f"./{base_name}_mask.png"
splatFile = f"{base_name}_splat.ply"
glbFile = f"{base_name}.glb"

config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# if not args.image:
#    print("No input image provided")
#    exit(1)
    
# load image (RGBA only, mask is embedded in the alpha channel)
print("loading main image")

image = load_image(inputFileName)

print("loading mask")

mask = load_mask(maskFileName)


# run model
output = inference(image, mask, seed=77)

output["glb"].export(glbFile)
print(f"Your reconstruction has been saved to {glbFile}")
