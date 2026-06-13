"""
Task-02: Batch image generation from a list of prompts
Prodigy Infotech Internship
"""

import argparse
import os
import torch
from diffusers import StableDiffusionPipeline
from generate_sd import generate_image


def batch_generate(prompts_file, output_dir, steps=30, guidance=7.5):
    """Generate images for every prompt in a text file."""

    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"📋 Found {len(prompts)} prompts in {prompts_file}")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    pipe   = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
    ).to(device)
    pipe.enable_attention_slicing()

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: \"{prompt}\"")
        filename = f"{i:02d}_{prompt[:40].replace(' ', '_').replace('/', '-')}.png"
        out_path = os.path.join(output_dir, filename)

        result = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance)
        result.images[0].save(out_path)
        print(f"   ✅ Saved: {out_path}")

    print(f"\n🎉 Done! {len(prompts)} images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate images from a prompts file")
    parser.add_argument("--prompts",    type=str, default="prompts.txt", help="Path to prompts text file")
    parser.add_argument("--output_dir", type=str, default="./outputs",   help="Directory to save generated images")
    parser.add_argument("--steps",      type=int, default=30,            help="Inference steps per image")
    parser.add_argument("--guidance",   type=float, default=7.5,         help="Guidance scale")
    args = parser.parse_args()

    batch_generate(args.prompts, args.output_dir, args.steps, args.guidance)
