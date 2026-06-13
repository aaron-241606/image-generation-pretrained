"""
Task-02: Image Generation using Stable Diffusion (HuggingFace Diffusers)
Prodigy Infotech Internship
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os


def generate_image(
    prompt,
    output_path="output.png",
    model_id="runwayml/stable-diffusion-v1-5",
    negative_prompt="blurry, bad quality, distorted, ugly",
    num_inference_steps=50,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=None,
):
    """Generate an image from a text prompt using Stable Diffusion."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"🖥️  Using device: {device}")
    print(f"📦 Loading model: {model_id}")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # Reduces memory usage

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
        print(f"🌱 Seed: {seed}")

    print(f"\n🎨 Generating image for prompt:\n   \"{prompt}\"\n")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    image = result.images[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image using Stable Diffusion")
    parser.add_argument("--prompt",   type=str, required=True,          help="Text prompt for image generation")
    parser.add_argument("--output",   type=str, default="output.png",   help="Output image path")
    parser.add_argument("--model",    type=str, default="runwayml/stable-diffusion-v1-5", help="HuggingFace model ID")
    parser.add_argument("--steps",    type=int, default=50,             help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,          help="Guidance scale")
    parser.add_argument("--width",    type=int, default=512,            help="Image width")
    parser.add_argument("--height",   type=int, default=512,            help="Image height")
    parser.add_argument("--seed",     type=int, default=None,           help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        output_path=args.output,
        model_id=args.model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
