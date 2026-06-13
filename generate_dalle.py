"""
Task-02: Image Generation using DALL-E (OpenAI API)
Prodigy Infotech Internship
"""

import argparse
import os
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO


def generate_dalle(
    prompt,
    output_path="output.png",
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
):
    """Generate an image from a text prompt using DALL-E."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    print(f"🎨 Generating image with {model}...")
    print(f"   Prompt : \"{prompt}\"")
    print(f"   Size   : {size}")
    print(f"   Style  : {style}\n")

    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        n=1,
    )

    image_url = response.data[0].url
    print(f"🔗 Image URL: {image_url}\n")

    # Download and save
    img_data = requests.get(image_url).content
    image = Image.open(BytesIO(img_data))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image using DALL-E")
    parser.add_argument("--prompt",  type=str, required=True,        help="Text prompt for image generation")
    parser.add_argument("--output",  type=str, default="output.png", help="Output image path")
    parser.add_argument("--model",   type=str, default="dall-e-3",   help="DALL-E model version")
    parser.add_argument("--size",    type=str, default="1024x1024",  help="Image size (1024x1024, 1792x1024, 1024x1792)")
    parser.add_argument("--quality", type=str, default="standard",   help="Image quality: standard or hd")
    parser.add_argument("--style",   type=str, default="vivid",      help="Image style: vivid or natural")
    args = parser.parse_args()

    generate_dalle(
        prompt=args.prompt,
        output_path=args.output,
        model=args.model,
        size=args.size,
        quality=args.quality,
        style=args.style,
    )
