# Task-02: Image Generation with Pre-trained Models

> Utilize pre-trained generative models like DALL-E-mini or Stable Diffusion to create images from text prompts.

---

## 📌 Overview

This project demonstrates text-to-image generation using two approaches:
- **Stable Diffusion** via HuggingFace `diffusers` library (local)
- **DALL-E via OpenAI API** (cloud-based)

---

## 🗂️ Project Structure

```
image-generation-pretrained/
├── generate_sd.py         # Stable Diffusion image generation
├── generate_dalle.py      # DALL-E image generation via OpenAI API
├── batch_generate.py      # Generate multiple images from a prompt list
├── prompts.txt            # Sample text prompts
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/aaron-241606/image-generation-pretrained.git
cd image-generation-pretrained
pip install -r requirements.txt
```

---

## 🚀 Usage

### Stable Diffusion (local)
```bash
python generate_sd.py --prompt "a sunset over mountain peaks, oil painting style" --output output.png
```

### DALL-E (OpenAI API)
```bash
export OPENAI_API_KEY="your-api-key"
python generate_dalle.py --prompt "a futuristic city at night with neon lights" --output output.png
```

### Batch generation
```bash
python batch_generate.py --prompts prompts.txt --output_dir ./outputs
```

---

## 🧠 Model Details

| Model             | Type              | Backend         | Resolution  |
|-------------------|-------------------|-----------------|-------------|
| Stable Diffusion  | Diffusion Model   | HuggingFace     | 512×512     |
| DALL-E 3          | Transformer-based | OpenAI API      | 1024×1024   |

---

## 📚 References

- [#1 Stable Diffusion – CompVis](https://github.com/CompVis/stable-diffusion)
- [#2 HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- [#3 OpenAI DALL-E API](https://platform.openai.com/docs/guides/images)
- [#4 High-Resolution Image Synthesis Paper](https://arxiv.org/abs/2112.10752)

---

## 🏢 Credits

**Prodigy Infotech** – Task-02 Internship Project
