[中文文档](README_zh.md)

# videogenerator-cli

AI video generation from the terminal. Generates videos with Wan2.1 or HunyuanVideo, then uses a local [Ollama](https://ollama.com) model (default: GLM4 9B) to auto-label the output file based on your prompt and the current date.

## how it works

1. you give it a text prompt
2. (optional) Ollama picks optimal frames, steps, fps, and resolution based on your prompt
3. it generates a video using a diffusers pipeline (Wan2.1 by default)
4. it sends the prompt + current date to your local Ollama instance
5. Ollama generates a clean filename label
6. the video is saved as `<label-2026-08-25>.mp4`

## requirements

- python 3.10+
- NVIDIA GPU with CUDA (minimum ~2GB VRAM for Wan2.1 1.3B, ~5GB for HunyuanVideo)
- [Ollama](https://ollama.com) running locally

## install

```bash
# python deps
pip install diffusers transformers accelerate torch imageio[ffmpeg]

# ollama model for labeling
ollama pull glm4:9b

# clone
git clone https://github.com/creepernet-on-gh/videogenerator-cli.git
cd videogenerator-cli
```

## usage

```bash
# basic - generates video and auto-labels it
python videogenerator.py "a cat walking on a piano keyboard"

# custom model (HunyuanVideo 1.5, needs ~5GB VRAM)
python videogenerator.py "underwater coral reef" -m HunyuanVideo/HunyuanVideo-1.5-Diffusers

# custom resolution, frames, steps
python videogenerator.py "sunset over mountains" --width 1280 --height 720 --frames 161 --steps 50

# reproducible with seed
python videogenerator.py "rain on a window" --seed 42

# let ollama pick settings based on your prompt
python videogenerator.py "a slow drone shot over a foggy forest at dawn" --let-ollama-select

# skip ollama labeling (uses raw prompt as filename)
python videogenerator.py "abstract neon lights" --no-label

# custom output directory
python videogenerator.py "snowy village" -o ~/videos

# different ollama model for labeling
python videogenerator.py "cherry blossoms" --ollama llama3.1:8b
```

## all options

| flag | default | description |
|------|---------|-------------|
| `prompt` | (required) | text prompt describing the video |
| `-m, --model` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | diffusers model ID |
| `-o, --output` | `./output` | output directory |
| `-f, --frames` | `41` | number of frames to generate |
| `-s, --steps` | `30` | inference steps |
| `--fps` | `16` | output video FPS |
| `--width` | `832` | frame width in pixels |
| `--height` | `480` | frame height in pixels |
| `--seed` | random | random seed for reproducibility |
| `--ollama` | `glm4:9b` | ollama model for filename labeling |
| `--let-ollama-select` | off | let ollama choose frames/steps/fps/width/height |
| `--no-label` | off | skip ollama, use prompt slug as filename |

## supported models

| model | params | VRAM | grade |
|-------|--------|------|-------|
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | 1.3B | ~1.2 GB | S |
| Wan-AI/Wan2.2-T2V-5B-Diffusers | 5B | ~3.1 GB | S |
| HunyuanVideo/HunyuanVideo-1.5-Diffusers | 8.3B | ~4.8 GB | A |

## license

MIT
