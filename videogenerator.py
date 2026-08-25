#!/usr/bin/env python3
"""
videogenerator-cli

Generates AI videos using Wan2.1/HunyuanVideo diffusers pipelines,
then uses a local Ollama model (GLM4:9B) to auto-label the output file.

Requirements:
    pip install diffusers transformers accelerate torch imageio[ffmpeg]
    ollama pull glm4:9b
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# --- config ---
DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_OLLAMA = "glm4:9b"
DEFAULT_FRAMES = 81
DEFAULT_STEPS = 30
DEFAULT_FPS = 16
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
OLLAMA_API = "http://localhost:11434"


# --- ollama helpers ---

def ollama_running() -> bool:
    """Check if ollama is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as _:
            return True
    except (urllib.error.URLError, OSError):
        return False


def ollama_label(prompt: str, model: str, date_str: str) -> str:
    """Ask ollama to generate a filename label from the prompt and date."""
    system = (
        "You are a filename generator. The user will give you a video prompt and a date. "
        "Respond with ONLY a short descriptive filename label (no extension, no slashes, "
        "no special characters, lowercase, spaces replaced with dashes, max 60 chars). "
        "Include the date at the end in YYYY-MM-DD format. "
        "Do not include any explanation or extra text."
    )
    user_msg = f"Prompt: {prompt}\nDate: {date_str}"

    payload = json.dumps({
        "model": model,
        "system": system,
        "prompt": user_msg,
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_API}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            label = result.get("response", "").strip()
    except Exception as e:
        print(f"  [warn] ollama labeling failed: {e}, using fallback")
        label = prompt[:50].lower().replace(" ", "-")

    # sanitize
    label = label.replace("/", "-").replace("\\", "-").replace(" ", "-")
    label = "".join(c for c in label if c.isalnum() or c in "-_.")
    return label[:80] or "video"


# --- video generation ---

def generate_video(
    prompt: str,
    model: str,
    output_dir: str,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    width: int,
    height: int,
    seed: int | None,
    ollama_model: str,
    label: bool,
) -> str:
    """Generate a video and return the output path."""
    import torch
    from diffusers import WanPipeline

    print(f"[1/3] loading model: {model}")
    pipe = WanPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe.to("cuda")

    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(seed)

    print(f"[2/3] generating video ({num_frames} frames, {num_inference_steps} steps)...")
    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
        width=width,
        height=height,
    )
    frames = result.frames[0]

    # --- filename ---
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    if label:
        print(f"[3/3] labeling with ollama ({ollama_model})...")
        slug = ollama_label(prompt, ollama_model, date_str)
    else:
        slug = prompt[:50].lower().replace(" ", "-")[:60]

    out_path = Path(output_dir) / f"{slug}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] encoding {len(frames)} frames -> {out_path}")
    import imageio.v2 as iio
    writer = iio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    for frame in frames:
        # frame is (H, W, 3) numpy uint8 or float
        import numpy as np
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()

    return str(out_path)


# --- main ---

def main():
    parser = argparse.ArgumentParser(
        prog="videogenerator",
        description="Generate AI videos with auto-labeling via Ollama",
    )
    parser.add_argument("prompt", help="text prompt describing the video")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"diffusers model id (default: {DEFAULT_MODEL})")
    parser.add_argument("-o", "--output", default="./output",
                        help="output directory (default: ./output)")
    parser.add_argument("-f", "--frames", type=int, default=DEFAULT_FRAMES,
                        help=f"number of frames (default: {DEFAULT_FRAMES})")
    parser.add_argument("-s", "--steps", type=int, default=DEFAULT_STEPS,
                        help=f"inference steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"output fps (default: {DEFAULT_FPS})")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"frame width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help=f"frame height (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducibility")
    parser.add_argument("--ollama", default=DEFAULT_OLLAMA,
                        help=f"ollama model for labeling (default: {DEFAULT_OLLAMA})")
    parser.add_argument("--no-label", action="store_true",
                        help="skip ollama labeling, use raw prompt slug")

    args = parser.parse_args()

    # --- checks ---
    use_label = not args.no_label
    if use_label and not ollama_running():
        print(f"[warn] ollama not running at {OLLAMA_API}")
        print("  either start ollama or use --no-label")
        sys.exit(1)

    try:
        import torch
        if not torch.cuda.is_available():
            print("[error] no cuda GPU detected. video generation requires an NVIDIA GPU.")
            sys.exit(1)
    except ImportError:
        print("[error] pytorch not installed. run: pip install torch")
        sys.exit(1)

    try:
        from diffusers import WanPipeline
    except ImportError:
        print("[error] diffusers not installed. run: pip install diffusers transformers accelerate")
        sys.exit(1)

    out = generate_video(
        prompt=args.prompt,
        model=args.model,
        output_dir=args.output,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        ollama_model=args.ollama,
        label=use_label,
    )

    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
