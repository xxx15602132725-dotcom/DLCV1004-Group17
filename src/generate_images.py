from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List
import torch
import numpy as np
from PIL import Image
from .gan_loader import load_generator
from .sample_latents import (
    parse_seed_range,
    z_from_seed,
    w_from_z,
    interpolate_z,
    interpolate_w,
)


class StyleGen:
    def __init__(self, network_path: str, device: str = "cuda"):
        self.device = device
        self.G = load_generator(network_path, device=device, verbose=True)

    @staticmethod
    def _postprocess(tensor_img: torch.Tensor) -> Image.Image:
        processed = (tensor_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(processed[0].cpu().numpy(), "RGB")

    @torch.no_grad()
    def render_image(self, seed: int, psi: float = 0.7, noise: str = "const") -> Image.Image:
        z = z_from_seed(self.G, seed)
        w = w_from_z(self.G, z, truncation_psi=psi)
        out = self.G.synthesis(w, noise_mode=noise)
        return self._postprocess(out)

    @torch.no_grad()
    def render_interpolation(self, seed_a: int, seed_b: int, steps: int, psi: float, noise: str, space: str) -> List[
        Image.Image]:
        z1 = z_from_seed(self.G, seed_a)
        z2 = z_from_seed(self.G, seed_b)

        output_frames = []

        if space.lower() == "w":
            w1 = w_from_z(self.G, z1, truncation_psi=psi)
            w2 = w_from_z(self.G, z2, truncation_psi=psi)
            vectors = interpolate_w(w1, w2, steps)
            for vec in vectors:
                res = self.G.synthesis(vec, noise_mode=noise)
                output_frames.append(self._postprocess(res))
        else:
            vectors = interpolate_z(z1, z2, steps)
            for vec in vectors:
                w_mapped = w_from_z(self.G, vec, truncation_psi=psi)
                res = self.G.synthesis(w_mapped, noise_mode=noise)
                output_frames.append(self._postprocess(res))

        return output_frames


def save_gif_file(images: List[Image.Image], path: Path, duration: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0,
    )


def get_parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="StyleGAN2-ADA Generation Tool")

    cli.add_argument("--network", required=True, help="Network pickle path")
    cli.add_argument("--device", default="cuda", help="Compute device")
    cli.add_argument("--outdir", default="out", help="Output root directory")
    cli.add_argument("--seeds", default=None, help="Seed range (e.g., 100-103)")
    cli.add_argument("--trunc", type=float, default=0.7, help="Truncation psi")
    cli.add_argument("--noise", default="const", choices=["const", "random", "none"], help="Noise mode")
    cli.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg"], help="Output format")

    cli.add_argument("--gif", action="store_true", help="Enable GIF mode")
    cli.add_argument("--seed-a", type=int, default=100, help="Start seed for GIF")
    cli.add_argument("--seed-b", type=int, default=200, help="End seed for GIF")
    cli.add_argument("--steps", type=int, default=60, help="Total frames")
    cli.add_argument("--space", default="z", choices=["z", "w"], help="Interpolation space")
    cli.add_argument("--gif-name", default="interpolation.gif", help="GIF filename")
    cli.add_argument("--gif-duration", type=int, default=50, help="Frame delay (ms)")

    return cli


def run_task():
    parser = get_parser()
    opt = parser.parse_args()

    output_path = Path(opt.outdir)
    output_path.mkdir(parents=True, exist_ok=True)

    engine = StyleGen(opt.network, opt.device)

    if opt.gif:
        frames = engine.render_interpolation(
            seed_a=opt.seed_a,
            seed_b=opt.seed_b,
            steps=opt.steps,
            psi=opt.trunc,
            noise=opt.noise,
            space=opt.space
        )
        target_file = output_path / opt.gif_name
        save_gif_file(frames, target_file, opt.gif_duration)
        print(f"GIF saved successfully: {target_file}")

    else:
        if not opt.seeds:
            print("Error: --seeds argument is required for image generation mode.")
            sys.exit(1)

        seed_list = parse_seed_range(opt.seeds)
        for s in seed_list:
            result = engine.render_image(s, psi=opt.trunc, noise=opt.noise)
            fname = f"seed{s:04d}.{opt.ext}"
            full_path = output_path / fname
            result.save(full_path)
            print(f"Image saved: {full_path}")


if __name__ == "__main__":
    run_task()