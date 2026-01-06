import os
import argparse
import numpy as np
import torch
from PIL import Image
from src.gan_loader import load_generator

# python -m src.generate_random --network pretrained/wikiart_pytorch.pkl --outdir output_scan_classes --seeds 42 --cls 0-166

def parse_range(input_str):
    """Parse string in format '0-10,42' into a list of integers"""
    if input_str is None:
        return [None]

    indices = []
    for part in input_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Generate samples from StyleGAN2 (Batch & Scan Mode)")
    parser.add_argument('--network', type=str, required=True, help='Path to .pkl network')
    parser.add_argument('--outdir', type=str, default='output_scan', help='Output directory')
    # seeds and cls now support range input (e.g., 0-10)
    parser.add_argument('--seeds', type=str, default='42', help='Seeds (e.g., "42" or "0-5")')
    parser.add_argument('--cls', type=str, default=None, help='Class indices (e.g., "30" or "0-166"). If None, random.')
    parser.add_argument('--trunc', type=float, default=0.7, help='Truncation psi')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print(f"Loading network: {args.network}")
    G = load_generator(args.network)

    seeds = parse_range(args.seeds)
    classes = parse_range(args.cls) if args.cls else [None]

    print(f"Plan: {len(seeds)} Seeds x {len(classes)} Classes = {len(seeds) * len(classes)} Images")
    print("Starting generation...")

    count = 0
    for seed in seeds:
        # Fix random noise z (same seed results in same base composition)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        for c_idx in classes:
            # Construct class label
            label = torch.zeros([1, G.c_dim], device=device)
            if G.c_dim != 0:
                if c_idx is None:
                    final_c = np.random.randint(G.c_dim)
                    c_name = f"random{final_c}"
                else:
                    final_c = c_idx
                    c_name = f"class{final_c:03d}"  # Format as 001, 002...
                label[:, final_c] = 1
            else:
                c_name = "uncond"

            # Generate
            img = G(z, label, truncation_psi=args.trunc, noise_mode='const')

            # Convert to image format
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # Filename: easy for sorting and viewing
            # seed42_class000.png, seed42_class001.png ...
            filename = f"seed{seed:04d}_{c_name}.png"
            out_path = os.path.join(args.outdir, filename)
            Image.fromarray(img[0].cpu().numpy(), 'RGB').save(out_path)

            count += 1
            if count % 10 == 0:
                print(f"Generated {count} images...", end='\r')

    print(f"\nDone! {count} images saved to {args.outdir}")


if __name__ == "__main__":
    main()