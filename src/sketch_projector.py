import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Import custom utilities
# Note: Must run from root directory to ensure src is found
from .gan_loader import load_generator
from .loss_utils import detect_edges

"""
Sketch Projector
Function: Projects user input line sketches into the StyleGAN2 latent space to generate structurally corresponding artistic images.
Usage (Command Line):
    conda activate stylegan
    python -m src.sketch_projector --network pretrained/wikiart_pytorch.pkl --sketch input_sketches/2.jpg --outdir output_art/test2 --cls 10
    python -m src.sketch_projector --network pretrained/wikiart_pytorch.pkl --sketch input_sketches/3.jpg --outdir output_art/class030 --edge_weight 10.0 --reg_weight 3.0 --steps 2000 --lr 0.05 --cls 30 --seed 42

# Higher parameters imply stronger constraints
Core Parameters:
    --network: Path to pre-trained model (.pkl)
    --sketch:  Path to sketch image (Recommended white background with black lines)
    --cls:     (Optional) Specific WikiArt class index (0-166), random if not specified
    --steps:   Optimization steps (Default 1000)
"""

def project_sketch(
        G,
        sketch_path,
        outdir,
        num_steps=2000,
        lr=0.05,
        lambda_edge=10.0,  # Edge Loss weight: Higher values enforce stricter structure
        lambda_reg=3.0,    # Regularization weight: Prevents image collapse/artifacts
        seed=42,
        class_idx=None     # Specific WikiArt class index (0-166), random if None
):
    """
    Main Projection Logic Loop
    """
    device = torch.device('cuda')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------
    # 1. Preprocess Sketch
    # -------------------------------------------
    print(f"Loading sketch: {sketch_path}")
    target_pil = Image.open(sketch_path).convert('L')  # Force convert to grayscale

    # Resize to match model resolution (512x512)
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)

    # Convert to Tensor: [1, 1, H, W], Range [0, 1]
    target_tensor = torch.from_numpy(np.array(target_pil)).float() / 255.0
    target_tensor = target_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # Assume sketch is white background/black lines, invert to black background/white lines
    # (Since edge detection operators respond to high intensity values)
    if target_tensor.mean() > 0.5:
        print("Inverting sketch colors (assuming white background)...")
        target_tensor = 1.0 - target_tensor

    # Save processed sketch edge reference
    ref_img = (target_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(ref_img).save(outdir / "target_edge_reference.png")

    # -------------------------------------------
    # 2. Initialize Latent Code
    # -------------------------------------------
    print("Preparing latent code...")

    # Handle class label (c)
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            # If not specified, select a random style
            class_idx = np.random.randint(G.c_dim)
            print(f"Randomly selected class index: {class_idx}")
        else:
            print(f"Using specified class index: {class_idx}")
        label[:, class_idx] = 1

    # Generate a batch of random z, calculate average w (w_avg) as optimization starting point
    # This ensures the generation starts from a valid artistic style rather than noise
    z_samples = np.random.RandomState(seed).randn(10000, G.z_dim)
    z_tensor = torch.from_numpy(z_samples).to(device)
    # For Conditional models, mapping requires labels
    label_batch = label.repeat(10000, 1)
    w_samples = G.mapping(z_tensor, label_batch)
    w_avg = torch.mean(w_samples, dim=0, keepdim=True)

    # w_opt is the core variable to optimize (Latent Code)
    w_opt = w_avg.detach().clone()
    w_opt.requires_grad = True

    optimizer = torch.optim.Adam([w_opt], lr=lr)

    # -------------------------------------------
    # 3. Optimization Loop
    # -------------------------------------------
    print(f"Projecting... (Steps: {num_steps})")
    pbar = tqdm(range(num_steps))

    for step in pbar:
        # (A) Generate Image
        # noise_mode='const' ensures determinism, avoiding noise interference with edge detection
        synth_img = G.synthesis(w_opt, noise_mode='const')

        # (B) Extract Edges from Generated Image
        synth_edge = detect_edges(synth_img)

        # (C) Calculate Loss
        # Edge Loss: Force generated edges to match sketch edges
        loss_edge = F.mse_loss(synth_edge, target_tensor)

        # Regularization Loss: Prevent w from drifting too far (manifold escape)
        loss_reg = torch.mean((w_opt - w_avg) ** 2)

        # Total Loss
        total_loss = lambda_edge * loss_edge + lambda_reg * loss_reg

        # (D) Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_description(f"Edge Loss: {loss_edge.item():.4f}")

        # (E) Periodically save intermediate results
        if step % 200 == 0 or step == num_steps - 1:
            with torch.no_grad():
                # Convert back to RGB [0, 255]
                img_gen = (synth_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                Image.fromarray(img_gen[0].cpu().numpy(), 'RGB').save(outdir / f"proj_step{step:04d}.png")

    print(f"Done! Final result saved to {outdir}")


if __name__ == "__main__":
    """
    Command Line Entry Point
    Example: python -m src.sketch_projector --network pretrained/wikiart.pkl --sketch input_sketches/tree.png --edge_weight 2.0 --reg_weight 0.2
    """
    parser = argparse.ArgumentParser()
    # Basic Parameters
    parser.add_argument('--network', type=str, required=True, help='Path to .pkl model')
    parser.add_argument('--sketch', type=str, required=True, help='Path to input sketch image')
    parser.add_argument('--outdir', type=str, default='output_art', help='Directory to save results')

    # Style and Randomness
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cls', type=int, default=None, help='Specific class index (optional)')

    # [Core Adjustment Area] New parameters added here
    parser.add_argument('--steps', type=int, default=1000, help='Optimization steps')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--edge_weight', type=float, default=5.0, help='Structural constraint weight (Recommended 2.0 - 10.0)')
    parser.add_argument('--reg_weight', type=float, default=0.1, help='Style constraint weight (Recommended 0.05 - 0.5)')

    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading network from: {args.network}")
    G = load_generator(args.network)

    print(f"Start Projecting with: Edge_Weight={args.edge_weight}, Reg_Weight={args.reg_weight}, Steps={args.steps}")

    # 2. Execute Projection (Pass command line args to function)
    project_sketch(
        G,
        args.sketch,
        args.outdir,
        num_steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        class_idx=args.cls,
        lambda_edge=args.edge_weight,  # Connected to structure constraint
        lambda_reg=args.reg_weight     # Connected to style constraint
    )