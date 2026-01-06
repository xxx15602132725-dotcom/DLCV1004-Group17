from __future__ import annotations
from typing import List
import numpy as np
import torch

def z_from_seed(G, seed: int) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    z_vec = rng.randn(1, G.z_dim).astype(np.float32)
    device = next(G.parameters()).device
    return torch.from_numpy(z_vec).to(device)

def w_from_z(G, z: torch.Tensor, truncation_psi: float = 0.7) -> torch.Tensor:
    return G.mapping(z, None, truncation_psi=truncation_psi)

def _linear_interpolate(v1: torch.Tensor, v2: torch.Tensor, steps: int) -> List[torch.Tensor]:
    if steps < 2:
        raise ValueError("Interpolation requires at least 2 steps.")
    
    results = []
    for i in range(steps):
        ratio = i / (steps - 1)
        interpolated = v1 * (1 - ratio) + v2 * ratio
        results.append(interpolated)
    return results

def interpolate_z(z_a: torch.Tensor, z_b: torch.Tensor, steps: int) -> List[torch.Tensor]:
    return _linear_interpolate(z_a, z_b, steps)

def interpolate_w(w_a: torch.Tensor, w_b: torch.Tensor, steps: int) -> List[torch.Tensor]:
    return _linear_interpolate(w_a, w_b, steps)

def parse_seed_range(seed_str: str) -> List[int]:
    unique_seeds = []
    seen = set()
    
    fragments = [s.strip() for s in seed_str.split(",") if s.strip()]
    
    for item in fragments:
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            if start > end:
                start, end = end, start
            batch = range(start, end + 1)
            for s in batch:
                if s not in seen:
                    unique_seeds.append(s)
                    seen.add(s)
        else:
            val = int(item)
            if val not in seen:
                unique_seeds.append(val)
                seen.add(val)
                
    return unique_seeds