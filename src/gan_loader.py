from __future__ import annotations
import os
import sys
from pathlib import Path
import torch


class PathConfigurator:
    @staticmethod
    def inject_dependencies():
        try:
            import dnnlib
            import legacy
            return
        except ImportError:
            pass

        search_paths = []
        env_var = os.environ.get("STYLEGAN2_ADA_PATH")
        if env_var:
            search_paths.append(Path(env_var))

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]

        candidates = [
            project_root / "stylegan2",
            project_root / "stylegan2-ada-pytorch",
            project_root.parent / "stylegan2",
            project_root.parent / "stylegan2-ada-pytorch",
            project_root.parent / "stylegan2-ada-pytorch-main",
        ]
        search_paths.extend(candidates)

        for path in search_paths:
            if path.exists() and (path / "dnnlib").exists():
                sys.path.append(str(path))
                return

        raise ImportError("Failed to locate StyleGAN2-ADA-PyTorch core libraries.")


class ModelLoader:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    def load(self, pkl_path: str, verbose: bool = True):
        PathConfigurator.inject_dependencies()
        import dnnlib
        import legacy

        if verbose:
            print(f"Loading network source: {pkl_path}")
            print(f"Compute device: {self.device}")

        with dnnlib.util.open_url(pkl_path) as stream:
            network_snapshot = legacy.load_network_pkl(stream)
            generator = network_snapshot["G_ema"].to(self.device)

        generator.eval()
        return generator


def load_generator(network_pkl: str, device: str = "cuda", verbose: bool = True):
    loader = ModelLoader(device)
    return loader.load(network_pkl, verbose)