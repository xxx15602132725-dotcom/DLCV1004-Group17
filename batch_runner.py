import os
from src.gan_loader import load_generator
from src.sketch_projector import project_sketch


def main():
    # ================= CONFIGURATION =================
    # 1. Input and Output Settings
    input_folder = "input_sketches"  # Directory containing input sketches
    base_outdir = "output_art/class04"  # Base output directory pattern

    # 2. Model Settings
    network_pkl = "pretrained/wikiart_pytorch.pkl"

    # 3. Hyperparameters
    params = {
        "num_steps": 2000,  # Optimization iterations
        "lr": 0.05,  # Learning rate
        "lambda_edge": 10.0,  # Structural constraint weight
        "lambda_reg": 3.0,  # Regularization weight
        "seed": 42,  # Random seed for reproducibility
        "class_idx": 4  # Fixed class index (e.g., Landscape)
    }
    # =================================================
    """
    Class 004 (水墨/版画风格 / Ink/Etching)
    Class 012 (抽象色块 / Abstract)
    Class 142 (清淡风景 / Light Landscape)
    Class 148 (厚重油画 / Landscape Oil)
    """

    # 1. Validate Input Directory
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    # 2. Retrieve Image Files
    # Filter for valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for f in os.listdir(input_folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_extensions:
            image_files.append(f)

    # Sort files to ensure deterministic processing order
    image_files.sort()

    if not image_files:
        print("Warning: No valid images found in the input folder.")
        return

    print(f"Found {len(image_files)} sketches. Initializing batch processing...")

    # 3. Load Model (Load once to save overhead)
    print(f"Loading network from: {network_pkl} ...")
    G = load_generator(network_pkl)
    print("Model loaded successfully.")

    # 4. Processing Loop
    for i, filename in enumerate(image_files):
        sketch_path = os.path.join(input_folder, filename)

        # specific output directory for each image (e.g., test5.0, test5.1)
        current_outdir = f"{base_outdir}.{i}"

        print(f"\n[{i + 1}/{len(image_files)}] Processing: {filename}")
        print(f"   Output Directory: {current_outdir}")
        print(f"   Parameters: Edge={params['lambda_edge']}, Reg={params['lambda_reg']}, Steps={params['num_steps']}")

        try:
            project_sketch(
                G=G,
                sketch_path=sketch_path,
                outdir=current_outdir,
                num_steps=params["num_steps"],
                lr=params["lr"],
                lambda_edge=params["lambda_edge"],
                lambda_reg=params["lambda_reg"],
                seed=params["seed"],
                class_idx=params["class_idx"]
            )
            print(f"Completed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\nBatch processing finished.")


if __name__ == "__main__":
    main()