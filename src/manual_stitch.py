import os
import glob
from PIL import Image, ImageDraw, ImageFont

"""
Manual Stitching Tool
Function: Scans the output_grid_test folder and automatically generates a comparison summary image based on folder names.
Does not require model loading; pure image processing, very fast.
"""

# ================= Configuration Area =================
# Your experiment result folder (Ensure the name matches your grid_search run)
INPUT_DIR = "output_grid_test/E_8-20_R_2-8"
# Output filename for the stitched image
OUTPUT_FILENAME = "manual_summary.png"


# ======================================================

def get_params_from_folder(folder_name):
    """
    Extract edge=10.0, reg=0.5 from folder name 'E_10.0_R_0.5'
    """
    try:
        parts = folder_name.split('_')
        # Assume format is E_{edge}_R_{reg}
        edge_val = float(parts[1])
        reg_val = float(parts[3])
        return edge_val, reg_val
    except:
        return None, None


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Folder '{INPUT_DIR}' not found. Please check the name.")
        return

    # 1. Scan all subfolders
    subfolders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
    data = {}  # Storage format: (edge, reg) -> image_path

    unique_edges = set()
    unique_regs = set()

    print(f"Scanning {INPUT_DIR} ...")

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        edge, reg = get_params_from_folder(folder_name)

        if edge is not None and reg is not None:
            # Find the result image in the folder
            # Prioritize final_art.png; if not found, find the last step image
            img_path = os.path.join(folder, "final_art.png")
            if not os.path.exists(img_path):
                # Try finding proj_stepxxxx.png
                steps = glob.glob(os.path.join(folder, "proj_step*.png"))
                if steps:
                    steps.sort()
                    img_path = steps[-1]  # Take the last one
                else:
                    print(f"Warning: No images in {folder_name}, skipping.")
                    continue

            unique_edges.add(edge)
            unique_regs.add(reg)
            data[(edge, reg)] = img_path

    # 2. Sort parameters (Ensure grid is ordered small to large)
    sorted_edges = sorted(list(unique_edges))
    sorted_regs = sorted(list(unique_regs))

    print(f"Identified Edge parameters: {sorted_edges}")
    print(f"Identified Reg  parameters: {sorted_regs}")

    if not data:
        print("No valid data found, terminating.")
        return

    # 3. Create canvas
    cell_w, cell_h = 512, 512
    margin_left = 240
    margin_top = 180

    grid_w = margin_left + len(sorted_regs) * cell_w
    grid_h = margin_top + len(sorted_edges) * cell_h

    canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try loading font (Use default if failed)
    try:
        # Common Windows font path
        font = ImageFont.truetype("arial.ttf", 40)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except:
        print("arial.ttf not found, using default font.")
        font = None
        font_small = None

    # 4. Draw header (Reg)
    for j, reg in enumerate(sorted_regs):
        x = margin_left + j * cell_w + cell_w // 2 - 60
        draw.text((x, 20), f"Reg={reg}", fill="black", font=font)

    # 5. Draw sidebar (Edge)
    for i, edge in enumerate(sorted_edges):
        y = margin_top + i * cell_h + cell_h // 2 - 20
        draw.text((10, y), f"Edge={edge}", fill="black", font=font)

    # 6. Paste images
    print("Stitching images...")
    for i, edge in enumerate(sorted_edges):
        for j, reg in enumerate(sorted_regs):
            img_path = data.get((edge, reg))

            x = margin_left + j * cell_w
            y = margin_top + i * cell_h

            if img_path:
                try:
                    img = Image.open(img_path).resize((cell_w, cell_h))
                    canvas.paste(img, (x, y))
                except Exception as e:
                    print(f"Image read error: {e}")
            else:
                # If a combination is missing, draw N/A
                draw.text((x + 200, y + 200), "N/A", fill="red", font=font)

    # 7. Save
    save_path = os.path.join(INPUT_DIR, OUTPUT_FILENAME)
    canvas.save(save_path)
    print(f"\nSuccess! Summary image saved to: {save_path}")
    print("Please check the output file.")


if __name__ == "__main__":
    main()