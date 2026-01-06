import os
import argparse
import itertools
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

# 导入你的核心工具
from .gan_loader import load_generator
from .sketch_projector import project_sketch


def create_image_grid(results, edge_weights, reg_weights, out_path):
    """
    将所有结果拼成一张带有文字标签的大网格图
    python -m src.grid_search --network pretrained/wikiart_pytorch.pkl --sketch input_sketches/3.jpg --base_outdir output_grid_test/E_8-20_R_2-8 --steps 3000 --cls 30
    """
    cell_w, cell_h = 512, 512  # 假设每张图是 512x512
    margin_top = 50  # 留出顶部写 reg_weight
    margin_left = 100  # 留出左边写 edge_weight

    # 计算大图尺寸
    grid_w = margin_left + len(reg_weights) * cell_w
    grid_h = margin_top + len(edge_weights) * cell_h

    grid_img = Image.new('RGB', (grid_w, grid_h), color='white')
    draw = ImageDraw.Draw(grid_img)

    # 尝试加载一个字体，如果没有就用默认
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = None  # 使用默认字体

    # 1. 绘制表头 (Reg Weights)
    for j, reg in enumerate(reg_weights):
        x = margin_left + j * cell_w + cell_w // 2 - 50
        draw.text((x, 10), f"Reg={reg}", fill="black", font=font)

    # 2. 绘制侧边栏 (Edge Weights)
    for i, edge in enumerate(edge_weights):
        y = margin_top + i * cell_h + cell_h // 2 - 10
        draw.text((10, y), f"Edge={edge}", fill="black", font=font)

    # 3. 填充图片
    for i, edge in enumerate(edge_weights):
        for j, reg in enumerate(reg_weights):
            key = (edge, reg)
            img_path = results.get(key)

            x_pos = margin_left + j * cell_w
            y_pos = margin_top + i * cell_h

            if img_path and os.path.exists(img_path):
                try:
                    img = Image.open(img_path).resize((cell_w, cell_h))
                    grid_img.paste(img, (x_pos, y_pos))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            else:
                # 如果运行失败，画个红叉
                draw.text((x_pos + 50, y_pos + 200), "Failed", fill="red", font=font)

    # 保存大图
    grid_img.save(out_path)
    print(f"✅ Grid summary saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Sketch Projector")
    parser.add_argument('--network', type=str, required=True, help='Path to .pkl')
    parser.add_argument('--sketch', type=str, required=True, help='Path to sketch image')
    parser.add_argument('--base_outdir', type=str, default='output_grid', help='Base output directory')
    parser.add_argument('--steps', type=int, default=500, help='Steps per run')
    parser.add_argument('--cls', type=int, default=None, help='Class index')

    args = parser.parse_args()

    # ==========================================
    # 在此处定义你的搜索空间 (Search Space)
    # ==========================================
    # 建议：Edge 不要太大，Reg 不要太小
    #edge_list = [2.0, 5.0, 10.0]  # 纵轴
    #reg_list = [0.1, 0.5, 1.0, 3.0]  # 横轴

    # ==========================================
    # ==========================================
    # 第二轮：精细搜索 (Fine-Grained Search)
    # 策略：以上一轮最佳 (Edge=10, Reg=3) 为中心，向高值探索
    # ==========================================

    edge_list = [8.0, 10.0, 15.0, 20.0]  # 抛弃了 2 和 5
    reg_list = [2.0, 3.0, 5.0, 8.0]  # 抛弃了 0.1 和 0.5

    # ==========================================

    print(f"Loading model: {args.network} ...")
    G = load_generator(args.network)

    results = {}  # 记录结果路径 (edge, reg) -> path

    total_experiments = len(edge_list) * len(reg_list)
    current_idx = 0

    print(f"🚀 Starting Grid Search: {total_experiments} experiments.")

    for edge in edge_list:
        for reg in reg_list:
            current_idx += 1
            print(f"\n[{current_idx}/{total_experiments}] Running: Edge={edge}, Reg={reg} ...")

            # 为每个实验创建一个子文件夹
            run_name = f"E_{edge}_R_{reg}"
            outdir = os.path.join(args.base_outdir, run_name)

            # 调用核心投影函数
            try:
                project_sketch(
                    G,
                    args.sketch,
                    outdir,
                    num_steps=args.steps,
                    lr=0.05,  # 学习率建议固定一个小一点的值
                    lambda_edge=edge,  # 传入当前 Edge 权重
                    lambda_reg=reg,  # 传入当前 Reg 权重
                    seed=42,  # 种子固定，控制变量
                    class_idx=args.cls
                )

                # 记录结果文件路径
                final_img = os.path.join(outdir, "final_art.png")
                results[(edge, reg)] = final_img

            except Exception as e:
                print(f"❌ Experiment failed: {e}")

    # 所有实验跑完，生成汇总大图
    print("\nGenerating summary grid...")
    summary_path = os.path.join(args.base_outdir, "summary_grid.png")
    create_image_grid(results, edge_list, reg_list, summary_path)


if __name__ == "__main__":
    main()