import sys
import os
import pickle
import torch
import legacy  # This calls the patched legacy.py


def convert_and_save(tf_pkl_path, output_pkl_path):
    print(f'>>> Loading TensorFlow model: {tf_pkl_path} ...')
    print('>>> (This triggers the conversion logic in legacy.py)')

    # 1. Load using legacy (using the patch to skip errors)
    with dnnlib.util.open_url(tf_pkl_path) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema']  # We only need the generator

    print(f'>>> Conversion successful! Generator structure: {G.img_resolution}x{G.img_resolution}')

    # 2. Construct standard PyTorch model dictionary
    # The StyleGAN PyTorch format typically requires a dict containing G, D, G_ema
    # For image generation, storing just G_ema is sufficient, or store the complete set
    data = dict(G=G, G_ema=G, D=None)

    # 3. Save as new PKL file
    print(f'>>> Saving in PyTorch format: {output_pkl_path} ...')
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print('>>> Done! You can now use the new model directly without relying on TensorFlow logic.')


if __name__ == "__main__":
    # Configure your paths
    import dnnlib  # Need to import this library

    # Input: Your old TF model
    input_model = "pretrained/wikiart.pkl"

    # Output: The new name you want to save
    output_model = "pretrained/wikiart_pytorch.pkl"

    convert_and_save(input_model, output_model)