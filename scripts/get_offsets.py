#use by supplying the dataset config file with the tokenizer
# like python scripts/get_offsets.py conf/dataset/miditok.yaml

import sys
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

def calculate_offsets(file_path):
    print(f"Reading config from: {file_path}")
    
    # 1. Load the YAML file directly
    try:
        cfg = OmegaConf.load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 2. Instantiate the Tokenizer
    # We look for the 'tokenizer' key in the yaml
    if 'tokenizer' not in cfg:
        print("Error: Could not find 'tokenizer' key in the YAML file.")
        return

    print("Instantiating tokenizer...")
    try:
        # recursivity=True ensures the inner TokenizerConfig in _args_ is also instantiated
        tokenizer = instantiate(cfg.tokenizer, _convert_="partial")
    except Exception as e:
        print(f"Error instantiating tokenizer: {e}")
        print("Make sure 'miditok' is installed and the config structure is correct.")
        return

    # 3. Analyze Vocabulary
    vocab = tokenizer.vocab
    vocab_sizes = [len(v) for v in vocab]
    
    print(f"\n--- ANALYSIS ---")
    print(f"Attributes per note: {len(vocab_sizes)}")
    print(f"Vocab sizes: {vocab_sizes}")

    # 4. Calculate Offsets
    # We apply the +4 Shift immediately for Special Tokens (PAD, BOS, EOS, UNK)
    SHIFT = 4
    
    final_offsets = []
    current_offset = SHIFT
    
    for size in vocab_sizes:
        final_offsets.append(current_offset)
        current_offset += size
        
    # Total Size = Sum of attributes + 4 Specials + 1 Mask
    total_vocab_size = sum(vocab_sizes) + SHIFT + 1

    # 5. Output
    print("\n" + "="*50)
    print(f"   COPY INTO conf/model/musicbert_diffusion.yaml")
    print("="*50)
    print(f"offsets: {final_offsets}")
    print(f"vocab_size: {total_vocab_size}")
    print(f"tokens_per_note: {len(vocab_sizes)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/get_offsets.py <path_to_dataset_yaml>")
        sys.exit(1)
    
    calculate_offsets(sys.argv[1])