#!/usr/bin/env python3
"""
Convert VERL checkpoint to HuggingFace format for easy sharing and usage.

Usage:
    python convert_verl_to_hf.py \
        --verl-checkpoint checkpoints_gsm8k_local_20251124_172128/global_step_500 \
        --output-dir local_rewards_step500_hf \
        --base-model Qwen/Qwen2.5-0.5B
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import shutil

def convert_verl_to_hf(verl_checkpoint_path, output_dir, base_model_name):
    """
    Convert VERL checkpoint to HuggingFace format.
    
    Args:
        verl_checkpoint_path: Path to VERL checkpoint (e.g., .../global_step_500)
        output_dir: Directory to save HuggingFace format model
        base_model_name: Base model identifier (e.g., Qwen/Qwen2.5-0.5B)
    """
    
    print("="*80)
    print("CONVERTING VERL CHECKPOINT TO HUGGINGFACE FORMAT")
    print("="*80)
    
    # Paths
    model_weights_path = os.path.join(verl_checkpoint_path, "actor", "model_world_size_1_rank_0.pt")
    tokenizer_path = os.path.join(verl_checkpoint_path, "actor", "huggingface")
    
    # Validate paths exist
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found at: {model_weights_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer files not found at: {tokenizer_path}")
    
    print(f"\n1. Loading VERL checkpoint from: {verl_checkpoint_path}")
    print(f"   - Model weights: {model_weights_path}")
    print(f"   - Tokenizer: {tokenizer_path}")
    
    # Load VERL checkpoint
    print("\n2. Loading checkpoint into memory...")
    checkpoint = torch.load(model_weights_path, map_location="cpu")
    
    # Check checkpoint structure
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("   Found 'model' key in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("   Found 'state_dict' key in checkpoint")
        else:
            state_dict = checkpoint
            print("   Using checkpoint directly as state_dict")
    else:
        state_dict = checkpoint
        print("   Checkpoint is already a state_dict")
    
    print(f"   State dict has {len(state_dict)} keys")
    
    # Load base model architecture
    print(f"\n3. Loading base model architecture: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    print("   Base model loaded")
    
    # Load trained weights into model
    print("\n4. Loading trained weights into model...")
    
    # Handle potential key mismatches
    model_state_dict = base_model.state_dict()
    
    # Try to match keys
    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    
    for key in state_dict.keys():
        if key in model_state_dict:
            matched_keys.append(key)
        else:
            # Try without 'module.' prefix (common in distributed training)
            key_without_module = key.replace('module.', '')
            if key_without_module in model_state_dict:
                matched_keys.append(key)
                state_dict[key_without_module] = state_dict.pop(key)
            else:
                unexpected_keys.append(key)
    
    for key in model_state_dict.keys():
        if key not in state_dict and key.replace('module.', '') not in state_dict:
            missing_keys.append(key)
    
    print(f"    Matched {len(matched_keys)} keys")
    if missing_keys:
        print(f"  Missing {len(missing_keys)} keys (will use base model weights)")
    if unexpected_keys:
        print(f"   Unexpected {len(unexpected_keys)} keys (will be ignored)")
    
    # Load the state dict
    base_model.load_state_dict(state_dict, strict=False)
    print("   Weights loaded successfully")
    
    # Create output directory
    print(f"\n5. Saving model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    base_model.save_pretrained(
        output_dir,
        safe_serialization=True  # Use safetensors format (recommended)
    )
    print("   Model saved")
    
    # Copy tokenizer files
    print("\n6. Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)
    print("   Tokenizer saved")
    
    # Verify the conversion
    print("\n7. Verifying conversion...")
    test_model = AutoModelForCausalLM.from_pretrained(output_dir)
    test_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    print("   Model can be loaded from HuggingFace format")
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {output_dir}/")
    print("\nFiles created:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"  - {file} ({size_mb:.1f} MB)")
    
    print("\n" + "="*80)
    print("USAGE:")
    print("="*80)
    print(f"\nYour team can now load the model with:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
""")
    
    print("\nOr run the demo script:")
    print(f"    python demo_local_rewards.py --checkpoint {output_dir} --mode demo\n")

def main():
    parser = argparse.ArgumentParser(
        description="Convert VERL checkpoint to HuggingFace format"
    )
    parser.add_argument(
        '--verl-checkpoint',
        type=str,
        required=True,
        help='Path to VERL checkpoint directory (contains actor/ subdirectory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for HuggingFace format model'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen2.5-0.5B',
        help='Base model identifier (default: Qwen/Qwen2.5-0.5B)'
    )
    
    args = parser.parse_args()
    
    try:
        convert_verl_to_hf(
            args.verl_checkpoint,
            args.output_dir,
            args.base_model
        )
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())