"""
Convert VERL checkpoint to HuggingFace format
"""
import torch
from transformers import AutoModelForCausalLM

# Load base model architecture
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load trained weights from VERL checkpoint
print("Loading trained weights...")
checkpoint = torch.load(
    "checkpoints_gsm8k_20251109_050411/global_step_150/actor/model_world_size_1_rank_0.pt",
    map_location='cpu'
)

# Extract model state dict
if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Load weights into model
model.load_state_dict(state_dict, strict=False)

# Save in HuggingFace format
print("Saving model in HuggingFace format...")
output_path = "checkpoints_gsm8k_20251109_050411/global_step_150/actor/huggingface"
model.save_pretrained(output_path)

print(f" Model saved to {output_path}")
print("Now you can run the evaluation script!")
