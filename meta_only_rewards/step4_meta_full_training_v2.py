"""
VERL GRPO Training - Full Training Script
Matches paper hyperparameters for GSM8K
"""

import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 70)
    print("  VERL GRPO Training - GSM8K Full Training")
    print("=" * 70)
    print()
    
    # Configuration
    base_dir = "/teamspace/studios/this_studio"
    
    print("Configuration (matching paper):")
    print("  Dataset: GSM8K (full)")
    print("  Model: Qwen/Qwen2.5-0.5B")
    print("  Algorithm: GRPO with KL regularization")
    print("  Batch size: 64 (mini-batch: 8)")
    print("  Responses per prompt: 5")
    print("  Temperature: 0.6")
    print("  Learning rate: 1e-6")
    print("  KL coefficient: 0.001")
    print("  Epochs: 5")
    print("  Max prompt length: 512")
    print("  Max response length: 1024")
    print()
    
    response = input("Start full training? (yes/no): ").strip().lower()
    if response != "yes":
        print("Training cancelled.")
        return 0
    print()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{base_dir}/checkpoints_gsm8k_{timestamp}"
    log_dir = f"{base_dir}/logs_gsm8k_{timestamp}"
    
    # Training command
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        
        # ===== Data =====
        f"data.train_files=[{base_dir}/data_prepared/train.parquet,{base_dir}/data_prepared/valid.parquet]",
        #"data.prompt_key=prompt", 
        
        f"data.val_files=[{base_dir}/data_prepared/test.parquet]",
        "data.prompt_key=messages", 
        "data.train_batch_size=64",           # Paper: 64
        "data.max_prompt_length=512",         # Paper: 512
        "data.max_response_length=1024",      # Paper: 1024
        
        # ===== Algorithm =====
        "algorithm.adv_estimator=grpo",
        "algorithm.kl_ctrl.kl_coef=0.001",    # Paper: β = 1e-3 (KL stabilizer)
        
        # ===== Model =====
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B",
        
        # ===== Actor (Training) =====
        "actor_rollout_ref.actor.optim.lr=1e-6",              # Paper: 1e-6
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",      # Paper: 8
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        
        # ===== Rollout (Generation during training) =====
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=5",                      # Paper: 5 responses
        "actor_rollout_ref.rollout.temperature=0.6",          # Paper: 0.6
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        #"actor_rollout_ref.model.apply_chat_template=false",
        # ===== Evaluation Sampling ===== giving error
        # "+actor_rollout_ref.rollout.eval_temperature=1.0",   # Paper: 1.0
        # "+actor_rollout_ref.rollout.eval_top_p=0.95",        # Paper: 0.95
        
        # ===== Critic & Reward =====
        # "critic.enable=false",
        # "reward_model.enable=false",
        "critic.enable=false", #true",
        #"critic.model.path=Qwen/Qwen2.5-0.5B",
        #"critic.optim.lr=1e-6",
        # "critic.micro_batch_size_per_gpu=8",
        #"critic.ppo_micro_batch_size_per_gpu=8",

        #"critic.model_config.path=Qwen/Qwen2.5-0.5B",
        #"critic.micro_batch_size_per_gpu=8",
        #"reward_model.enable=true",
        #"reward_model.type=custom",
        #f"reward_model.path={base_dir}/step3_reward_function.py::reward_function",
        "custom_reward_function.path=/teamspace/studios/this_studio/step3_meta_reward_function.py",
        "custom_reward_function.name=reward_function",
        
        # ===== Trainer =====
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=5",              # Paper: 5 epochs (145 steps)
        "trainer.test_freq=10",                # Paper: eval at 0, 10, 100
        "trainer.save_freq=10",                # Save every 10 steps
        f"trainer.default_local_dir={checkpoint_dir}"
    ]
    
    # Create log directory
    log_file = Path(f"{log_dir}/training.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_file}")
    print()
    print("Training will take approximately 3-4 hours on A100 80GB.")
    print("Press Ctrl+C to stop training (checkpoints will be saved).")
    print()
    
    # Set environment
    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'  # Disable wandb for now
    
    # Run training
    with open(log_file, "w") as f:
        # Write header
        f.write("=" * 70 + "\n")
        f.write("VERL GRPO Training Log\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        try:
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                f.flush()
            
            process.wait()
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Stopping gracefully...")
            process.terminate()
            process.wait(timeout=30)
            print("Training stopped. Checkpoints saved.")
            return 1
    
    print()
    if process.returncode != 0:
        print("=" * 70)
        print("  Training Failed")
        print("=" * 70)
        print(f"Check logs: {log_file}")
        return 1
    
    print("=" * 70)
    print(" Training Completed Successfully!")
    print("=" * 70)
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Logs: {log_file}")
    print()
    print("Next steps:")
    print("  1. Evaluate model on test set")
    print("  2. Generate Pass@K metrics (K=1,4,16)")
    print("  3. Compare with baseline")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
