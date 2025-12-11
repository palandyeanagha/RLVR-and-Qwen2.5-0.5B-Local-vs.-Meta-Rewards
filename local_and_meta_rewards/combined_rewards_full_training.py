import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 70)
    print("  VERL GRPO Training - GSM8K Full Training (Combined Rewards)")
    print("=" * 70)
    print()
    
    # Configuration
    base_dir = "/teamspace/studios/this_studio"
    
    print("Configuration (combined rewards: 0.9*local + 0.1*meta):")
    print("  Dataset: GSM8K (full, preprocessed)")
    print("  Model: Qwen/Qwen2.5-0.5B")
    print("  Reward: 0.9*local (intermediate steps) + 0.1*meta (final answer)")
    print("  Algorithm/params: matches paper")
    print()
    
    response = input("Start full training? (yes/no): ").strip().lower()
    if response != "yes":
        print("Training cancelled.")
        return 0
    print()
    
    # Create timestamp/output folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{base_dir}/checkpoints_gsm8k_combined_{timestamp}"
    log_dir = f"{base_dir}/logs_gsm8k_combined_{timestamp}"
    
    # Main training command - using combined rewards
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        # DATA FILES
        f"data.train_files=[{base_dir}/data_prepared_local/train.parquet,{base_dir}/data_prepared_local/valid.parquet]",
        f"data.val_files=[{base_dir}/data_prepared_local/test.parquet]",
        "data.prompt_key=messages",
        "data.train_batch_size=64",
        "data.max_prompt_length=512",
        "data.max_response_length=1024",
        # ALGORITHM/HYPERPARAMETERS
        "algorithm.adv_estimator=grpo",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.temperature=0.6",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.3",
        # REWARD: combined rewards (0.1*local + 0.9*meta)
        f"custom_reward_function.path={base_dir}/local_meta_rewards/combined_rewards.py",
        "custom_reward_function.name=reward_function",
        "critic.enable=false",
        # TRAINER OUTPUT
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.total_epochs=5",
        "trainer.test_freq=10",
        "trainer.save_freq=50",
        f"trainer.default_local_dir={checkpoint_dir}"
    ]
    
    # Create logs/checkpoints
    log_file = Path(f"{log_dir}/training.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_file}")
    print("Training will take several hours. Use Ctrl+C to stop and checkpoint.")
    print()
    
    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'
    
    with open(log_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VERL GRPO Training Log - Combined Rewards (0.1*local + 0.9*meta)\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.flush()
        
        process = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True,
                                  bufsize=1,
                                  env=env)
        try:
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                f.flush()
            process.wait()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Stopping gracefully.")
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
    print("  2. Compare with local-only and meta-only baselines")
    print("  3. Analyze the effect of combined rewards")
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