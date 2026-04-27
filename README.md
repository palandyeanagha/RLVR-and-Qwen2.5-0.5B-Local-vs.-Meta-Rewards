# Reinforcement Learning with Verifiable Rewards: Local vs. Meta Reward Shaping on Qwen2.5-0.5B

> **Post-trained Qwen2.5-0.5B using RLVR with GRPO on GSM8K, improving math reasoning accuracy from 33% to 63%. Explored how different reward signal designs (local, meta, and hybrid) affect reasoning quality and training stability.**

## 🎯 Problem

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful alternative to RLHF for improving LLM reasoning — it uses programmatically verifiable reward signals instead of human preference data. But a key open question remains: **how should reward signals be structured?**

This project investigates whether decomposing rewards into **local** (step-level) and **meta** (trajectory-level) components improves reasoning accuracy and inference stability compared to using either signal alone.

## 🏗️ Approach

We post-trained **Qwen2.5-0.5B** using **Group Relative Policy Optimization (GRPO)** via the VERL framework on the GSM8K math reasoning dataset. Three reward configurations were compared:

| Configuration | What It Rewards | Hypothesis |
|--------------|----------------|------------|
| **Local Only** | Correctness of individual reasoning steps | Encourages step-by-step rigor |
| **Meta Only** | Overall answer correctness + trace quality | Encourages coherent solution strategies |
| **Hybrid (Local + Meta)** | Weighted combination of both signals | Balances step-level precision with global coherence |

### Pipeline

```
Qwen2.5-0.5B (base) → GRPO Training (VERL) → Reward Shaping → Evaluation
                              ↓
                    GSM8K (8.5K training problems)
                              ↓
                    Local / Meta / Hybrid Rewards
                              ↓
                    Chain-of-Thought Analysis + GPT-4o Error Evaluation
```

## 📊 Key Results

- **Baseline accuracy:** 33% → **Best post-trained accuracy:** 63% (hybrid reward)
- Hybrid local/meta reward weighting improved inference stability and reasoning trace quality
- Local-only rewards produced more structured step-by-step traces but occasionally missed global coherence
- Meta-only rewards achieved good final-answer accuracy but with less interpretable reasoning paths
- GPT-4o-based error analysis revealed distinct failure modes across reward configurations

## 📁 Repository Structure

```
├── baseline/              # Baseline Qwen2.5-0.5B evaluation scripts
├── local_only_rewards/    # Training configs & results for local reward setup
├── meta_only_rewards/     # Training configs & results for meta reward setup
├── local_and_meta_rewards/# Hybrid reward configuration
├── evaluation/            # Evaluation scripts, CoT analysis, error categorization
└── README.md
```

## 🛠️ Tech Stack

- **Model:** Qwen2.5-0.5B-Instruct
- **Training Framework:** [VERL](https://github.com/volcengine/verl) (Volcano Engine RL)
- **RL Algorithm:** GRPO (Group Relative Policy Optimization)
- **Dataset:** GSM8K (grade school math reasoning)
- **Evaluation:** Custom CoT trace validators, GPT-4o-based error analysis
- **Infrastructure:** Python, PyTorch, Hugging Face Transformers

## 🔗 Context

This project was conducted as part of exploring post-training methods for improving LLM reasoning capabilities.
