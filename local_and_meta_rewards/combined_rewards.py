"""
Combined Reward Function: Local + Meta Rewards

Weighting: 0.1 * local + 0.9 * meta
- Local reward (0-1): Rewards correct intermediate reasoning steps
- Meta reward (0 or 1): Rewards correct final answer
- Combined: Emphasizes final answer correctness while still encouraging good reasoning
"""

import re
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

# ============================================================================
# LOAD EXPECTED STEPS (for local rewards)
# ============================================================================

print("Loading expected_steps mapping from parquet...")
EXPECTED_STEPS_MAP = {}

try:
    for split in ['train', 'valid', 'test']:
        df = pd.read_parquet(f'/teamspace/studios/this_studio/data_prepared_local/{split}.parquet')
        for _, row in df.iterrows():
            target = str(row['target'])
            expected_steps = row['expected_steps']
            if isinstance(expected_steps, np.ndarray):
                expected_steps = expected_steps.tolist()
            # Use (target, question_hash) as key to handle duplicates
            question = row['question']
            key = (target, hash(question) % 1000000)
            EXPECTED_STEPS_MAP[key] = expected_steps
    print(f"✓ Loaded {len(EXPECTED_STEPS_MAP)} expected_steps mappings")
except Exception as e:
    print(f"✗ Error loading expected_steps: {e}")

# ============================================================================
# META REWARD FUNCTIONS (final answer correctness)
# ============================================================================

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.
    Tries multiple patterns in order of preference.
    """
    # Pattern 1: #### number at the LAST line (GSM8K preferred)
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1)

    # Pattern 2: \boxed{number} - Qwen format
    match = re.search(r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 3: "answer is" or "answer:"
    match = re.search(
        r'(?:answer is|answer:|final answer is|final answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 4: Last number in text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None


def compute_meta_reward(predicted: str, target: str) -> float:
    """
    Meta reward: 1.0 if final answer correct, 0.0 if wrong.
    """
    if predicted is None or target is None:
        return 0.0
    
    # Try numeric comparison
    try:
        pred_num = float(predicted)
        target_num = float(target)
        
        # Small tolerance for floating point
        if abs(pred_num - target_num) < 1e-5:
            return 1.0
    except (ValueError, TypeError):
        pass
    
    # String comparison fallback
    pred_clean = str(predicted).strip().lower()
    target_clean = str(target).strip().lower()
    
    return 1.0 if pred_clean == target_clean else 0.0


# ============================================================================
# LOCAL REWARD FUNCTIONS (intermediate reasoning steps)
# ============================================================================

def extract_intermediate_numbers(text: str) -> List[str]:
    """Extract intermediate calculation results from model output."""
    numbers = []
    lines = text.strip().split('\n')
    
    for line in lines:
        # Skip final answer line
        if '####' in line:
            continue
        
        # Pattern 1: Tagged numbers like <<calculation>>result
        tagged = re.findall(r'<<[^>]+>>(-?\d+(?:\.\d+)?)', line)
        if tagged:
            numbers.extend(tagged)
            continue
        
        # Pattern 2: Numbers after equals sign
        equals = re.findall(r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if equals:
            numbers.append(equals[-1].replace(',', ''))
            continue
        
        # Pattern 3: Labeled results (Total: 42, Sum: 100, etc.)
        colon_pattern = re.findall(
            r'(?:Total|Answer|Result|Sum):\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            line,
            re.IGNORECASE
        )
        if colon_pattern:
            numbers.append(colon_pattern[-1].replace(',', ''))
            continue
        
        # Pattern 4: "is" or "equals" patterns
        is_pattern = re.findall(
            r'(?:is|equals)\s+(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            line,
            re.IGNORECASE
        )
        if is_pattern:
            numbers.append(is_pattern[-1].replace(',', ''))
            continue
    
    return numbers


def match_steps_flexible(predicted_steps: List[str], expected_steps: List[str]) -> float:
    """
    Flexible step matching (order-invariant).
    Returns proportion of expected steps found in predicted steps.
    """
    if not expected_steps:
        return 0.0
    if not predicted_steps:
        return 0.0
    
    matches = 0
    for exp in expected_steps:
        for pred in predicted_steps:
            try:
                if abs(float(pred) - float(exp)) < 1e-5:
                    matches += 1
                    break
            except (ValueError, TypeError):
                pass
    
    score = matches / len(expected_steps)
    
    # Small penalty for excessive intermediate steps (prevents verbosity)
    if len(predicted_steps) > len(expected_steps) * 1.5:
        score *= 0.95
    
    return score


def compute_local_reward(solution_str: str, ground_truth: str) -> float:
    """
    Local reward using pre-loaded expected_steps mapping.
    Returns score from 0.0 to 1.0 based on intermediate step correctness.
    """
    # Extract question from solution_str to build lookup key
    question_match = re.search(r'Question: ([^\n]+)', solution_str)
    if not question_match:
        # Fallback: use ground_truth with hash 0
        key = (ground_truth, 0)
        expected_steps = EXPECTED_STEPS_MAP.get(key, [])
    else:
        question = question_match.group(1)
        key = (ground_truth, hash(question) % 1000000)
        expected_steps = EXPECTED_STEPS_MAP.get(key, [])
    
    if not expected_steps:
        # No expected steps available -> return 0
        return 0.0
    
    # Extract intermediate numbers from model output
    predicted_steps = extract_intermediate_numbers(solution_str)
    
    if not predicted_steps:
        return 0.0
    
    # Compute local reward (proportion of correct steps)
    local_reward = match_steps_flexible(predicted_steps, expected_steps)
    
    return local_reward


# ============================================================================
# COMBINED REWARD FUNCTION
# ============================================================================

def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Combined reward function: 0.1 * local + 0.9 * meta
    
    This weighting prioritizes final answer correctness (90%) while still
    encouraging correct intermediate reasoning steps (10%).
    
    Args:
        data_source: Name of the dataset (e.g., 'gsm8k')
        solution_str: Generated response from the model
        ground_truth: Correct final answer
        extra_info: Optional extra information (unused)
    
    Returns:
        Combined reward score from 0.0 to 1.0
    """
    # Compute local reward (intermediate steps)
    local_reward = compute_local_reward(solution_str, ground_truth)
    
    # Compute meta reward (final answer)
    predicted_answer = extract_final_answer(solution_str)
    meta_reward = compute_meta_reward(predicted_answer, ground_truth)
    
    # Combined reward: 0.1 * local + 0.9 * meta
    #combined_reward = 0.1 * local_reward + 0.9 * meta_reward
    #combined_reward = 0.5 * local_reward + 0.5 * meta_reward
    combined_reward = 0.9 * local_reward + 0.1 * meta_reward
    
    return combined_reward


# ==================================
# 
# ==========================================
# TEST FUNCTION
# ============================================================================

def test_reward_function():
    """Test the combined reward function with example cases."""
    
    print("\n" + "=" * 80)
    print("TESTING COMBINED REWARD FUNCTION (0.9 * local + 0.1 * meta)")
    print("=" * 80)
    
    test_cases = [
        {
            "solution": """Question: Janet has 16 eggs. She uses 3 for breakfast and 4 for muffins. She sells the rest for $2 each. How much does she make?

Answer:
Step 1: Calculate eggs used
- Breakfast: 3
- Muffins: 4
- Total used: 3 + 4 = 7

Step 2: Calculate eggs to sell
- Total eggs: 16
- Used: 7
- To sell: 16 - 7 = 9

Step 3: Calculate revenue
- Price: $2
- Eggs: 9
- Revenue: 9 × 2 = 18

#### 18""",
            "target": "18",
            "expected_local": 1.0,  # All steps correct
            "expected_meta": 1.0,   # Final answer correct
            "expected_combined": 0.9 * 1.0 + 0.1 * 1.0,  # = 1.0
            "desc": "Perfect: All steps correct + final answer correct"
        },
        {
            "solution": """Question: What is 10 + 20?

Answer:
10 + 20 = 50

#### 50""",
            "target": "30",
            "expected_local": 0.0,  # Wrong intermediate step
            "expected_meta": 0.0,   # Wrong final answer
            "expected_combined": 0.0,
            "desc": "All wrong: Wrong steps + wrong final answer"
        },
        {
            "solution": """Question: What is 10 + 20?

Answer:
Step 1: 10 + 20 = 30

#### 25""",
            "target": "30",
            "expected_local": 1.0,  # Intermediate step correct
            "expected_meta": 0.0,   # Final answer wrong
            "expected_combined": 0.9 * 1.0 + 0.1 * 0.0,  # = 0.1
            "desc": "Mixed: Correct reasoning but wrong final answer"
        },
        {
            "solution": """Question: What is 10 + 20?

Answer:
The answer is clearly 30.

#### 30""",
            "target": "30",
            "expected_local": 0.0,  # No intermediate steps shown
            "expected_meta": 1.0,   # Final answer correct
            "expected_combined": 0.9 * 0.0 + 0.1 * 1.0,  # = 0.9
            "desc": "Lucky guess: No reasoning but correct final answer"
        },
    ]
    
    print("\nRunning test cases...\n")
    
    all_pass = True
    for i, tc in enumerate(test_cases, 1):
        solution = tc["solution"]
        target = tc["target"]
        
        # Compute rewards
        local = compute_local_reward(solution, target)
        predicted = extract_final_answer(solution)
        meta = compute_meta_reward(predicted, target)
        combined = reward_function("gsm8k", solution, target)
        
        print(f"{'Test ' + str(i):-^80}")
        print(f"Description: {tc['desc']}")
        print(f"\nRewards:")
        print(f"  Local:    {local:.3f} (expected ~{tc['expected_local']:.3f})")
        print(f"  Meta:     {meta:.3f} (expected ~{tc['expected_meta']:.3f})")
        print(f"  Combined: {combined:.3f} (expected ~{tc['expected_combined']:.3f})")
        
        # Check if approximately correct (allow small tolerance)
        tolerance = 0.15
        local_ok = abs(local - tc['expected_local']) < tolerance
        meta_ok = abs(meta - tc['expected_meta']) < tolerance
        combined_ok = abs(combined - tc['expected_combined']) < tolerance
        
        if local_ok and meta_ok and combined_ok:
            print("  Status: ✓ PASS")
        else:
            print("  Status: ✗ FAIL")
            all_pass = False
        print()
    
    print("=" * 80)
    if all_pass:
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nCombined reward function is working correctly.")
        print("\nWeighting: 0.9 * local + 0.1 * meta")
        print("  - Prioritizes final answer correctness (90%)")
        print("  - Still encourages good reasoning (10%)")
        print("\nNext steps:")
        print("  1. Copy this file to your training directory")
        print("  2. Update your training config to use this reward function")
        print("  3. Run training with combined rewards")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
    
    return all_pass


if __name__ == "__main__":
    success = test_reward_function()
    exit(0 if success else 1)