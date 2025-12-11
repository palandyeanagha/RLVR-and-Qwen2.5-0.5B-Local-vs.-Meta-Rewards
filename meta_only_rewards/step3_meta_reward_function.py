"""
Step 3: Reward Function for GRPO Training

Simple meta reward: 1.0 if correct, 0.0 if wrong.
No intermediate reasoning evaluation yet.
"""

import re
from typing import List, Dict, Any, Optional


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.
    
    Tries multiple patterns:
    1. \boxed{number} (Qwen's preferred format)
    2. #### number (GSM8K format)
    3. "The answer is number"
    4. Last number in text
    """
    
    # Pattern 1: #### number at the LAST line (GSM8K preferred)
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if match:
        return match.group(1)

    # Pattern 1: \boxed{number} - Qwen format
    match = re.search(r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}', text)
    if match:
        return match.group(1).replace(',', '')
    
    # # Pattern 2: #### number
    # match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    # if match:
    #     return match.group(1).replace(',', '')
    
    # Pattern 3: "answer is" or "answer:"
    match = re.search(
        r'(?:answer is|answer:|final answer is|final answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).replace(',', '')
    
    # Pattern 4: Last number
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None


def compute_reward(predicted: str, target: str) -> float:
    """
    Compute reward: 1.0 if correct, 0.0 if wrong.
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
    
    # String comparison
    pred_clean = str(predicted).strip().lower()
    target_clean = str(target).strip().lower()
    
    return 1.0 if pred_clean == target_clean else 0.0


# def reward_function(batch: Dict[str, Any], data_source: Optional[str] = None, solution_str: Optional[str] = None, ground_truth: Optional[str] = None,
# extra_info: Optional[Dict[str, Any]] = None): # ← ADD THIS) -> List[float]:#batch: Dict[str, Any]) -> List[float]:
#     """
#     Main reward function for VERL.
    
#     Args:
#         batch: Dict with 'outputs' and 'references'
        
#     Returns:
#         List of rewards (1.0 or 0.0)
#     """
#     outputs = batch.get("outputs", [])
#     references = batch.get("references", [])
    
#     rewards = []
    
#     for output, reference in zip(outputs, references):
#         # Extract predicted answer
#         predicted = extract_answer(output)
        
#         # Compute reward
#         reward = compute_reward(predicted, reference)
        
#         rewards.append(reward)
    
#     return rewards

from typing import Optional, Dict, Any

def reward_function(
    data_source: str,
    solution_str: str,        # ← generated response (1 of 5)
    ground_truth: str,        # ← correct answer (from 'references')
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Called 5 times per prompt (n=5).
    Returns reward for ONE response.
    """
    predicted = extract_answer(solution_str)
    return compute_reward(predicted, ground_truth)


# Test the reward function
def test_reward_function():
    """Test with example cases."""
    
    print("=" * 70)
    print("  Testing Reward Function")
    print("=" * 70)
    
    test_cases = [
        {
            "output": "Step 1: 10 + 5 = 15\nStep 2: 15 * 2 = 30\n\\boxed{30}",
            "target": "30",
            "expected": 1.0,
            "desc": "Correct with \\boxed{} format (Qwen style)"
        },
        {
            "output": "Step 1: 10 + 5 = 15\nStep 2: 15 * 2 = 30\n#### 30",
            "target": "30",
            "expected": 1.0,
            "desc": "Correct with #### format"
        },
        {
            "output": "We calculate: 10 + 5 = 15, then 15 * 2 = 30. The answer is 30.",
            "target": "30",
            "expected": 1.0,
            "desc": "Correct with 'answer is' format"
        },
        {
            "output": "First 10 + 5 = 15. Then 15 * 2 = 30",
            "target": "30",
            "expected": 1.0,
            "desc": "Correct, last number"
        },
        {
            "output": "10 + 5 = 15, 15 * 2 = 30. \\boxed{25}",
            "target": "30",
            "expected": 0.0,
            "desc": "Wrong answer in boxed format"
        },
        {
            "output": "I'm not sure about this problem.",
            "target": "30",
            "expected": 0.0,
            "desc": "No number"
        },
    ]
    
    batch = {
        "outputs": [tc["output"] for tc in test_cases],
        "references": [tc["target"] for tc in test_cases]
    }
    
    rewards = reward_function(batch)
    
    all_pass = True
    for i, (tc, reward) in enumerate(zip(test_cases, rewards)):
        status = "PASS" if reward == tc["expected"] else "FAIL"
        if reward != tc["expected"]:
            all_pass = False
        
        print(f"\nTest {i+1}: {tc['desc']}")
        print(f"  Expected: {tc['expected']}, Got: {reward}")
        print(f"  {status}")
    
    print("\n" + "=" * 70)
    if all_pass:
        print(" All Tests Passed!")
        print("=" * 70)
        print("\n  Reward function is working correctly.")
        print("\n  Next: Step 4 - Run Training")
        print("  Run: bash step4_run_training.sh")
        print("\n" + "=" * 70)
    else:
        print(" Some Tests Failed")
        print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    success = test_reward_function()
    exit(0 if success else 1)