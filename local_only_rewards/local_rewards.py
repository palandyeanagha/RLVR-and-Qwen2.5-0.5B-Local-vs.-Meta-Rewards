"""
WORKAROUND: Load expected_steps from parquet at module load time
"""
import re
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union

# Load the mapping at module import time
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
    print(f"Loaded {len(EXPECTED_STEPS_MAP)} expected_steps mappings")
except Exception as e:
    print(f"Error loading expected_steps: {e}")

def extract_intermediate_numbers(text: str) -> List[str]:
    """Extract intermediate calculation results from model output."""
    numbers = []
    lines = text.strip().split('\n')
    
    for line in lines:
        if '####' in line:
            continue
        tagged = re.findall(r'<<[^>]+>>(-?\d+(?:\.\d+)?)', line)
        if tagged:
            numbers.extend(tagged)
            continue
        equals = re.findall(r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if equals:
            numbers.append(equals[-1].replace(',', ''))
            continue
        colon_pattern = re.findall(
            r'(?:Total|Answer|Result|Sum):\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            line,
            re.IGNORECASE
        )
        if colon_pattern:
            numbers.append(colon_pattern[-1].replace(',', ''))
            continue
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
    """Flexible step matching (order-invariant)."""
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
    if len(predicted_steps) > len(expected_steps) * 1.5:
        score *= 0.95
    return score

def local_reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Local reward using pre-loaded expected_steps mapping.
    """
    # Extract question from solution_str (hacky but works)
    # The question is embedded in the solution for validation
    question_match = re.search(r'Question: ([^\n]+)', solution_str)
    if not question_match:
        # Fallback: just use ground_truth with hash 0
        key = (ground_truth, 0)
        expected_steps = EXPECTED_STEPS_MAP.get(key, [])
    else:
        question = question_match.group(1)
        key = (ground_truth, hash(question) % 1000000)
        expected_steps = EXPECTED_STEPS_MAP.get(key, [])
    
    if not expected_steps:
        # Fallback: return 0
        return 0.0
    
    predicted_steps = extract_intermediate_numbers(solution_str)
    
    if not predicted_steps:
        return 0.0
    
    local_reward = match_steps_flexible(predicted_steps, expected_steps)
    
    return local_reward
