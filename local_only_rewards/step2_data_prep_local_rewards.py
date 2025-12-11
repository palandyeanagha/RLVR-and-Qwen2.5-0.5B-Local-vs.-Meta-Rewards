"""
Step 2: Prepare Data for VERL Training (LOCAL REWARDS - CORRECTED)

Converts JSONL files to VERL-compatible parquet format,
including intermediate steps for local reward functions.
"""

import json
import pandas as pd
import re
from pathlib import Path

def extract_numeric_answer(answer_text):
    """
    Extract final numeric answer from GSM8K format.
    Example: "... calculations #### 108"
    """
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def extract_intermediate_steps(answer_text):
    """
    Extract intermediate calculation results from GSM8K answer.
    
    GSM8K has two formats:
    1. <<expression=result>>number  (tagged)
    2. expression = number          (untagged)
    
    We extract both types.
    
    Example:
        "Friends = 41 * 2 = <<41*2=82>>82
         Johann = 180 - 82 = <<180-82=98>>98
         #### 98"
    
    Returns: ['82', '98']
    """
    steps = []
    
    # Method 1: Extract from <<expression=result>>number
    tagged = re.findall(r'<<[^>]+>>(-?\d+(?:\.\d+)?)', answer_text)
    steps.extend(tagged)
    
    # Method 2: Extract from "= number" for untagged calculations
    lines = answer_text.split('\n')
    for line in lines:
        # Skip lines already processed (have << >> tags)
        if '<<' in line:
            continue
        
        # Skip the final answer line (####)
        if '####' in line:
            continue
        
        # Look for "= number"
        matches = re.findall(r'=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', line)
        if matches:
            # Take last number in line (in case of "10 = 5*2 = 10")
            steps.append(matches[-1].replace(',', ''))
    
    return steps

def prepare_jsonl_file(input_file, output_file):
    """
    Convert JSONL to VERL parquet format with intermediate steps.
    """
    
    print(f"\nProcessing: {input_file}")
    
    data = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                question = item['question']
                answer = item['answer']
                
                # Extract final answer
                numeric_answer = extract_numeric_answer(answer)
                if numeric_answer is None:
                    print(f"  Warning: Could not extract answer from line {line_num}")
                    numeric_answer = ""
                
                # Extract intermediate reasoning steps
                steps = extract_intermediate_steps(answer)
                
                # Create user prompt
                user_content = (
                    f"Question: {question}\n\n"
                    f"Let's solve this step by step.\n\n"
                    f"Then output final answer in last line EXACTLY like:\n"
                    f"#### <number>\n\n"
                    f"Answer:"
                )
                
                # Store metadata for reward function
                reward_model_info = {
                    'ground_truth': numeric_answer,
                    'question': question,
                    'full_answer': answer,
                    'expected_steps': steps
                }
                
                data_source_name = "math"
                
                # Create record
                data.append({
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are a helpful assistant.'
                        },
                        {
                            'role': 'user',
                            'content': user_content
                        }
                    ],
                    'question': question,
                    'full_answer': answer,
                    'target': numeric_answer,
                    'expected_steps': steps,
                    'data_source': data_source_name,
                    'reward_model': reward_model_info,
                    'id': line_num - 1
                })
                
            except json.JSONDecodeError as e:
                print(f"  Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"  Error: Missing field {e} in line {line_num}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save as parquet
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    
    print(f"  ✓ Saved {len(df)} samples to {output_file}")
    
    # Print sample
    if len(df) > 0:
        print(f"\n  Sample:")
        print(f"    Question: {df.iloc[0]['question'][:60]}...")
        print(f"    Expected steps: {df.iloc[0]['expected_steps']}")
        print(f"    Target: {df.iloc[0]['target']}")
    
    # Print statistics
    print(f"\n  Statistics:")
    print(f"    Total samples: {len(df)}")
    print(f"    Samples with targets: {df['target'].notna().sum()}")
    print(f"    Avg steps per sample: {df['expected_steps'].apply(len).mean():.1f}")
    
    return len(df)

def main():
    print("=" * 70)
    print("  Step 2: Preparing Data for VERL (LOCAL REWARDS)")
    print("=" * 70)
    
    # UPDATE THESE PATHS for your environment
    input_dir = '/teamspace/studios/this_studio'
    input_files = {
        'train': f'{input_dir}/data/train.jsonl',
        'valid': f'{input_dir}/data/valid.jsonl',
        'test': f'{input_dir}/data/test.jsonl'
    }
    
    output_dir = f'{input_dir}/data_prepared_local'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_files = {
        'train': f'{output_dir}/train.parquet',
        'valid': f'{output_dir}/valid.parquet',
        'test': f'{output_dir}/test.parquet'
    }
    
    # Check input files exist
    print("\n[1/4] Checking input files...")
    for name, path in input_files.items():
        if Path(path).exists():
            print(f"  ✓ Found {name}: {path}")
        else:
            print(f"  ✗ Missing {name}: {path}")
            print(f"\nError: Cannot find {path}")
            print("Please make sure your data files are in the correct location.")
            return False
    
    # Process each file
    print("\n[2/4] Converting JSONL to parquet with step extraction...")
    
    counts = {}
    for name in ['train', 'valid', 'test']:
        counts[name] = prepare_jsonl_file(
            input_files[name],
            output_files[name]
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("  Step 2 Complete!")
    print("=" * 70)
    
    print("\n[3/4] Summary:")
    print(f"  Training samples:   {counts['train']}")
    print(f"  Validation samples: {counts['valid']}")
    print(f"  Test samples:       {counts['test']}")
    print(f"  Total:              {sum(counts.values())}")
    
    print(f"\n[4/4] Output files:")
    for name, path in output_files.items():
        print(f"  {path}")
    
    print("\n" + "=" * 70)
    print("  ✓ Data prepared with intermediate steps for local rewards!")
    print("=" * 70)
    print("\nNext: Run local_rewards.py to test reward function")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)