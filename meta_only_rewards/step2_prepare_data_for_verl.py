"""
Step 2: Prepare Data for VERL Training

Converts your JSONL files to VERL-compatible parquet format.
Uses messages format to work with chat templates.
"""

import json
import pandas as pd
import re
from pathlib import Path


def extract_numeric_answer(answer_text):
    """
    Extract final numeric answer from GSM8K format.
    
    GSM8K answers end with: #### <number>
    Example: "... calculations #### 108"
    """
    # Look for #### followed by number
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        # Remove commas from number
        return match.group(1).replace(',', '')
    
    # Fallback: try to find any number at the end
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return None


def prepare_jsonl_file(input_file, output_file):
    """
    Convert JSONL to VERL parquet format with messages.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output parquet file
    """
    
    print(f"\nProcessing: {input_file}")
    
    data = []
    
    # Read JSONL file
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                
                question = item['question']
                answer = item['answer']
                
                # Extract numeric answer
                numeric_answer = extract_numeric_answer(answer)
                
                if numeric_answer is None:
                    print(f"  Warning: Could not extract answer from line {line_num}")
                    numeric_answer = ""
                
                # Create user message content
                user_content = (
                    f"Question: {question}\n\n"
                    f"Let's solve this step by step.\n\n"
                    f"Then output final answer in last line EXACTLY like:\n"
                    f"#### <number>\n\n"
                    f"Answer:"
                )
                
                reward_model_info = {
                    'ground_truth': numeric_answer,
                    'question': question,
                    'full_answer': answer
                }
                
                data_source_name = "math"
                
                # Use messages format for chat template
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
    
    print(f"  Saved {len(df)} samples to {output_file}")
    
    # Print sample
    if len(df) > 0:
        print(f"\n  Sample messages:")
        msgs = df.iloc[0]['messages']
        for msg in msgs:
            print(f"    {msg['role']}: {msg['content'][:80]}...")
        print(f"\n  Sample target: {df.iloc[0]['target']}")
    
    # Print statistics
    print(f"\n  Statistics:")
    print(f"    Total samples: {len(df)}")
    print(f"    Samples with targets: {df['target'].notna().sum()}")
    
    return len(df)


def main():
    print("=" * 70)
    print("  Step 2: Preparing Data for VERL (Messages Format)")
    print("=" * 70)
    
    # Your data files
    input_files = {
        'train': 'data/train.jsonl',
        'valid': 'data/valid.jsonl',
        'test': 'data/test.jsonl'
    }
    
    output_dir = 'data_prepared'
    
    output_files = {
        'train': f'{output_dir}/train.parquet',
        'valid': f'{output_dir}/valid.parquet',
        'test': f'{output_dir}/test.parquet'
    }
    
    # Check input files exist
    print("\n[1/4] Checking input files...")
    for name, path in input_files.items():
        if Path(path).exists():
            print(f"  Found {name}: {path}")
        else:
            print(f"  Missing {name}: {path}")
            print(f"\nError: Cannot find {path}")
            print("Please make sure your data files are in the correct location.")
            return False
    
    # Process each file
    print("\n[2/4] Converting JSONL to parquet...")
    
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
    print(f"  {output_files['train']}")
    print(f"  {output_files['valid']}")
    print(f"  {output_files['test']}")
    
    print("\n" + "=" * 70)
    print("  Data is now in messages format for chat template!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
