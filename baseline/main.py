# main.py - UPDATED VERSION
# ============================================================================
# 1. DATA LOADING - UPDATED TO USE EXISTING SPLITS
# ============================================================================

import json
import re
import random
from typing import List, Dict, Tuple
from datasets import Dataset
import pandas as pd

class GSM8KDataset:
    def __init__(self, split='train', data_dir='data'):
        """
        Load pre-split GSM8K data.
        
        Args:
            split: 'train', 'valid', or 'test'
            data_dir: directory containing the data files
        """
        self.split = split
        self.data_dir = data_dir
        
        # Load from JSONL (preferred for easy iteration)
        file_path = f"{data_dir}/{split}.jsonl"
        
        print(f"Loading {split} split from {file_path}...")
        self.data = self.load_jsonl(file_path)
        
        print(f"Loaded {split} split: {len(self.data)} examples")
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def extract_final_answer(self, answer_text: str) -> float:
        """Extract the numerical answer from GSM8K answer format."""
        # GSM8K answers end with "#### NUMBER"
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        if match:
            # Remove commas and convert to float
            return float(match.group(1).replace(',', ''))
        return None

