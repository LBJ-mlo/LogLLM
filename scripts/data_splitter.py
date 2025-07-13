#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

def load_data(filename: str) -> List[Dict[str, Any]]:
    """Load data from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data: List[Dict[str, Any]], filename: str):
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_data_by_label(data: List[Dict[str, Any]], val_size: int = 800, test_size: int = 800):
    """
    Split data into train, validation, and test sets while maintaining label distribution
    
    Args:
        data: List of data samples
        val_size: Number of samples for validation set (default: 800)
        test_size: Number of samples for test set (default: 800)
    """
    # Separate data by label
    normal_data = [item for item in data if item['label'] == 0]
    anomaly_data = [item for item in data if item['label'] == 1]
    
    print(f"Original data distribution:")
    print(f"  Normal sequences: {len(normal_data)}")
    print(f"  Anomaly sequences: {len(anomaly_data)}")
    print(f"  Total sequences: {len(data)}")
    
    # Calculate ratios based on desired sizes
    total_size = len(data)
    val_ratio = val_size / total_size
    test_ratio = test_size / total_size
    train_ratio = 1 - val_ratio - test_ratio
    
    print(f"Split ratios: Train={train_ratio:.3f}, Val={val_ratio:.3f}, Test={test_ratio:.3f}")
    
    # Split normal data
    normal_train, normal_temp = train_test_split(
        normal_data, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    normal_val, normal_test = train_test_split(
        normal_temp, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Split anomaly data
    anomaly_train, anomaly_temp = train_test_split(
        anomaly_data, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    anomaly_val, anomaly_test = train_test_split(
        anomaly_temp, 
        test_size=test_ratio/(val_ratio + test_ratio), 
        random_state=42
    )
    
    # Combine splits
    train_data = normal_train + anomaly_train
    val_data = normal_val + anomaly_val
    test_data = normal_test + anomaly_test
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def print_split_statistics(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
    """Print statistics for each data split"""
    print(f"\nData Split Statistics:")
    print("=" * 50)
    
    for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        normal_count = sum(1 for item in split_data if item['label'] == 0)
        anomaly_count = sum(1 for item in split_data if item['label'] == 1)
        total_count = len(split_data)
        
        print(f"{split_name} Set:")
        print(f"  Total sequences: {total_count}")
        print(f"  Normal sequences: {normal_count} ({normal_count/total_count*100:.1f}%)")
        print(f"  Anomaly sequences: {anomaly_count} ({anomaly_count/total_count*100:.1f}%)")
        print()

def main():
    print("Loading log sequences data...")
    
    # Load the original data
    data = load_data("data/log_sequences_strings.json")
    
    print(f"Loaded {len(data)} sequences from log_sequences_strings.json")
    
    # Split the data
    print("\nSplitting data into train/validation/test sets...")
    train_data, val_data, test_data = split_data_by_label(
        data, 
        val_size=800, 
        test_size=800
    )
    
    # Print statistics
    print_split_statistics(train_data, val_data, test_data)
    
    # Save the splits
    print("Saving data splits...")
    save_data(train_data, "data/train_data.json")
    save_data(val_data, "data/validation_data.json")
    save_data(test_data, "data/test_data.json")
    
    print("Data splits saved successfully!")
    print("Files created:")
    print("  - data/train_data.json")
    print("  - data/validation_data.json")
    print("  - data/test_data.json")
    
    # Show sample from each split
    print(f"\nSample from each split:")
    print("=" * 50)
    
    for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        sample = split_data[0]
        print(f"\n{split_name} Set Sample:")
        print(f"  Sequence ID: {sample['sequence_id']}")
        print(f"  Label: {sample['label']} ({'Normal' if sample['label'] == 0 else 'Anomaly'})")
        print(f"  Length: {sample['length']}")
        print(f"  First 2 log entries:")
        for i, log in enumerate(sample['log_entries'][:2]):
            print(f"    {i+1}. {log}")

if __name__ == "__main__":
    main() 