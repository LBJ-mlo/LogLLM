#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# Load the Unsloth training data
with open('data/unsloth_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Unsloth Instruction Tuning Format (Updated):")
print("=" * 60)

print(f"Total training samples: {len(data)}")

# Show first few samples
for i, sample in enumerate(data[:2]):
    print(f"\n--- Sample {i+1} ---")
    print("Instruction:")
    print(sample['instruction'][:300] + "..." if len(sample['instruction']) > 300 else sample['instruction'])
    print("\nInput (Log Sequence):")
    print(sample['input'][:200] + "..." if len(sample['input']) > 200 else sample['input'])
    print("\nOutput:")
    print(sample['output'])
    print("-" * 60)

# Show category distribution in training data
print(f"\n" + "=" * 60)
print("Category distribution in training data:")

categories = {}
unknown_count = 0
confidence_scores = []

for sample in data:
    output = sample['output']
    
    # Extract classification and confidence
    if "**Classification**: UNKNOWN" in output:
        cat = "UNKNOWN"
        unknown_count += 1
    elif "**Classification**: normal" in output:
        cat = "normal"
    elif "**Classification**: service_cascade_failure" in output:
        cat = "service_cascade_failure"
    elif "**Classification**: authentication_failure" in output:
        cat = "authentication_failure"
    elif "**Classification**: resource_exhaustion" in output:
        cat = "resource_exhaustion"
    elif "**Classification**: network_timeout" in output:
        cat = "network_timeout"
    elif "**Classification**: data_corruption" in output:
        cat = "data_corruption"
    else:
        cat = "other"
    
    categories[cat] = categories.get(cat, 0) + 1
    
    # Extract confidence score
    if "**Confidence**: " in output:
        try:
            confidence_line = [line for line in output.split('\n') if "**Confidence**: " in line][0]
            confidence = float(confidence_line.split(': ')[1])
            confidence_scores.append(confidence)
        except:
            pass

for cat, count in categories.items():
    print(f"  {cat}: {count}")

print(f"\nUnknown classifications: {unknown_count}")
if confidence_scores:
    print(f"Confidence score range: {min(confidence_scores):.2f} - {max(confidence_scores):.2f}")
    print(f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")

print(f"\nFormat validation:")
print(f"  - All samples have 'instruction' field: {all('instruction' in sample for sample in data)}")
print(f"  - All samples have 'input' field: {all('input' in sample for sample in data)}")
print(f"  - All samples have 'output' field: {all('output' in sample for sample in data)}")
print(f"  - Input field contains log sequences: {all(len(sample['input']) > 100 for sample in data)}")
print(f"  - Output contains confidence scores: {all('**Confidence**: ' in sample['output'] for sample in data)}") 