#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# Load the classification dataset
with open('data/anomaly_classification_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Anomaly Classification Dataset:")
print("=" * 60)

# Show category distribution
categories = {}
for item in data:
    cat = item["category"]
    categories[cat] = categories.get(cat, 0) + 1

print(f"Total sequences: {len(data)}")
print("Category distribution:")
for cat, count in categories.items():
    print(f"  {cat}: {count}")

# Show example from each category
print("\n" + "=" * 60)
print("Examples from each category:")

for category in sorted(categories.keys()):
    category_samples = [item for item in data if item["category"] == category]
    if category_samples:
        sample = category_samples[0]
        print(f"\n--- {category.upper()} ---")
        print(f"Description: {sample['description']}")
        print(f"Length: {sample['length']} log entries")
        print("Log entries:")
        for i, log in enumerate(sample['log_entries'][:3]):  # Show first 3
            print(f"  {i+1}. {log}")
        if len(sample['log_entries']) > 3:
            print(f"  ... and {len(sample['log_entries']) - 3} more entries")

# Show LLM training format
print("\n" + "=" * 60)
print("LLM Training Format Example:")

with open('data/llm_training_samples.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

if training_data:
    sample = training_data[0]
    print("\nConversation format:")
    for conv in sample['conversations']:
        print(f"\n{conv['from'].upper()}:")
        print(conv['value'][:200] + "..." if len(conv['value']) > 200 else conv['value'])

print(f"\nTotal training samples: {len(training_data)}") 