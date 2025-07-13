#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import Counter

ALLOWED_CLASSES = {
    'normal',
    'service_cascade_failure',
    'authentication_failure',
    'resource_exhaustion',
    'network_timeout',
    'data_corruption'
}

def clean_training_data():
    """严格清理训练数据，只保留六大明确类别"""
    
    # 加载原始数据
    with open('../data/unsloth_training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 分析原始数据分布
    classifications = []
    for item in data:
        output = item['output']
        match = re.search(r'\*\*Classification\*\*:\s*(\w+)', output)
        if match:
            classifications.append(match.group(1))
        else:
            classifications.append('unknown')
    print("原始数据分布:")
    print(Counter(classifications))
    
    # 清理数据：只保留六大类别，移除其它所有样本
    cleaned_data = []
    for item in data:
        output = item['output']
        match = re.search(r'\*\*Classification\*\*:\s*(\w+)', output)
        if not match:
            continue
        classification = match.group(1).lower()
        if classification not in ALLOWED_CLASSES:
            continue
        # 简化instruction
        simplified_instruction = "Classify the following log sequence into one of these categories: normal, service_cascade_failure, authentication_failure, resource_exhaustion, network_timeout, data_corruption. Respond with only the category name."
        # 简化output
        simplified_output = classification
        cleaned_item = {
            "instruction": simplified_instruction,
            "input": item['input'],
            "output": simplified_output
        }
        cleaned_data.append(cleaned_item)
    print(f"清理后数据样本数: {len(cleaned_data)}")
    # 分析清理后的数据分布
    cleaned_classifications = [item['output'] for item in cleaned_data]
    print("清理后数据分布:")
    print(Counter(cleaned_classifications))
    # 保存清理后的数据
    output_file = '../data/cleaned_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"\n清理后的数据已保存到: {output_file}")
    # 生成一些示例
    print("\n清理后的数据示例:")
    for i, item in enumerate(cleaned_data[:3]):
        print(f"样本 {i+1}:")
        print(f"  Instruction: {item['instruction']}")
        print(f"  Input: {item['input'][:100]}...")
        print(f"  Output: {item['output']}")
        print()

if __name__ == "__main__":
    clean_training_data() 