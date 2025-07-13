#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter

def filter_anomaly_data():
    """筛选出只包含异常类别的训练数据"""
    
    # 加载清理后的数据
    with open('../data/cleaned_training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据样本数: {len(data)}")
    
    # 分析原始数据分布
    outputs = [item['output'] for item in data]
    print("原始数据分布:")
    print(Counter(outputs))
    
    # 只保留异常类别数据
    anomaly_classes = {
        'service_cascade_failure',
        'authentication_failure', 
        'resource_exhaustion',
        'network_timeout',
        'data_corruption'
    }
    
    anomaly_data = []
    for item in data:
        if item['output'] in anomaly_classes:
            anomaly_data.append(item)
    
    print(f"\n筛选后异常数据样本数: {len(anomaly_data)}")
    
    # 分析异常数据分布
    anomaly_outputs = [item['output'] for item in anomaly_data]
    print("异常数据分布:")
    print(Counter(anomaly_outputs))
    
    # 保存异常数据
    output_file = '../data/anomaly_only_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(anomaly_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n异常数据已保存到: {output_file}")
    
    # 生成一些示例
    print("\n异常数据示例:")
    for i, item in enumerate(anomaly_data[:3]):
        print(f"样本 {i+1} ({item['output']}):")
        print(f"  Input: {item['input'][:100]}...")
        print(f"  Output: {item['output']}")
        print()

if __name__ == "__main__":
    filter_anomaly_data() 