#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据检查脚本
"""

import json
from collections import Counter

def check_training_data(data_path):
    """检查训练数据质量"""
    print("="*60)
    print("训练数据质量检查")
    print("="*60)
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总样本数: {len(data)}")
    
    # 1. 检查基本字段
    print("\n1. 基本字段检查:")
    missing_fields = 0
    for i, sample in enumerate(data):
        if not all(k in sample for k in ['prompt', 'input', 'output']):
            missing_fields += 1
            if missing_fields <= 3:  # 只显示前3个错误
                print(f"  样本 {i} 缺失字段: {list(sample.keys())}")
    
    print(f"  缺失字段样本数: {missing_fields}")
    
    # 2. 检查JSON格式
    print("\n2. JSON格式检查:")
    valid_json = 0
    for i, sample in enumerate(data):
        try:
            json.loads(sample['output'])
            valid_json += 1
        except:
            if i < 3:  # 只显示前3个错误
                print(f"  样本 {i} JSON格式错误: {sample['output'][:50]}...")
    
    print(f"  有效JSON格式: {valid_json}/{len(data)}")
    
    # 3. 检查输出格式
    print("\n3. 输出格式检查:")
    valid_outputs = 0
    categories = []
    for sample in data:
        try:
            output = json.loads(sample['output'])
            if 'status' in output and 'category' in output:
                valid_outputs += 1
                categories.append(output['category'])
        except:
            pass
    
    print(f"  有效输出格式: {valid_outputs}/{len(data)}")
    
    # 4. 类别分布
    print("\n4. 类别分布:")
    category_counts = Counter(categories)
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")
    
    # 5. 序列长度统计
    print("\n5. 日志序列长度统计:")
    lengths = []
    for sample in data:
        # 计算代码块内的行数
        input_text = sample['input']
        if '```' in input_text:
            content = input_text.split('```')[1].strip()
            lines = content.split('\n')
            lengths.append(len(lines))
    
    if lengths:
        print(f"  最短: {min(lengths)} 行")
        print(f"  最长: {max(lengths)} 行")
        print(f"  平均: {sum(lengths)/len(lengths):.1f} 行")
        print(f"  中位数: {sorted(lengths)[len(lengths)//2]} 行")
    
    # 6. 数据质量总结
    print("\n6. 数据质量总结:")
    quality_score = (valid_json / len(data)) * 100
    print(f"  数据质量评分: {quality_score:.1f}%")
    
    if quality_score >= 95:
        print("  ✅ 数据质量良好，可以开始训练")
    elif quality_score >= 80:
        print("  ⚠️  数据质量一般，建议检查后再训练")
    else:
        print("  ❌ 数据质量较差，需要修复后再训练")
    
    print("="*60)

if __name__ == "__main__":
    data_path = "data/unified_finetune/balanced_finetune_dataset.json"
    check_training_data(data_path) 