#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一格式的大模型微调数据生成器

将现有的日志序列数据转换为统一的Prompt-Output格式，用于大模型微调。
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path

# 异常类型映射
ANOMALY_CATEGORIES = {
    "network_timeout": "Network Timeout",
    "connection_failure": "Connection Failure", 
    "database_error": "Database Error",
    "database_connection_failure": "Database Error",
    "service_failure": "Service Failure",
    "service_cascade_failure": "Service Failure",
    "resource_exhaustion": "Resource Exhaustion",
    "permission_error": "Permission Error",
    "authentication_failure": "Authentication Failure",
    "data_validation_error": "Data Validation Error",
    "data_corruption": "Data Validation Error",
    "external_service_error": "External Service Error",
    "configuration_error": "Configuration Error",
    "concurrent_conflict": "Concurrent Conflict"
}

# 有效类别列表
VALID_CATEGORIES = {
    "Normal", "Network Timeout", "Connection Failure", "Database Error", 
    "Service Failure", "Resource Exhaustion", "Permission Error", 
    "Authentication Failure", "Data Validation Error", "External Service Error", 
    "Configuration Error", "Concurrent Conflict"
}

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_log_sequence(log_entries: List[str]) -> str:
    """将日志条目列表格式化为字符串"""
    return "\n".join(log_entries)

def create_unified_sample(log_entries: List[str], label: Any, category: str = None) -> Dict[str, str]:
    """
    创建统一格式的微调样本
    
    Args:
        log_entries: 日志条目列表
        label: 标签（0/1 或 字符串）
        category: 异常类别（如果有）
    
    Returns:
        统一格式的样本字典，如果类别无效则返回None
    """
    # 格式化日志序列
    log_sequence = format_log_sequence(log_entries)
    
    # 构建Prompt
    prompt = "Analyze the following log sequence to determine if it is normal or abnormal. If abnormal, identify its specific category from the predefined list."
    
    # 构建Input
    input_text = f"```\n{log_sequence}\n```"
    
    # 构建Output
    if isinstance(label, int):
        # 处理数字标签 (0: 正常, 1: 异常)
        if label == 0:
            output = {"status": "Normal", "category": "None"}
        else:
            # 对于异常但没有具体类别的情况，跳过
            return None
    else:
        # 处理字符串标签
        if label == "normal" or label == "Normal":
            output = {"status": "Normal", "category": "None"}
        else:
            # 处理异常类别
            category_name = ANOMALY_CATEGORIES.get(category, category) if category else None
            if category_name is None or category_name == "Unknown" or category_name not in VALID_CATEGORIES:
                return None
            output = {"status": "Abnormal", "category": category_name}
    
    return {
        "prompt": prompt,
        "input": input_text,
        "output": json.dumps(output, ensure_ascii=False)
    }

def process_training_data(data_path: str, output_path: str):
    """处理训练数据（train_data.json格式）"""
    print(f"处理训练数据: {data_path}")
    
    data = load_json_data(data_path)
    unified_samples = []
    skipped_count = 0
    
    for item in data:
        sample = create_unified_sample(
            log_entries=item["log_entries"],
            label=item["label"]
        )
        if sample is not None:
            unified_samples.append(sample)
        else:
            skipped_count += 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_samples, f, ensure_ascii=False, indent=2)
    
    print(f"生成 {len(unified_samples)} 个训练样本，跳过 {skipped_count} 个无效样本，保存到: {output_path}")

def process_classification_data(data_path: str, output_path: str):
    """处理异常分类数据（anomaly_classification_dataset.json格式）"""
    print(f"处理异常分类数据: {data_path}")
    
    data = load_json_data(data_path)
    unified_samples = []
    skipped_count = 0
    
    for item in data:
        sample = create_unified_sample(
            log_entries=item["log_entries"],
            label="abnormal",  # 分类数据都是异常
            category=item.get("category")
        )
        if sample is not None:
            unified_samples.append(sample)
        else:
            skipped_count += 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_samples, f, ensure_ascii=False, indent=2)
    
    print(f"生成 {len(unified_samples)} 个分类样本，跳过 {skipped_count} 个无效样本，保存到: {output_path}")

def create_balanced_dataset(train_data_path: str, classification_data_path: str, 
                           output_path: str, normal_ratio: float = 0.5):
    """
    创建平衡的数据集
    
    Args:
        train_data_path: 训练数据路径
        classification_data_path: 分类数据路径
        output_path: 输出路径
        normal_ratio: 正常样本比例
    """
    print("创建平衡数据集...")
    
    # 加载训练数据
    train_data = load_json_data(train_data_path)
    
    # 分离正常和异常样本
    normal_samples = []
    abnormal_samples = []
    
    for item in train_data:
        sample = create_unified_sample(
            log_entries=item["log_entries"],
            label=item["label"]
        )
        if sample is not None:
            if item["label"] == 0:
                normal_samples.append(sample)
            else:
                abnormal_samples.append(sample)
    
    # 加载分类数据（都是异常）
    classification_data = load_json_data(classification_data_path)
    for item in classification_data:
        sample = create_unified_sample(
            log_entries=item["log_entries"],
            label="abnormal",
            category=item.get("category")
        )
        if sample is not None:
            abnormal_samples.append(sample)
    
    # 平衡数据集
    target_normal_count = int(len(abnormal_samples) * normal_ratio / (1 - normal_ratio))
    if len(normal_samples) > target_normal_count:
        normal_samples = normal_samples[:target_normal_count]
    
    # 合并数据集
    balanced_samples = normal_samples + abnormal_samples
    
    # 打乱数据
    import random
    random.shuffle(balanced_samples)
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_samples, f, ensure_ascii=False, indent=2)
    
    print(f"生成平衡数据集: {len(balanced_samples)} 个样本")
    print(f"  - 正常样本: {len(normal_samples)}")
    print(f"  - 异常样本: {len(abnormal_samples)}")
    print(f"保存到: {output_path}")

def main():
    """主函数"""
    # 设置路径
    data_dir = Path("data")
    output_dir = Path("data/unified_finetune")
    output_dir.mkdir(exist_ok=True)
    
    # 输入文件路径
    train_data_path = data_dir / "train_data.json"
    validation_data_path = data_dir / "validation_data.json"
    test_data_path = data_dir / "test_data.json"
    classification_data_path = data_dir / "anomaly_classification_dataset.json"
    
    # 输出文件路径
    unified_train_path = output_dir / "unified_train_data.json"
    unified_val_path = output_dir / "unified_validation_data.json"
    unified_test_path = output_dir / "unified_test_data.json"
    unified_classification_path = output_dir / "unified_classification_data.json"
    balanced_dataset_path = output_dir / "balanced_finetune_dataset.json"
    
    print("开始生成统一格式的微调数据...")
    
    # 处理训练数据
    if train_data_path.exists():
        process_training_data(str(train_data_path), str(unified_train_path))
    
    # 处理验证数据
    if validation_data_path.exists():
        process_training_data(str(validation_data_path), str(unified_val_path))
    
    # 处理测试数据
    if test_data_path.exists():
        process_training_data(str(test_data_path), str(unified_test_path))
    
    # 处理异常分类数据
    if classification_data_path.exists():
        process_classification_data(str(classification_data_path), str(unified_classification_path))
    
    # 创建平衡数据集
    if train_data_path.exists() and classification_data_path.exists():
        create_balanced_dataset(
            str(train_data_path),
            str(classification_data_path),
            str(balanced_dataset_path),
            normal_ratio=0.4  # 40%正常，60%异常
        )
    
    print("\n数据生成完成！")
    print(f"输出目录: {output_dir}")
    
    # 显示样本示例
    if unified_train_path.exists():
        print("\n样本示例:")
        with open(unified_train_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            if samples:
                print(json.dumps(samples[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 