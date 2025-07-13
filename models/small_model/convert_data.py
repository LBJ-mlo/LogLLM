#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def convert_json_to_jsonl(input_file, output_file):
    """将JSON格式的数据转换为JSONL格式"""
    
    print(f"正在转换 {input_file} 到 {output_file}...")
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"读取到 {len(data)} 条记录")
    
    # 转换为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 确保数据格式正确
            if 'log_entries' in item and 'label' in item:
                # 将log_entries列表转换为字符串
                log_sequence = '\n'.join(item['log_entries'])
                
                # 转换标签格式
                if item['label'] == 0:
                    label = "正常"
                elif item['label'] == 1:
                    label = "异常"
                else:
                    label = "正常"  # 默认值
                
                # 写入JSONL格式
                jsonl_item = {
                    "log_sequence": log_sequence,
                    "label": label
                }
                f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')
    
    print(f"转换完成！输出文件：{output_file}")

def main():
    # 输入和输出文件路径
    input_file = "../../data/log_sequences_strings.json"
    output_file = "../../data/log_sequences_strings.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    # 执行转换
    convert_json_to_jsonl(input_file, output_file)
    
    # 验证输出文件
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"验证：输出文件包含 {len(lines)} 行")

if __name__ == "__main__":
    main() 