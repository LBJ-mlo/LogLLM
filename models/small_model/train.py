#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

def run_command(command, description):
    """运行命令并显示输出"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("输出:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def main():
    print("日志异常检测小模型训练启动脚本")
    print("="*60)
    
    # 1. 检查数据文件
    json_file = "../../data/log_sequences_strings.json"
    jsonl_file = "../../data/log_sequences_strings.jsonl"
    
    if not os.path.exists(json_file):
        print(f"错误: 数据文件 {json_file} 不存在")
        print("请确保已生成日志序列数据")
        return
    
    # 2. 转换数据格式
    if not os.path.exists(jsonl_file):
        print("数据格式转换...")
        if not run_command("python convert_data.py", "转换JSON到JSONL格式"):
            print("数据转换失败，退出训练")
            return
    else:
        print("JSONL格式数据已存在，跳过转换")
    
    # 3. 检查依赖
    print("\n检查依赖...")
    try:
        import torch
        import transformers
        import sklearn
        print("✓ 所有依赖已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 4. 开始训练
    print("\n开始模型训练...")
    if run_command("python moe_log_detector.py", "MoE模型训练"):
        print("\n✓ 训练完成！")
        print("模型文件保存在:")
        print("  - ./moe_results/ (检查点)")
        print("  - ./final_moe_model/ (最终模型)")
        print("  - ./moe_logs/ (训练日志)")
    else:
        print("\n✗ 训练失败")

if __name__ == "__main__":
    main() 