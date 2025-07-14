#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from moe_log_detector import MoEConfig, EncoderWithMoE

class LogAnomalyDetector:
    def __init__(self, model_path="./final_moe_model", bert_path="./bert-base-uncased"):
        """
        初始化日志异常检测器
        
        Args:
            model_path: 训练好的MoE模型路径
            bert_path: BERT模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载BERT模型和分词器
        print("加载BERT模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModel.from_pretrained(bert_path)
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # 加载MoE模型
        print("加载MoE模型...")
        self.moe_model = EncoderWithMoE.from_pretrained(model_path)
        self.moe_model = self.moe_model.to(self.device)
        self.moe_model.eval()
        
        # 标签映射
        self.label_map = {0: "正常", 1: "异常"}
        
        print("模型加载完成！")
    
    def preprocess_log_sequence(self, log_sequence, max_log_entry_length=128, max_sequence_length=32):
        """
        预处理日志序列
        
        Args:
            log_sequence: 日志序列字符串，用换行符分隔
            max_log_entry_length: 单条日志最大长度
            max_sequence_length: 序列最大长度
            
        Returns:
            预处理后的张量
        """
        log_entries = log_sequence.strip().split('\n')
        embeddings_list = []
        
        for log_entry in log_entries:
            # 分词
            encoding = self.tokenizer(
                log_entry,
                max_length=max_log_entry_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # 获取BERT嵌入
            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"]
                )
                last_hidden_state = outputs.last_hidden_state
                # 使用CLS token
                embedding = last_hidden_state[:, 0, :].squeeze(0)
                embeddings_list.append(embedding.cpu())
        
        # 堆叠嵌入
        precomputed_embeddings = torch.stack(embeddings_list)
        
        # 创建序列级别的注意力掩码
        seq_length = precomputed_embeddings.shape[0]
        sequence_attention_mask = torch.ones(seq_length, dtype=torch.bool)
        
        # 截断或填充到 max_sequence_length
        if seq_length > max_sequence_length:
            precomputed_embeddings = precomputed_embeddings[:max_sequence_length, :]
            sequence_attention_mask = sequence_attention_mask[:max_sequence_length]
        else:
            padding_needed = max_sequence_length - seq_length
            padding_tensor = torch.zeros(
                padding_needed,
                precomputed_embeddings.shape[1],
                dtype=precomputed_embeddings.dtype
            )
            padding_mask = torch.zeros(padding_needed, dtype=torch.bool)
            precomputed_embeddings = torch.cat([precomputed_embeddings, padding_tensor], dim=0)
            sequence_attention_mask = torch.cat([sequence_attention_mask, padding_mask], dim=0)
        
        return precomputed_embeddings.unsqueeze(0), sequence_attention_mask.unsqueeze(0)
    
    def predict(self, log_sequence):
        """
        预测日志序列是否异常
        
        Args:
            log_sequence: 日志序列字符串
            
        Returns:
            预测结果字典
        """
        # 预处理
        embeddings, attention_mask = self.preprocess_log_sequence(log_sequence)
        embeddings = embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.moe_model(
                precomputed_embeddings=embeddings,
                #attention_mask=attention_mask
            )
            logits = outputs["logits"]
            # 检查logits是否有效
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("警告: 检测到无效的logits值，使用默认预测")
                return {
                    "prediction": "正常",
                    "confidence": 0.5,
                    "probabilities": {"正常": 0.5, "异常": 0.5},
                    "logits": [0.0, 0.0],
                    "warning": "模型输出包含NaN/Inf值"
                }
            
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            # 确保概率值有效
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                print("警告: 检测到无效的概率值")
                probabilities = torch.tensor([[0.5, 0.5]], device=self.device)
                confidence = 0.5
        
        return {
            "prediction": self.label_map[prediction],
            "confidence": confidence,
            "probabilities": {
                "正常": probabilities[0][0].item(),
                "异常": probabilities[0][1].item()
            },
            "logits": logits[0].cpu().numpy().tolist()
        }

def main():
    """主函数 - 演示推理过程"""
    
    # 初始化检测器
    detector = LogAnomalyDetector()
    
    # 示例日志序列
    example_logs = [
        {
            "name": "正常日志序列",
            "logs": """2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 INFO [Database] Connection established
2024-01-01 10:00:03 INFO [API] Server listening on port 8080
2024-01-01 10:00:04 INFO [Cache] Redis cache initialized
2024-01-01 10:00:05 INFO [Auth] Authentication service ready"""
        },
        {
            "name": "异常日志序列 - 数据库连接失败",
            "logs": """2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 ERROR [Database] Connection failed: timeout
2024-01-01 10:00:03 ERROR [Database] Retry connection failed
2024-01-01 10:00:04 ERROR [Database] Database service unavailable
2024-01-01 10:00:05 WARN [System] Application running in degraded mode"""
        },
        {
            "name": "异常日志序列 - API错误",
            "logs": """2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 INFO [API] Server listening on port 8080
2024-01-01 10:00:03 ERROR [API] 500 Internal Server Error
2024-01-01 10:00:04 ERROR [API] Request timeout
2024-01-01 10:00:05 ERROR [API] Service unavailable"""
        },
        {
            "name": "异常日志序列 - 权限错误",
            "logs": """2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 INFO [Auth] User login attempt
2024-01-01 10:00:03 ERROR [Auth] Permission denied: insufficient privileges
2024-01-01 10:00:04 ERROR [Auth] Access denied to resource
2024-01-01 10:00:05 WARN [System] Security alert: unauthorized access attempt"""
        }
    ]
    
    print("=" * 80)
    print("日志异常检测推理演示")
    print("=" * 80)
    
    # 对每个示例进行推理
    for i, example in enumerate(example_logs, 1):
        print(f"\n示例 {i}: {example['name']}")
        print("-" * 60)
        print("输入日志序列:")
        print(example['logs'])
        print("\n推理结果:")
        
        try:
            result = detector.predict(example['logs'])
            print(f"预测结果: {result['prediction']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"概率分布:")
            print(f"  正常: {result['probabilities']['正常']:.4f}")
            print(f"  异常: {result['probabilities']['异常']:.4f}")
            
            # 显示警告信息
            if 'warning' in result:
                print(f"⚠️  {result['warning']}")
            
            # 添加颜色标识
            if result['prediction'] == "异常":
                print("🔴 检测到异常！")
            else:
                print("🟢 日志正常")
                
        except Exception as e:
            print(f"推理过程中出现错误: {e}")
        
        print("-" * 60)
    
    print("\n推理演示完成！")

if __name__ == "__main__":
    main() 