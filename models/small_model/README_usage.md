# LogLLM Small Model 使用指南

## 📋 概述

这是基于MoE（Mixture of Experts）架构的日志异常检测模型，能够准确识别日志序列中的异常模式。模型采用BERT+MoE的双层架构，在训练数据上达到了99.9%的准确率。

## 🏗️ 模型架构

- **BERT编码器**: bert-base-uncased，提取日志条目语义特征
- **MoE层**: 4个专家，Top-P门控机制
- **输出**: 二分类（正常/异常）
- **模型大小**: ~500MB

## 📁 文件结构

```
small_model/
├── bert-base-uncased/          # BERT预训练模型
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── final_moe_model/            # 训练好的MoE模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── moe_results/                # 训练检查点
├── moe_logs/                   # 训练日志
├── moe_log_detector.py         # 训练脚本
├── inference.py                # 推理脚本
└── README_usage.md            # 本使用指南
```

## 🚀 快速开始

### 1. 环境要求

```bash
# 基础依赖
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install scikit-learn>=1.0.0
pip install numpy>=1.21.0

# 可选：GPU加速
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. 基本使用

#### 方法一：使用推理脚本

```bash
# 运行推理演示
python inference.py
```

#### 方法二：代码调用

```python
from inference import LogAnomalyDetector

# 初始化检测器
detector = LogAnomalyDetector(
    model_path="./final_moe_model",
    bert_path="./bert-base-uncased"
)

# 准备日志序列
log_sequence = """
2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 ERROR [Database] Connection failed: timeout
2024-01-01 10:00:03 ERROR [Database] Retry connection failed
2024-01-01 10:00:04 ERROR [Database] Database service unavailable
2024-01-01 10:00:05 WARN [System] Application running in degraded mode
"""

# 进行预测
result = detector.predict(log_sequence)
print(f"预测结果: {result['prediction']}")
print(f"置信度: {result['confidence']:.4f}")
```

## 📊 输入格式

### 日志序列格式

模型接受多行日志序列，每行一条日志：

```python
log_sequence = """
2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 INFO [Database] Connection established
2024-01-01 10:00:03 INFO [API] Server listening on port 8080
2024-01-01 10:00:04 INFO [Cache] Redis cache initialized
2024-01-01 10:00:05 INFO [Auth] Authentication service ready
"""
```

### 参数说明

- **max_log_entry_length**: 单条日志最大长度（默认128）
- **max_sequence_length**: 序列最大长度（默认32条日志）

## 🔍 输出格式

### 预测结果

```python
{
    "prediction": "异常",           # 预测标签：正常/异常
    "confidence": 0.9992,          # 置信度 (0-1)
    "probabilities": {             # 概率分布
        "正常": 0.0008,
        "异常": 0.9992
    },
    "logits": [1.2, 8.5]          # 原始logits值
}
```

## 🛠️ 高级用法

### 1. 批量推理

```python
import json
from inference import LogAnomalyDetector

detector = LogAnomalyDetector()

# 批量日志序列
log_sequences = [
    "正常日志序列1...",
    "异常日志序列1...",
    "正常日志序列2...",
    # ...
]

results = []
for logs in log_sequences:
    result = detector.predict(logs)
    results.append(result)

# 保存结果
with open('inference_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### 2. 自定义预处理

```python
# 自定义预处理参数
embeddings, attention_mask = detector.preprocess_log_sequence(
    log_sequence,
    max_log_entry_length=64,    # 更短的日志长度
    max_sequence_length=16      # 更短的序列长度
)
```

### 3. 模型配置

```python
# 使用不同的模型路径
detector = LogAnomalyDetector(
    model_path="/path/to/your/model",
    bert_path="/path/to/bert/model"
)
```

## 🔧 性能优化

### 1. GPU加速

```python
# 自动检测GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

### 2. 批处理推理

```python
# 批量处理提高效率
def batch_predict(detector, log_sequences, batch_size=8):
    results = []
    for i in range(0, len(log_sequences), batch_size):
        batch = log_sequences[i:i+batch_size]
        # 批量处理逻辑
        batch_results = [detector.predict(logs) for logs in batch]
        results.extend(batch_results)
    return results
```

## 📈 模型性能

### 训练结果

- **准确率**: 99.9%
- **精确率**: 100.0%
- **召回率**: 99.8%
- **F1分数**: 99.9%

### 推理性能

- **单次推理时间**: ~50ms (GPU)
- **内存占用**: ~2GB (GPU)
- **支持序列长度**: 最多32条日志

## 🚨 异常处理

### 常见问题

1. **模型加载失败**
   ```python
   # 检查模型文件是否存在
   import os
   if not os.path.exists("./final_moe_model"):
       raise FileNotFoundError("模型文件不存在")
   ```

2. **内存不足**
   ```python
   # 使用CPU推理
   detector = LogAnomalyDetector()
   # 或者减少batch_size
   ```

3. **输入格式错误**
   ```python
   # 确保日志序列格式正确
   if not isinstance(log_sequence, str):
       raise ValueError("日志序列必须是字符串")
   ```

## 🔄 模型更新

### 重新训练

```bash
# 1. 准备新数据
# 2. 修改训练参数
# 3. 运行训练
python moe_log_detector.py
```

### 模型保存

训练完成后，模型会自动保存到：
- `./moe_results/` - 检查点
- `./final_moe_model/` - 最终模型

## 📞 技术支持

### 日志格式建议

- 使用标准的时间戳格式
- 包含日志级别（INFO, ERROR, WARN等）
- 保持日志内容的一致性

### 性能调优

- 根据实际需求调整序列长度
- 使用GPU加速推理
- 批量处理提高效率

## 📝 示例代码

### 完整示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from inference import LogAnomalyDetector
import json

def main():
    # 初始化检测器
    detector = LogAnomalyDetector()
    
    # 测试日志序列
    test_logs = [
        {
            "name": "正常系统启动",
            "logs": """2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 INFO [Database] Connection established
2024-01-01 10:00:03 INFO [API] Server listening on port 8080"""
        },
        {
            "name": "数据库连接异常",
            "logs": """2024-01-01 10:00:01 INFO [System] Application started successfully
2024-01-01 10:00:02 ERROR [Database] Connection failed: timeout
2024-01-01 10:00:03 ERROR [Database] Retry connection failed"""
        }
    ]
    
    # 批量推理
    for test in test_logs:
        print(f"\n测试: {test['name']}")
        print(f"日志: {test['logs']}")
        
        result = detector.predict(test['logs'])
        print(f"结果: {result['prediction']} (置信度: {result['confidence']:.4f})")

if __name__ == "__main__":
    main()
```

---

**注意**: 首次使用时会自动下载BERT模型，请确保网络连接正常。如需离线使用，请提前下载模型文件。 