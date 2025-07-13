# 日志异常检测小模型训练

## 概述

这是一个基于MoE（Mixture of Experts）架构的日志序列异常检测模型。该模型使用BERT预训练模型提取日志条目的特征，然后通过Top-P MoE层进行序列级异常检测。

## 模型架构

- **BERT编码器**: 使用bert-base-uncased提取日志条目特征
- **Top-P MoE层**: 4个专家，使用Top-P门控机制
- **多头自注意力**: 8个注意力头
- **分类器**: 2分类（正常/异常）

## 训练策略

### 两阶段训练
1. **第一阶段**: 激活所有专家，无门控机制
2. **第二阶段**: 激活Top-P门控机制

### 超参数
- 学习率: 1e-4
- 批次大小: 64
- 训练轮数: 2
- 专家数量: 4
- Top-P阈值: 0.8

## 数据格式

输入数据应为JSONL格式，每行包含：
```json
{
    "log_sequence": "日志条目1\n日志条目2\n...",
    "label": "正常" 或 "异常"
}
```

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据文件：
   - 将日志序列数据保存为 `../data/log_sequences_strings.jsonl`

3. 运行训练：
```bash
python moe_log_detector.py
```

## 输出

- 训练日志保存在 `./moe_logs/`
- 模型检查点保存在 `./moe_results/`
- 最终模型保存在 `./final_moe_model/`

## 评估指标

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-Score（F1分数） 