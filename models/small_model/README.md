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

## 训练过程
<img width="1345" height="783" alt="image" src="https://github.com/user-attachments/assets/7e5dd614-52c6-454c-ba8e-0f0384dd6133" />
<img width="1361" height="792" alt="image" src="https://github.com/user-attachments/assets/0aa279d1-852d-42d7-8956-96b24f5a3788" />
<img width="1353" height="802" alt="image" src="https://github.com/user-attachments/assets/84c6ad8e-22f6-47c4-92d7-1e3b845c1764" />
```bash
stage1_data = [
    {'loss': 1.5013, 'grad_norm': 29.37301254272461, 'learning_rate': 0.0, 'epoch': 0.0},
    {'loss': 1.4489, 'grad_norm': 10.772336959838867, 'learning_rate': 1.8e-06, 'epoch': 0.2},
    {'loss': 1.2695, 'grad_norm': 8.640754699707031, 'learning_rate': 3.8e-06, 'epoch': 0.4},
    {'loss': 1.1841, 'grad_norm': 2.7628719806671143, 'learning_rate': 5.8e-06, 'epoch': 0.6},
    {'loss': 1.0364, 'grad_norm': 4.981997966766357, 'learning_rate': 7.8e-06, 'epoch': 0.8},
    {'loss': 0.9673, 'grad_norm': 9.481902122497559, 'learning_rate': 9.800000000000001e-06, 'epoch': 1.0},
    {'loss': 0.7358, 'grad_norm': 3.3190104961395264, 'learning_rate': 1.18e-05, 'epoch': 1.2},
    {'loss': 0.4431, 'grad_norm': 2.467216730117798, 'learning_rate': 1.3800000000000002e-05, 'epoch': 1.4},
    {'loss': 0.1932, 'grad_norm': 7.134607315063477, 'learning_rate': 1.58e-05, 'epoch': 1.6},
    {'loss': 0.09, 'grad_norm': 6.032463073730469, 'learning_rate': 1.78e-05, 'epoch': 1.8},
    {'loss': 0.0688, 'grad_norm': 10.950860023498535, 'learning_rate': 1.9800000000000004e-05, 'epoch': 2.0}
]
Data for Stage 2 Training
stage2_data = [
    {'loss': 0.0403, 'grad_norm': 5.444004535675049, 'learning_rate': 0.0, 'epoch': 0.0}, # Corrected epoch from 0.02 to 0.0 for consistency
    {'loss': 0.0463, 'grad_norm': 2.461456060409546, 'learning_rate': 1.8e-06, 'epoch': 0.2},
    {'loss': 0.0313, 'grad_norm': 2.3055379390716553, 'learning_rate': 3.8e-06, 'epoch': 0.4},
    {'loss': 0.0473, 'grad_norm': 5.870611667633057, 'learning_rate': 5.8e-06, 'epoch': 0.6},
    {'loss': 0.0313, 'grad_norm': 0.6006909608840942, 'learning_rate': 7.8e-06, 'epoch': 0.8},
    {'loss': 0.0568, 'grad_norm': 12.453761100769043, 'learning_rate': 9.800000000000001e-06, 'epoch': 1.0},
    {'loss': 0.0382, 'grad_norm': 1.9335322380065918, 'learning_rate': 1.18e-05, 'epoch': 1.2},
    {'loss': 0.0228, 'grad_norm': 3.7193222045898438, 'learning_rate': 1.3800000000000002e-05, 'epoch': 1.4},
    {'loss': 0.0144, 'grad_norm': 3.0188748836517334, 'learning_rate': 1.58e-05, 'epoch': 1.6},
    {'loss': 0.051, 'grad_norm': 13.929482460021973, 'learning_rate': 1.76e-05, 'epoch': 1.8},
    {'loss': 0.0356, 'grad_norm': 0.6225671768188477, 'learning_rate': 1.9600000000000002e-05, 'epoch': 2.0}
]

测试结果：

{'eval_loss': 0.0074135782197117805, 'eval_accuracy': 0.999375, 'eval_precision': 1.0,
'eval_recall': 0.997907949790795,'eval_f1': 0.9989528795811519, 'eval_runtime': 91.2377,
'eval_samples_per_second': 17.537, 'eval_steps_per_second': 0.274, 'epoch': 2.0}
```
================================================================================
日志异常检测推理演示（批量测试&耗时统计）
================================================================================
Batch 1: 16条日志，耗时: 586.48 ms
Batch 2: 16条日志，耗时: 434.12 ms
Batch 3: 16条日志，耗时: 436.09 ms
Batch 4: 16条日志，耗时: 436.73 ms
Batch 5: 16条日志，耗时: 429.89 ms
Batch 6: 16条日志，耗时: 430.54 ms
Batch 7: 4条日志，耗时: 109.03 ms
全部100条日志批量推理总耗时: 2862.88 ms, 平均每条: 28.63 ms

## 稠密模型相关指标
测试集: loss: 0.0255，准确率: 0.9894, 精确率: 1.0000, 召回率: 0.9651, F1: 0.9822
================================================================================
稠密模型日志异常检测推理演示（批量测试&耗时统计）
================================================================================
Batch 1: 16条日志，耗时: 622.65 ms
Batch 2: 16条日志，耗时: 484.92 ms
Batch 3: 16条日志，耗时: 485.29 ms
Batch 4: 16条日志，耗时: 486.89 ms
Batch 5: 16条日志，耗时: 484.66 ms
Batch 6: 16条日志，耗时: 482.38 ms
Batch 7: 4条日志，耗时: 119.92 ms
全部100条日志批量推理总耗时: 3166.71 ms, 平均每条: 31.67 ms

<img width="1761" height="912" alt="image" src="https://github.com/user-attachments/assets/dc09b88e-015a-41d7-a3ac-b52175336bee" />

<img width="489" height="355" alt="image" src="https://github.com/user-attachments/assets/ea21d933-856b-4a01-a77c-a8ccbd813247" />

predict_bleu-4                 =     0.9551
  predict_model_preparation_time =     0.0034
  predict_rouge-1                =     4.1764
  predict_rouge-2                =     0.1208
  predict_rouge-l                =     1.2712
  predict_runtime                = 0:06:58.12
  predict_samples_per_second     =      0.048
  predict_steps_per_second       =      0.048
