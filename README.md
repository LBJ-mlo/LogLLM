# 日志异常检测系统

## 项目概述

本项目构建了一个完整的日志异常检测系统，采用双模型架构：
- **小模型**：基于BERT+MoE的序列级异常检测器
- **大模型**：基于大语言模型的异常分类和分析器

## 项目结构

```
LogModel/
├── data/                          # 数据文件目录
│   ├── log_sequences_strings.json     # 8000条原始日志序列
│   ├── train_data.json               # 训练集(6400条)
│   ├── validation_data.json          # 验证集(800条)
│   ├── test_data.json                # 测试集(800条)
│   ├── unsloth_training_data.json    # Unsloth格式训练数据
│   ├── anomaly_classification_dataset.json  # 异常分类数据集
│   └── 项目总结文档.md               # 详细项目文档
├── models/                        # 模型训练目录
│   ├── small_model/              # 小模型训练
│   │   ├── moe_log_detector.py   # MoE模型训练脚本
│   │   ├── convert_data.py       # 数据格式转换脚本
│   │   ├── train.py              # 训练启动脚本
│   │   ├── requirements.txt      # 依赖包列表
│   │   └── README.md             # 小模型说明文档
│   └── large_model/              # 大模型微调
│       └── README.md             # 大模型说明文档
├── scripts/                      # 脚本文件目录
│   ├── log_sequence_generator.py # 日志序列生成器
│   ├── data_splitter.py          # 数据划分脚本
│   ├── anomaly_classification_generator.py  # 异常分类数据生成器
│   ├── show_classification_data.py # 数据展示脚本
│   └── show_unsloth_format.py    # Unsloth格式展示脚本
├── config/                       # 配置文件目录
├── utils/                        # 工具函数目录
└── README.md                     # 项目总览
```

## 快速开始

### 1. 环境准备
```bash
# 安装小模型依赖
cd models/small_model
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 生成日志序列数据
cd scripts
python log_sequence_generator.py

# 划分数据集
python data_splitter.py

# 生成异常分类数据
python anomaly_classification_generator.py
```

### 3. 模型训练
```bash
# 训练小模型
cd models/small_model
python train.py
```

### 4. 大模型微调（待开发）
```bash
cd models/large_model
# 待开发脚本
```

## 核心特性

- **序列级异常检测**：分析6-15条日志的完整序列
- **多异常类型支持**：超时、连接失败、数据库错误等
- **中文日志优化**：专门针对中文日志设计
- **双模型架构**：小模型实时检测 + 大模型详细分析

## 技术栈

- **小模型**：PyTorch + Transformers + BERT + MoE
- **大模型**：Unsloth + QLoRA + 大语言模型
- **数据处理**：Python + scikit-learn + JSON

## 项目状态

- ✅ 小模型训练部分已完成
- 🔄 大模型微调部分待开发
- 🔄 推理服务待开发

## 详细文档

更多详细信息请参考 `data/项目总结文档.md` 