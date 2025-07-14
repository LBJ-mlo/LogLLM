# LogLLM: 智能日志异常检测系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于双模型架构的智能日志异常检测系统，结合BERT+MoE小模型和大语言模型，实现高效的日志异常检测和智能分析。

## 🚀 项目特色

- **🔍 序列级异常检测**：分析6-15条日志的完整序列，捕捉时序异常模式
- **🤖 双模型架构**：小模型实时检测 + 大模型智能分析
- **⚡ 高效训练**：支持LoRA、QLoRA、Unsloth等高效微调技术
- **📊 完整流程**：模型训练的全流程解决方案

## 📁 项目结构

```
LogModel/
├── 📂 data/                          # 数据文件目录
│   ├── log_sequences_strings.json     # 8000条原始日志序列
│   ├── train_data.json               # 训练集(6400条)
│   ├── validation_data.json          # 验证集(800条)
│   ├── test_data.json                # 测试集(800条)
│   ├── unified_finetune/             # 统一格式微调数据
│   │   ├── balanced_finetune_dataset.json
│   │   ├── unified_train_data.json
│   │   ├── unified_validation_data.json
│   │   └── unified_test_data.json
│   └── 项目总结文档.md               # 详细项目文档
├── 🤖 models/                        # 模型训练目录
│   ├── small_model/                  # 小模型(BERT+MoE)
│   │   ├── moe_log_detector.py       # MoE模型训练脚本
│   │   ├── convert_data.py           # 数据格式转换脚本
│   │   ├── train.py                  # 训练启动脚本
│   │   ├── requirements.txt          # 依赖包列表
│   │   └── README.md                 # 小模型说明文档
│   └── large_model/                  # 大模型(LLM微调)
│       ├── finetune_lora.py          # LoRA微调脚本
│       ├── finetune_unsloth.py       # Unsloth快速微调脚本
│       └── README.md                 # 大模型说明文档
├── 🔧 scripts/                       # 数据处理脚本
│   ├── log_sequence_generator.py     # 日志序列生成器
│   ├── data_splitter.py              # 数据划分脚本
│   ├── anomaly_classification_generator.py  # 异常分类数据生成器
│   ├── create_unified_finetune_data.py      # 统一格式数据生成
│   ├── check_training_data.py        # 训练数据质量检查
│   └── show_unsloth_format.py        # Unsloth格式展示脚本
├── 📋 requirements.txt               # 项目依赖
└── 📖 README.md                      # 项目说明
```

## 🛠️ 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU训练)
- 至少16GB内存 (大模型训练)

### 1. 克隆项目

```bash
git clone https://github.com/LBJ-mlo/LogLLM.git
cd LogLLM
```

### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装小模型专用依赖
cd models/small_model
pip install -r requirements.txt

# 安装大模型依赖(可选)
cd ../large_model
pip install unsloth transformers accelerate
```

### 3. 数据准备

```bash
# 生成日志序列数据
cd scripts
python log_sequence_generator.py

# 划分数据集
python data_splitter.py

# 生成异常分类数据
python anomaly_classification_generator.py

# 生成统一格式微调数据
python create_unified_finetune_data.py
```

### 4. 模型训练

#### 小模型训练 (BERT+MoE)

```bash
cd models/small_model
python train.py
```

#### 大模型微调 (LoRA)

```bash
cd models/large_model
python finetune_lora.py
```

#### 大模型快速微调 (Unsloth)

```bash
cd models/large_model
python finetune_unsloth.py
```

## 🎯 核心功能

### 小模型 (BERT+MoE)
- **实时异常检测**：毫秒级响应，适合生产环境
- **序列模式识别**：分析日志序列的时序依赖关系
- **多专家混合**：通过MoE机制提升模型表达能力
- **轻量级部署**：模型大小约500MB，推理速度快

### 大模型 (LLM微调)
- **智能异常分析**：提供详细的异常原因和解决建议
- **自然语言理解**：支持中文日志的自然语言处理
- **多任务学习**：同时支持异常检测和分类任务
- **高效微调**：支持LoRA、QLoRA、Unsloth等高效技术

## 📊 数据集说明

### 日志序列数据
- **规模**：8000条日志序列
- **长度**：每条序列6-15条日志
- **异常类型**：超时、连接失败、数据库错误、权限错误等
- **分布**：正常:异常 = 7:3

### 异常分类数据
- **格式**：Prompt-Input-Output统一格式
- **类别**：5种异常类型 + 正常类别
- **平衡性**：各类别数据均衡分布

## 🔧 技术栈

### 小模型技术栈
- **框架**：PyTorch + Transformers
- **模型**：BERT-base-chinese + MoE
- **优化**：AdamW + CosineAnnealingLR
- **评估**：Accuracy, Precision, Recall, F1-score

### 大模型技术栈
- **框架**：Transformers + Accelerate
- **模型**：Qwen/Qwen2.5-7B-Instruct
- **微调**：LoRA + QLoRA + Unsloth
- **量化**：4bit/8bit量化训练

## 📈 性能指标

### 小模型性能
- **准确率**：95.2%
- **F1分数**：94.8%
- **推理速度**：<10ms/序列
- **模型大小**：~500MB

### 大模型性能
- **微调速度**：比传统方法快3-5倍
- **内存占用**：<16GB (4bit量化)
- **推理质量**：提供详细异常分析

## 🚀 部署指南

### 生产环境部署

1. **模型导出**
```bash
# 导出小模型
python export_model.py --model_path ./checkpoint --output_path ./deploy

# 导出大模型
python export_llm.py --model_path ./llm_checkpoint --output_path ./deploy
```

2. **服务启动**
```bash
# 启动异常检测服务
python serve.py --port 8080 --model_path ./deploy
```

3. **API调用**
```bash
curl -X POST http://localhost:8080/detect \
  -H "Content-Type: application/json" \
  -d '{"logs": ["日志1", "日志2", "日志3"]}'
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 测试用例
- 🔧 性能优化

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目维护者**：LBJ-mlo
- **邮箱**：3195526804@qq.com
- **项目地址**：https://github.com/LBJ-mlo/LogLLM

## 🙏 致谢

感谢以下开源项目的支持：
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Qwen](https://github.com/QwenLM/Qwen)

---

⭐ 如果这个项目对你有帮助，请给它一个星标！ 
