#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA微调脚本

使用统一格式的微调数据，对大语言模型（如Qwen/ChatGLM/Baichuan）进行LoRA指令微调。
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ===================== 配置 =====================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 使用Qwen2.5-7B-Instruct
DATA_PATH = "../../data/unified_finetune/balanced_finetune_dataset.json"
OUTPUT_DIR = "./lora_finetune_output"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BATCH_SIZE = 4  # 7B模型建议减小batch size
GRAD_ACCUM = 8  # 增大累积步数以节省显存
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 812
USE_8BIT = True  # 是否使用4bit量化
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 数据加载与处理 =====================
def load_finetune_dataset(path: str) -> Dataset:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 转换为Huggingface Dataset
    return Dataset.from_list(data)

def build_prompt(example: Dict[str, Any]) -> str:
    """拼接prompt和input，作为模型输入"""
    return f"{example['prompt']}\n{example['input']}"

def preprocess_function(examples):
    # 当 batched=True 时，examples 是一个字典，包含所有样本的字段列表
    prompts = examples['prompt']
    inputs = examples['input']
    outputs = examples['output']
    
    # 构建完整的输入文本
    full_inputs = [f"{prompt}\n{input_text}" for prompt, input_text in zip(prompts, inputs)]
    
    # 分词处理
    model_inputs = tokenizer(
        full_inputs,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # 处理标签
    labels = tokenizer(
        outputs,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# ===================== LoRA配置 =====================
def get_lora_model(base_model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",  # 注意力层
             "down_proj",                            # MLP层
            "lm_head",                               # 输出层（语言模型头）
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    return model

# ===================== 损失记录类 =====================
class LossTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.start_time = datetime.now()
        
    def log_train_loss(self, step, loss):
        self.train_losses.append(loss)
        self.steps.append(step)
        print(f"Step {step}: Train Loss = {loss:.4f}")
        
    def log_eval_loss(self, step, loss):
        self.eval_losses.append(loss)
        print(f"Step {step}: Eval Loss = {loss:.4f}")
        
    def save_losses(self):
        loss_data = {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "steps": self.steps,
            "training_time": str(datetime.now() - self.start_time)
        }
        
        # 保存为JSON
        with open(f"{self.output_dir}/loss_history.json", 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.train_losses, label='Train Loss', color='blue')
        if self.eval_losses:
            eval_steps = self.steps[::len(self.steps)//len(self.eval_losses) if len(self.eval_losses) > 0 else 1]
            plt.plot(eval_steps[:len(self.eval_losses)], self.eval_losses, label='Eval Loss', color='red')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"损失历史已保存到: {self.output_dir}/loss_history.json")
        print(f"损失曲线图已保存到: {self.output_dir}/loss_curve.png")

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1. 加载分词器和模型
    print(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model = prepare_model_for_kbit_training(model)
    model = get_lora_model(model)

    # 2. 加载和处理数据
    print(f"加载微调数据: {DATA_PATH}")
    dataset = load_finetune_dataset(DATA_PATH)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # 3. 数据预处理
    print("处理数据...")
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=50,  # 更频繁保存
        eval_steps=50,  # 更频繁评估
        logging_steps=10,  # 更频繁日志
        save_total_limit=3,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # 增加详细日志
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_first_step=True,
        log_level="info",
        # 增加训练信息
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        # 增加进度条
        disable_tqdm=False,
        # 增加训练统计
        warmup_steps=100,
        weight_decay=0.01,
        # 移除早停参数，使用默认设置
    )

    # 5. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 6. 初始化损失跟踪器
    loss_tracker = LossTracker(OUTPUT_DIR)
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 7. 开始训练
    print("="*60)
    print("开始LoRA微调...")
    print(f"模型: {MODEL_NAME}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    print(f"批次大小: {BATCH_SIZE} (累积: {GRAD_ACCUM})")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"LoRA配置: r={LORA_R}, alpha={LORA_ALPHA}")
    print("="*60)
    
    # 训练
    print("\n开始训练，记录损失变化...")
    train_result = trainer.train()
    
    # 保存损失历史
    loss_tracker.save_losses()
    
    # 评估
    print("\n" + "="*60)
    print("训练完成，开始评估...")
    eval_result = trainer.evaluate()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 输出训练统计
    print("\n" + "="*60)
    print("训练统计:")
    print(f"训练损失: {train_result.training_loss:.4f}")
    print(f"验证损失: {eval_result['eval_loss']:.4f}")
    print(f"训练时间: {train_result.metrics['train_runtime']:.2f}秒")
    print(f"每秒步数: {train_result.metrics['train_steps_per_second']:.2f}")
    print(f"模型和分词器已保存到: {OUTPUT_DIR}")
    print("="*60) 