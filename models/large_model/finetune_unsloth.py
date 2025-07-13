#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth快速微调脚本 - 显著加速训练
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset, Dataset
import torch
from datetime import datetime
import matplotlib.pyplot as plt

# 导入Unsloth
from unsloth import FastLanguageModel

# ===================== 配置 =====================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "/root/autodl-tmp/LogModel/data/unified_finetune/balanced_finetune_dataset.json"
OUTPUT_DIR = "/root/autodl-tmp/unsloth_finetune_output"  # 数据盘
CACHE_DIR = "/root/autodl-tmp/hf_cache"  # 数据盘缓存
MAX_SEQ_LENGTH = 812
BATCH_SIZE = 4  # Unsloth支持更大的batch size
GRAD_ACCUM = 4
EPOCHS = 3
LEARNING_RATE = 2e-4  # Unsloth推荐更高的学习率
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# ===================== 数据加载与处理 =====================
def load_finetune_dataset(path: str) -> Dataset:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def format_prompt(example):
    """格式化训练数据为Unsloth格式"""
    prompt = example['prompt']
    input_text = example['input']
    output = example['output']
    
    # 构建完整的对话格式
    full_prompt = f"{prompt}\n{input_text}"
    
    return {
        "text": f"<|im_start|>user\n{full_prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    }

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
        plt.title('Unsloth Training and Evaluation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"损失历史已保存到: {self.output_dir}/loss_history.json")
        print(f"损失曲线图已保存到: {self.output_dir}/loss_curve.png")

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 设置环境变量 - 所有缓存到数据盘
    os.environ['HF_HOME'] = CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("="*60)
    print("开始Unsloth快速微调训练")
    print("="*60)
    print(f"模型: {MODEL_NAME}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"缓存目录: {CACHE_DIR}")
    
    # 1. 加载模型和分词器 (Unsloth优化)
    print(f"\n加载模型: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # None for auto detection
        load_in_4bit=True,  # 使用4bit量化
    )
    
    # 2. 添加LoRA适配器
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # 3. 加载和处理数据
    print(f"加载微调数据: {DATA_PATH}")
    dataset = load_finetune_dataset(DATA_PATH)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    # 4. 格式化数据
    print("格式化数据...")
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_prompt, remove_columns=eval_dataset.column_names)
    
    # 5. 设置训练参数
    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            max_steps=None,  # 使用epochs
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            save_strategy="steps",
            save_steps=50,
            eval_strategy="steps",
            eval_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir=f"{OUTPUT_DIR}/logs",
            report_to="none",
            dataloader_pin_memory=False,
            dataloader_num_workers=2,
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 6. 初始化损失跟踪器
    loss_tracker = LossTracker(OUTPUT_DIR)
    
    # 7. 创建训练器
    trainer = transformers.Trainer(**trainer_kwargs)
    
    # 8. 开始训练
    print(f"\n训练配置:")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    print(f"批次大小: {BATCH_SIZE} (累积: {GRAD_ACCUM})")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"Unsloth优化: 启用")
    print("="*60)
    
    # 训练
    print("\n开始Unsloth快速训练...")
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