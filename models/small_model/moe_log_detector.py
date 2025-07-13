#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import transformers
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import PreTrainedModel
import datetime
import random
import uuid
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- MoEConfig 定义 ---
class MoEConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        encoder_model_name="bert-base-uncased",
        num_experts=4,
        p_threshold=0.8,
        load_balancing_loss_weight=0.01,
        initializer_range=0.02,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=256,
        max_position_embeddings=64,
        vocab_size=30522,
        type_vocab_size=2,
        dropout_rate=0.1,
        pruned_heads=None,
        **kwargs
    ):
        self.encoder_model_name = encoder_model_name
        self.num_experts = num_experts
        self.p_threshold = p_threshold
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = dropout_rate
        self.pruned_heads = pruned_heads
        super().__init__(**kwargs)

# --- TopPMoE 层 ---
class TopPMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.training_phase = 1  # 1: 无门控，2: 有门控

        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout_rate
        )
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)

        # MoE FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.intermediate_size, config.hidden_size)
            ) for _ in range(config.num_experts)
        ])
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.p_threshold = config.p_threshold

    def set_training_phase(self, phase: int):
        self.training_phase = phase

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, hidden_size = x.size()
        num_heads = self.config.num_attention_heads

        # 准备注意力掩码
        if attention_mask is not None:
            attn_mask = ~attention_mask.unsqueeze(2).expand(-1, -1, seq_length)
            attn_mask = attn_mask.to(dtype=torch.bool)
            attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
        else:
            attn_mask = None

        # 将 x 转换为适合 MultiheadAttention 的格式
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        attn_output = attn_output.transpose(0, 1)
        x = x.transpose(0, 1)
        x = x + self.dropout(attn_output)
        x = self.attention_layer_norm(x)

        # MoE FFN
        if self.training_phase == 1:
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
            ffn_output = torch.mean(expert_outputs, dim=-1)
        else:
            gates = self.gate(x)
            gates = torch.softmax(gates, dim=-1)
            sorted_gates, _ = torch.sort(gates, descending=True, dim=-1)
            cumsum_gates = torch.cumsum(sorted_gates, dim=-1)
            top_p_mask = cumsum_gates < self.p_threshold
            top_p_mask = torch.cat([top_p_mask, torch.zeros_like(top_p_mask[..., :1])], dim=-1)
            top_p_mask = top_p_mask[..., :-1]
            gates = gates * top_p_mask.float()
            gate_sum = torch.sum(gates, dim=-1, keepdim=True)
            gates = gates / (gate_sum + 1e-10)
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
            ffn_output = torch.sum(expert_outputs * gates.unsqueeze(-2), dim=-1)

        x = x + self.dropout(ffn_output)
        x = self.ffn_layer_norm(x)

        load_balancing_loss = torch.tensor(0.0, device=x.device) if self.training_phase == 1 else \
            torch.mean(torch.square(torch.mean(gates, dim=(0, 1)) - 1.0 / self.config.num_experts))

        return x, load_balancing_loss

# --- EncoderWithMoE 模型 ---
class EncoderWithMoE(PreTrainedModel):
    config_class = MoEConfig
    base_model_prefix = "encoder_with_moe"

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.config = config
        self.bert2moe = nn.Linear(768, config.hidden_size)
        self.moe_layers = nn.ModuleList([TopPMoE(config) for _ in range(6)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.post_init()

    def set_training_phase(self, phase: int):
        for moe_layer in self.moe_layers:
            moe_layer.set_training_phase(phase)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        precomputed_embeddings: torch.Tensor = None,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        precomputed_embeddings = self.bert2moe(precomputed_embeddings)
        if precomputed_embeddings is None:
            raise ValueError("必须提供 precomputed_embeddings。")
        if precomputed_embeddings.dim() != 3 or precomputed_embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"precomputed_embeddings 的形状必须为 (batch_size, sequence_length, hidden_size)，"
                f"但得到 {precomputed_embeddings.shape}"
            )
        batch_size, seq_length = precomputed_embeddings.size()[:2]
        if seq_length > self.config.max_position_embeddings:
            raise ValueError(
                f"序列长度 {seq_length} 超过最大位置编码长度 {self.config.max_position_embeddings}"
            )

        position_ids = torch.arange(seq_length, dtype=torch.long, device=precomputed_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = precomputed_embeddings + position_embeddings

        moe_output = embeddings
        total_moe_loss = 0.0
        for moe_layer in self.moe_layers:
            moe_output, moe_load_balancing_loss = moe_layer(moe_output, attention_mask)
            total_moe_loss += moe_load_balancing_loss

        moe_output = self.layer_norm(moe_output)
        cls_output = torch.mean(moe_output, dim=1) # 对序列长度维度 (dim=1) 取均值
        logits = self.classifier(cls_output)

        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            total_loss = classification_loss + self.config.load_balancing_loss_weight * total_moe_loss

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Invalid logits detected")
        if labels is not None and total_loss is not None and (torch.isnan(total_loss) or torch.isinf(total_loss)):
            print("Invalid loss detected")

        return {"logits": logits, "loss": total_loss}

# --- LogSequenceDataset ---
class LogSequenceDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, bert_model, max_log_entry_length: int = 128, max_sequence_length: int = 64, pooling_method: str = "cls"):
        self.data = []
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.max_log_entry_length = max_log_entry_length
        self.max_sequence_length = max_sequence_length
        self.pooling_method = pooling_method
        self.label_map = {"正常": 0, "异常": 1}

        self.bert_model.eval()
        for param in self.bert_model.parameters():
            param.requires_grad = False

        print(f"正在加载和预处理数据文件：{data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                self.data.append(entry)
                if (i + 1) % 1000 == 0:
                    print(f"已加载 {i + 1} 条日志序列...")
        print(f"数据加载完成，总计 {len(self.data)} 条日志序列。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        log_sequence_text = item["log_sequence"]
        label = self.label_map[item["label"]]

        log_entries = log_sequence_text.strip().split('\n')
        embeddings_list = []

        device = next(self.bert_model.parameters()).device # 获取bert_model所在的设备

        for log_entry in log_entries:
            encoding = self.tokenizer(
                log_entry,
                max_length=self.max_log_entry_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            encoding = {k: v.to(device) for k, v in encoding.items()} # 将输入移到 GPU

            with torch.no_grad():
                outputs = self.bert_model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"] # BERT 内部的掩码
                )
                last_hidden_state = outputs.last_hidden_state

                if self.pooling_method == "cls":
                    embedding = last_hidden_state[:, 0, :].squeeze(0)
                else:
                    masked_hidden = last_hidden_state * encoding["attention_mask"].unsqueeze(-1)
                    embedding = masked_hidden.sum(dim=1) / encoding["attention_mask"].sum(dim=1, keepdim=True)
                    embedding = embedding.squeeze(0)

                embeddings_list.append(embedding.cpu()) # 将嵌入移回 CPU

        # 堆叠嵌入
        precomputed_embeddings = torch.stack(embeddings_list)

        # 创建序列级别的注意力掩码
        seq_length = precomputed_embeddings.shape[0]
        sequence_attention_mask = torch.ones(seq_length, dtype=torch.bool)

        # 截断或填充到 max_sequence_length
        if seq_length > self.max_sequence_length:
            precomputed_embeddings = precomputed_embeddings[:self.max_sequence_length, :]
            sequence_attention_mask = sequence_attention_mask[:self.max_sequence_length]
        else:
            padding_needed = self.max_sequence_length - seq_length
            padding_tensor = torch.zeros(
                padding_needed,
                precomputed_embeddings.shape[1],
                dtype=precomputed_embeddings.dtype
            )
            padding_mask = torch.zeros(padding_needed, dtype=torch.bool)
            precomputed_embeddings = torch.cat([precomputed_embeddings, padding_tensor], dim=0)
            sequence_attention_mask = torch.cat([sequence_attention_mask, padding_mask], dim=0)

        return {
            "precomputed_embeddings": precomputed_embeddings, # CPU tensor
            "labels": torch.tensor(label, dtype=torch.long), # CPU tensor
            "attention_mask": sequence_attention_mask # 确保这个序列级的掩码被返回
        }

# --- 主训练和测试逻辑 ---
if __name__ == "__main__":
    # 1. 初始化分词器和 BERT 模型
    encoder_model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_checkpoint)
    bert_model = AutoModel.from_pretrained(encoder_model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    print(f"Using device: {device}")

    # 2. 初始化 MoE 模型
    custom_moe_config = MoEConfig(
        encoder_model_name=encoder_model_checkpoint,
        num_experts=4,
        p_threshold=0.8,
        load_balancing_loss_weight=0.01,
        initializer_range=0.02,
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=256,
        max_position_embeddings=16,
        vocab_size=bert_model.config.vocab_size,
        type_vocab_size=bert_model.config.type_vocab_size,
        dropout_rate=0.1,
    )

    model = EncoderWithMoE(custom_moe_config)
    model = model.to(device)
    print("\nMoE 模型初始化完成：")
    print(model)

    # 3. 准备数据集
    max_log_entry_seq_len = 64
    max_log_sequence_len = 16

    # 使用我们生成的数据文件
    data_file = "../../data/log_sequences_strings.jsonl"
    
    dataset_full = LogSequenceDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        bert_model=bert_model,
        max_log_entry_length=max_log_entry_seq_len,
        max_sequence_length=max_log_sequence_len,
        pooling_method="cls"
    )

    # 分割数据集
    train_size = int(0.8 * len(dataset_full))
    test_size = len(dataset_full) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset_full, [train_size, test_size]
    )

    print(f"\n数据集大小：总计 {len(dataset_full)} 条序列。训练集：{len(train_dataset)}，测试集：{len(test_dataset)}")

    # 4. 配置 Trainer
    def compute_metrics(p):
        predictions = p.predictions.argmax(axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    training_args = TrainingArguments(
        output_dir="./moe_results",
        learning_rate=1e-4,
        num_train_epochs=2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        logging_dir="./moe_logs",
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        fp16=True if torch.cuda.is_available() else False
    )

    # 自定义 DataLoader 以启用 pin_memory
    def custom_data_collator(features):
        batch = {}
        batch["precomputed_embeddings"] = torch.stack([f["precomputed_embeddings"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator,
    )

    # 5. 两阶段训练
    print("\n开始第一阶段训练（激活所有专家，无门控）...")
    model.set_training_phase(1)
    trainer.train()
    print("第一阶段训练完成。")

    print("\n开始第二阶段训练（激活门控机制）...")
    model.set_training_phase(2)
    trainer.train()
    print("第二阶段训练完成。")

    # 6. 评估模型
    print("\n开始评估模型...")
    results = trainer.evaluate()
    print("测试结果：")
    print(results)

    # 7. 保存最终模型
    final_model_save_path = "./final_moe_model"
    trainer.save_model(final_model_save_path)
    print(f"\n最佳模型已保存到 {final_model_save_path}") 