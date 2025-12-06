"""
BERT 情感分类模型 - 激进修复版本
- 完全解冻 BERT (num_freeze_layers=0)
- 增强分类头
- 添加详细的调试日志
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from BERT_config import get_model_path, get_model_config, print_config

# GPU 优化设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU 已启用: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("使用 CPU 训练")


class BertSentimentClassifier(nn.Module):
    """BERT 情感分类模型 - 激进修复版本"""
    def __init__(self, model_name=None, num_classes=4, dropout=0.3, freeze_strategy='none'):
        """
        Args:
            model_name: HuggingFace 预训练模型名称或本地路径
            num_classes: 分类数量
            dropout: Dropout 概率
            freeze_strategy: 'none' (完全解冻), 'embed' (只冻结嵌入), 'half' (冻结前6层), 'full' (冻结9层)
        """
        super(BertSentimentClassifier, self).__init__()
        
        if model_name is None:
            model_name = get_model_path()
        
        # 加载预训练 BERT 模型和配置
        print(f"加载模型: {model_name}")
        
        # 判断是否为本地路径
        from pathlib import Path
        model_path = Path(model_name)
        
        if model_path.exists() and model_path.is_dir():
            # 本地模型路径
            print(f"从本地加载 BERT 模型...")
            self.bert = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=False,
                local_files_only=True
            )
        else:
            # HuggingFace 在线模型
            print(f"从 HuggingFace 下载 BERT 模型...")
            self.bert = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=False
            )
        
        self.config = self.bert.config
        self.freeze_strategy = freeze_strategy
        
        # 应用冻结策略
        self._apply_freeze_strategy(freeze_strategy)
        
        # 增强的分类头 (激进修复版本)
        hidden_size = self.config.hidden_size  # 768
        
        # 版本 1: 增强的多层分类头 (推荐)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # 版本 2: 简单分类头 (备选)
        # self.classifier = nn.Linear(hidden_size, num_classes)
        
        print(f"模型加载完成: {model_name}")
        print(f"BERT 隐藏层维度: {hidden_size}")
        print(f"冻结策略: {freeze_strategy}")
        print(f"分类数量: {num_classes}")
        print(f"分类头参数: {self._count_params(self.classifier):,}")
        
        # 初始化分类头权重
        self._init_classifier()
    
    def _apply_freeze_strategy(self, strategy):
        """应用冻结策略"""
        if strategy == 'none':
            # 完全解冻 - 最激进，推荐首先尝试
            for param in self.bert.parameters():
                param.requires_grad = True
            print("策略: 完全解冻 BERT (所有参数可训练)")
            
        elif strategy == 'embed':
            # 只冻结嵌入层
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for param in self.bert.encoder.parameters():
                param.requires_grad = True
            print("策略: 冻结嵌入层，解冻编码器")
            
        elif strategy == 'half':
            # 冻结前 6 层，解冻后 6 层
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(6):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            for i in range(6, 12):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            print("策略: 冻结前 6 层，解冻后 6 层")
            
        elif strategy == 'full':
            # 冻结 9 层 - 原始策略
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(9):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            for i in range(9, 12):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            print("策略: 冻结前 9 层，解冻后 3 层 (原始策略)")
        
        else:
            raise ValueError(f"未知的冻结策略: {strategy}")
    
    def _count_params(self, module):
        """计算模块的参数数量"""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    def _init_classifier(self):
        """初始化分类头权重"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # BERT 编码
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用 [CLS] token 的输出 (池化表示)
        cls_output = bert_output.pooler_output  # (batch_size, 768)
        
        # 分类头
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        
        return logits
    
    def get_trainable_params_count(self):
        """获取可训练的参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params_count(self):
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None, 
                gradient_accumulation_steps=1, enable_fp16=False, epoch=1):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        scaler: 混合精度缩放器
        gradient_accumulation_steps: 梯度累积步数
        enable_fp16: 是否启用混合精度
        epoch: 当前 epoch 编号
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 支持字典和元组格式的batch
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
        else:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
        
        # 混合精度前向传播
        if enable_fp16:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / gradient_accumulation_steps
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / gradient_accumulation_steps
        
        # 反向传播
        if enable_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if enable_fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # 计算准确率
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            batch_acc = (preds == labels).sum().item() / labels.size(0)
            print(f"  Epoch {epoch} Batch {batch_idx + 1}: Loss {loss.item() * gradient_accumulation_steps:.4f}, "
                  f"Batch Acc {batch_acc:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device, enable_fp16=False):
    """
    验证
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        enable_fp16: 是否启用混合精度
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
        preds: 预测标签
        labels: 真实标签
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # 支持字典和元组格式的batch
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
            
            if enable_fp16:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy, all_preds, all_labels


def predict_sample(model, tokenizer, texts, device, label_map):
    """
    预测单个样本
    
    Args:
        model: 模型
        tokenizer: 分词器
        texts: 文本列表
        device: 设备
        label_map: 标签映射 {id: label_name}
    
    Returns:
        predictions: 预测标签列表
        confidences: 置信度列表
    """
    model.eval()
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for text in texts:
            # 编码
            encoding = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            
            # 获取预测
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            
            pred_label = label_map.get(pred_id, f"Unknown_{pred_id}")
            predictions.append(pred_label)
            confidences.append(confidence)
    
    return predictions, confidences


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试 BERT 情感分类模型")
    print("=" * 60)
    
    # 创建模型实例 - 使用不同的冻结策略测试
    print("\n--- 策略 1: 完全解冻 (none) ---")
    model_none = BertSentimentClassifier(freeze_strategy='none')
    print(f"总参数: {model_none.get_total_params_count():,}")
    print(f"可训练参数: {model_none.get_trainable_params_count():,}")
    
    print("\n--- 策略 2: 只冻结嵌入 (embed) ---")
    model_embed = BertSentimentClassifier(freeze_strategy='embed')
    print(f"总参数: {model_embed.get_total_params_count():,}")
    print(f"可训练参数: {model_embed.get_trainable_params_count():,}")
    
    print("\n--- 策略 3: 冻结前 6 层 (half) ---")
    model_half = BertSentimentClassifier(freeze_strategy='half')
    print(f"总参数: {model_half.get_total_params_count():,}")
    print(f"可训练参数: {model_half.get_trainable_params_count():,}")
    
    print("\n" + "=" * 60)
