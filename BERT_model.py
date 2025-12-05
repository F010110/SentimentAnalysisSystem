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
    """BERT 情感分类模型"""
    def __init__(self, model_name=None, num_classes=4, dropout=0.3):
        """
        Args:
            model_name: HuggingFace 预训练模型名称或本地路径
            num_classes: 分类数量
            dropout: Dropout 概率
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
        
        # 冻结 BERT 部分层的参数，只微调最后几层
        self._freeze_bert_layers(num_freeze_layers=9)  # 冻结前 9 层，共 12 层
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
        print(f"模型加载完成: {model_name}")
        print(f"BERT 隐藏层维度: {self.config.hidden_size}")
        print(f"分类数量: {num_classes}")
    
    def _freeze_bert_layers(self, num_freeze_layers):
        """冻结 BERT 的前 N 层，减少参数更新"""
        if num_freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            
            for i in range(num_freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            
            print(f"已冻结 BERT 的前 {num_freeze_layers} 层")
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: 标记化的输入 ID
            attention_mask: 注意力掩码
        
        Returns:
            logits: 分类 logits
        """
        # BERT 编码
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用 [CLS] token 的输出 (池化表示)
        cls_output = bert_output.pooler_output  # (batch_size, hidden_size)
        
        # 分类头
        x = self.dropout(cls_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class BertTokenizer:
    """BERT 分词器包装器"""
    def __init__(self, model_name=None, max_length=128):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径
            max_length: 最大序列长度
        """
        if model_name is None:
            model_name = get_model_path()
        
        print(f"加载分词器: {model_name}")
        
        # 判断是否为本地路径（绝对路径或包含 / 或 \）
        from pathlib import Path
        model_path = Path(model_name)
        
        if model_path.exists() and model_path.is_dir():
            # 本地模型路径
            print(f"从本地加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
        else:
            # HuggingFace 在线模型
            print(f"从 HuggingFace 下载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.max_length = max_length
    
    def encode_text(self, text):
        """
        编码文本
        
        Args:
            text: 输入文本
        
        Returns:
            input_ids: 标记 ID
            attention_mask: 注意力掩码
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


def train_epoch(model, train_loader, optimizer, criterion, epoch, num_epochs, 
                gradient_accumulation_steps=1, enable_fp16=True):
    """训练一个 epoch，支持梯度累积和混合精度"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    scaler = GradScaler() if torch.cuda.is_available() and enable_fp16 else None
    
    # 初始化梯度累积
    accumulated_loss = 0
    accumulation_counter = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 混合精度训练 + 梯度累积
        if scaler is not None:
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / gradient_accumulation_steps
            loss.backward()
        
        accumulated_loss += loss.item()
        accumulation_counter += 1
        
        # 梯度累积：当积累到指定步数时进行优化器更新
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            total_loss += accumulated_loss
            accumulated_loss = 0
            accumulation_counter = 0
        
        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} (GA: {gradient_accumulation_steps}x)")
    
    avg_loss = total_loss / (len(train_loader) // gradient_accumulation_steps + 1)
    accuracy = correct / total
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, enable_fp16=True):
    """评估模型（支持混合精度）"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 混合精度评估
            if torch.cuda.is_available() and enable_fp16:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels


def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='训练损失', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='验证损失', color='red', linewidth=2)
    ax1.set_title('模型损失', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(history['train_acc'], label='训练准确率', color='green', linewidth=2)
    ax2.plot(history['val_acc'], label='验证准确率', color='orange', linewidth=2)
    ax2.set_title('模型准确率', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bert_training_history.png', dpi=300, bbox_inches='tight')
    print("训练历史图已保存: bert_training_history.png")
    plt.show()


def plot_confusion_matrix(all_labels, all_predictions, label_names=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('bert_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵已保存: bert_confusion_matrix.png")
    plt.show()


def predict_sentiment(model, tokenizer, text, label_to_sentiment):
    """预测单个文本的情感"""
    model.eval()
    
    with torch.no_grad():
        input_ids, attention_mask = tokenizer.encode_text(text)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        
        # 混合精度推理
        if torch.cuda.is_available():
            with autocast():
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)
        
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = predicted.item()
        predicted_sentiment = label_to_sentiment.get(predicted_label, f"Class {predicted_label}")
        confidence_score = confidence.item()
    
    return predicted_sentiment, confidence_score, probabilities.cpu().numpy()[0]


def save_model_checkpoint(model, tokenizer_obj, optimizer, epoch, save_path='bert_model_checkpoint.pt'):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer_obj.tokenizer,
        'model_config': {
            'model_name': model.bert.config.model_type,
            'num_classes': model.fc2.out_features,
            'hidden_size': model.config.hidden_size
        }
    }
    torch.save(checkpoint, save_path)
    print(f"模型检查点已保存: {save_path}")


def load_model_checkpoint(load_path='bert_model_checkpoint.pt'):
    """加载模型检查点"""
    checkpoint = torch.load(load_path, map_location=device)
    
    # 重建模型
    model = BertSentimentClassifier(
        model_name='bert-base-uncased',
        num_classes=checkpoint['model_config']['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"模型检查点已加载: {load_path}")
    return model, checkpoint['tokenizer']
