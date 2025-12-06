"""
BERT 训练 - 极简版本
完全复制过拟合测试的成功配置：
- LR = 1e-5
- freeze = half  
- 梯度裁剪 = 1.0
- dropout = 0.1

唯一区别：数据集从100条扩展到全量
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from BERT_model_final import BertSentimentClassifier

# ==================== 配置参数（和过拟合测试完全一致）====================
MODEL_NAME = 'models/bert-base-uncased'
TRAIN_DATA_PATH = 'dataset/twitter_training_cleaned.csv'
VAL_DATA_PATH = 'dataset/twitter_validation_cleaned.csv'

# 训练参数 - 和过拟合测试v2完全一致
BATCH_SIZE = 8              # 和过拟合测试一致
LEARNING_RATE = 1e-5        # 和过拟合测试一致
NUM_EPOCHS = 50             # 增加轮数
MAX_LENGTH = 128
MAX_GRAD_NORM = 1.0

# 模型参数 - 和过拟合测试一致
FREEZE_STRATEGY = 'half'    # 和过拟合测试一致
DROPOUT = 0.1               # 和过拟合测试一致（不是0.3）
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据集（和过拟合测试一致）====================
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe['cleaned_text'].values
        self.labels = dataframe['attitude_encoded'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== 训练函数（和过拟合测试一致）====================
def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每500个batch打印一次
        if (batch_idx + 1) % 500 == 0:
            batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss {loss.item():.4f}, Acc {batch_acc:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# ==================== 主函数 ====================
def main():
    print("="*70)
    print("BERT 训练 - 极简版本（复制过拟合测试成功配置）")
    print("="*70)
    print(f"\n配置 (和过拟合测试v2一致):")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  冻结策略: {FREEZE_STRATEGY}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  梯度裁剪: {MAX_GRAD_NORM}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  设备: {DEVICE}")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    
    # 标签映射
    label_map = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}
    if 'attitude_encoded' not in train_df.columns:
        train_df['attitude_encoded'] = train_df['attitude'].map(label_map)
        val_df['attitude_encoded'] = val_df['attitude'].map(label_map)
    
    print(f"  标签分布: {train_df['attitude_encoded'].value_counts().sort_index().to_dict()}")
    
    # 2. 加载 tokenizer
    print("\n[2/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print("  ✓ 完成")
    
    # 3. 创建数据集
    print("\n[3/5] 创建数据集...")
    train_dataset = SentimentDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = SentimentDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    # 4. 创建模型
    print("\n[4/5] 创建模型...")
    model = BertSentimentClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_strategy=FREEZE_STRATEGY
    ).to(DEVICE)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}% 可训练)")
    
    # 5. 训练
    print("\n[5/5] 开始训练...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("="*70)
    
    best_val_acc = 0
    patience = 0
    max_patience = 10
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, MAX_GRAD_NORM)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            print(f"★ 新的最佳准确率: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), 'best_model_simple.pth')
        else:
            patience += 1
            print(f"  ({patience}/{max_patience} 轮无提升)")
            if patience >= max_patience:
                print(f"\n早停触发")
                break
    
    print("\n" + "="*70)
    print("训练完成")
    print("="*70)
    print(f"\n最佳验证准确率: {best_val_acc:.2f}%")
    
    if best_val_acc >= 90:
        print("\n✅ 优秀！接近 SVM/GRU 水平")
    elif best_val_acc >= 70:
        print("\n✓ 及格，但还有提升空间")
    elif best_val_acc >= 50:
        print("\n⚠️ 偏低，建议:")
        print("   - 延长训练时间（100轮）")
        print("   - 降低学习率（5e-6）")
    else:
        print("\n❌ 失败，可能原因:")
        print("   - 数据标签有问题")
        print("   - 需要完全解冻BERT")

if __name__ == '__main__':
    main()
