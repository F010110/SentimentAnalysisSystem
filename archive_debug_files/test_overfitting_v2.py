"""
过拟合测试脚本 v2 - 修复版本
修复问题：
1. 降低学习率（5e-5 → 1e-5）防止梯度爆炸
2. 使用梯度裁剪
3. 只解冻最后2层BERT + 分类头（减少训练难度）
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

# ==================== 配置参数 ====================
MODEL_NAME = 'models/bert-base-uncased'
TRAIN_DATA_PATH = 'dataset/twitter_training_cleaned.csv'

# 小数据集配置 - 修复版
TINY_DATASET_SIZE = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-5      # ✓ 降低学习率（原来5e-5太高）
NUM_EPOCHS = 100          # ✓ 增加轮数
MAX_LENGTH = 128
MAX_GRAD_NORM = 1.0       # ✓ 添加梯度裁剪

# 其他参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
DROPOUT = 0.1
FREEZE_STRATEGY = 'half'  # ✓ 只训练最后几层（更容易收敛）

# ==================== 数据集 ====================
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

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估"""
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
    print("BERT 过拟合测试 v2 - 修复版本")
    print("="*70)
    print(f"\n修复内容:")
    print(f"  ✓ 学习率: 5e-5 → {LEARNING_RATE}")
    print(f"  ✓ 冻结策略: none → {FREEZE_STRATEGY} (只训练后半部分)")
    print(f"  ✓ 添加梯度裁剪: max_norm={MAX_GRAD_NORM}")
    print(f"  ✓ 增加训练轮数: 50 → {NUM_EPOCHS}")
    print(f"\n测试配置:")
    print(f"  数据集大小: {TINY_DATASET_SIZE} 条")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  设备: {DEVICE}")
    print(f"\n预期结果:")
    print(f"  ✓ 20-30 轮内达到 90%+ 准确率")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"  原始数据: {len(df)} 条")
    
    # 标签映射
    label_map = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}
    if 'attitude_encoded' not in df.columns:
        df['attitude_encoded'] = df['attitude'].map(label_map)
    
    # 从每个类别采样
    tiny_df_list = []
    for label in range(4):
        label_df = df[df['attitude_encoded'] == label]
        sampled = label_df.sample(n=min(TINY_DATASET_SIZE // 4, len(label_df)), random_state=42)
        tiny_df_list.append(sampled)
    
    tiny_df = pd.concat(tiny_df_list).reset_index(drop=True)
    print(f"  采样数据: {len(tiny_df)} 条")
    
    # 2. 加载 tokenizer
    print("\n[2/6] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"  ✓ 加载完成")
    
    # 3. 创建数据集
    print("\n[3/6] 创建数据集...")
    dataset = SentimentDataset(tiny_df, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  批次数: {len(dataloader)}")
    
    # 4. 创建模型
    print("\n[4/6] 创建模型...")
    model = BertSentimentClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_strategy=FREEZE_STRATEGY
    ).to(DEVICE)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # 5. 设置优化器
    print("\n[5/6] 设置优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 6. 训练
    print("\n[6/6] 开始训练...")
    print("="*70)
    
    best_accuracy = 0
    overfitting_achieved = False
    patience = 0
    max_patience = 20  # 如果20轮没提升就停止
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, dataloader, optimizer, criterion, DEVICE, MAX_GRAD_NORM)
        
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            patience = 0
        else:
            patience += 1
        
        # 检查是否过拟合
        if train_acc >= 90 and not overfitting_achieved:
            overfitting_achieved = True
            print(f"\n{'='*70}")
            print(f"🎉 过拟合达成！Epoch {epoch+1}: 训练准确率 {train_acc:.2f}%")
            print(f"{'='*70}\n")
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch < 10 or overfitting_achieved:
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:6.2f}% | "
                  f"Best: {best_accuracy:6.2f}%")
        
        # 提前停止
        if train_acc >= 95:
            print(f"\n✓ 训练准确率达到 {train_acc:.2f}%，提前停止。")
            break
        
        if patience >= max_patience:
            print(f"\n⚠️ {max_patience} 轮无提升，提前停止。")
            break
    
    # 7. 总结
    print("\n" + "="*70)
    print("测试结果总结")
    print("="*70)
    print(f"\n最佳训练准确率: {best_accuracy:.2f}%")
    
    if best_accuracy >= 90:
        print("\n✅ 测试通过！模型具备学习能力")
        print("   → 模型架构正常")
        print("   → 可以记住训练数据")
        print("   → 大数据集问题可能是:")
        print("      - 学习率需要调整")
        print("      - 需要更多训练轮数")
        print("      - 数据质量/标签问题")
    elif best_accuracy >= 70:
        print("\n⚠️ 部分通过：模型有学习能力但较弱")
        print("   → 建议:")
        print("      - 进一步降低学习率")
        print("      - 延长训练时间")
        print("      - 使用更激进的解冻策略")
    else:
        print("\n❌ 测试失败！模型学习能力不足")
        print(f"   → 当前准确率 {best_accuracy:.2f}% 接近随机水平 (25%)")
        print("   → 可能问题:")
        print("      - 学习率仍然过高或过低")
        print("      - 梯度消失/爆炸")
        print("      - 模型架构问题")
    
    print("="*70)

if __name__ == '__main__':
    main()
