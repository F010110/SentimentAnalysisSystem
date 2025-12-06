"""
过拟合测试脚本
目的：验证模型是否具备学习能力
方法：使用极小数据集（100条），训练多轮，观察是否能过拟合（训练准确率接近100%）

如果模型能过拟合小数据集 → 模型架构OK，问题在于大数据集训练策略
如果模型无法过拟合 → 模型架构/实现有问题
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# 导入自定义模块
sys.path.append(os.path.dirname(__file__))
from BERT_model_final import BertSentimentClassifier

# ==================== 配置参数 ====================
MODEL_NAME = 'models/bert-base-uncased'  # 使用本地模型路径
TRAIN_DATA_PATH = 'dataset/twitter_training_cleaned.csv'

# 小数据集配置
TINY_DATASET_SIZE = 100  # 只用100条数据
BATCH_SIZE = 8           # 小批次
LEARNING_RATE = 5e-5     # 较高学习率（为了快速过拟合）
NUM_EPOCHS = 50          # 多轮训练
MAX_LENGTH = 128

# 其他参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
DROPOUT = 0.1  # 降低 dropout（减少正则化，更容易过拟合）

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
def train_epoch(model, dataloader, optimizer, criterion, device):
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
    print("BERT 过拟合测试 - 验证模型学习能力")
    print("="*70)
    print(f"\n测试配置:")
    print(f"  数据集大小: {TINY_DATASET_SIZE} 条")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  设备: {DEVICE}")
    print(f"\n预期结果:")
    print(f"  ✓ 如果模型能学习：训练准确率应该在 10-20 轮内达到 90%+")
    print(f"  ✗ 如果模型无法学习：准确率一直徘徊在 25-30% (随机水平)")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"  原始数据: {len(df)} 条")
    except FileNotFoundError:
        print(f"  ❌ 文件未找到: {TRAIN_DATA_PATH}")
        return
    
    # 标签映射
    label_map = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}
    if 'attitude_encoded' not in df.columns:
        df['attitude_encoded'] = df['attitude'].map(label_map)
    
    # 从每个类别均匀采样，确保类别平衡
    print(f"\n  从每个类别采样 {TINY_DATASET_SIZE // 4} 条...")
    tiny_df_list = []
    for label in range(4):
        label_df = df[df['attitude_encoded'] == label]
        sampled = label_df.sample(n=min(TINY_DATASET_SIZE // 4, len(label_df)), random_state=42)
        tiny_df_list.append(sampled)
    
    tiny_df = pd.concat(tiny_df_list).reset_index(drop=True)
    print(f"  采样后数据: {len(tiny_df)} 条")
    print(f"  类别分布: {tiny_df['attitude_encoded'].value_counts().sort_index().to_dict()}")
    
    # 2. 加载 tokenizer
    print("\n[2/6] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"  ✓ 从本地加载: {MODEL_NAME}")
    
    # 3. 创建数据集（训练集=测试集，测试过拟合能力）
    print("\n[3/6] 创建数据集...")
    dataset = SentimentDataset(tiny_df, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次数: {len(dataloader)}")
    
    # 4. 创建模型
    print("\n[4/6] 创建模型...")
    model = BertSentimentClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_strategy='none'  # 完全解冻，确保能学习
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # 5. 设置优化器和损失函数
    print("\n[5/6] 设置优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 6. 训练
    print("\n[6/6] 开始训练...")
    print("="*70)
    
    best_accuracy = 0
    overfitting_achieved = False
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, dataloader, optimizer, criterion, DEVICE)
        
        # 每个 epoch 都评估同一个数据集（测试过拟合）
        eval_loss, eval_acc = evaluate(model, dataloader, criterion, DEVICE)
        
        # 记录最佳准确率
        if train_acc > best_accuracy:
            best_accuracy = train_acc
        
        # 检查是否过拟合（训练准确率 > 90%）
        if train_acc >= 90 and not overfitting_achieved:
            overfitting_achieved = True
            print(f"\n{'='*70}")
            print(f"🎉 过拟合达成！Epoch {epoch+1}: 训练准确率 {train_acc:.2f}%")
            print(f"{'='*70}\n")
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch < 5 or overfitting_achieved:
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:6.2f}% | "
                  f"Best: {best_accuracy:6.2f}%")
        
        # 如果已经达到 95%+ 准确率，可以提前停止
        if train_acc >= 95:
            print(f"\n训练准确率达到 {train_acc:.2f}%，提前停止训练。")
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
        print("   → 大数据集训练失败可能是:")
        print("      - 学习率不合适")
        print("      - 数据质量问题")
        print("      - 需要更多训练轮数")
        print("      - 正则化过强（dropout/weight decay）")
    elif best_accuracy >= 60:
        print("\n⚠️ 部分通过：模型有一定学习能力，但不够强")
        print("   → 可能问题:")
        print("      - 学习率偏低")
        print("      - 需要更多训练轮数")
        print("      - Dropout 过高")
        print("      - 模型部分参数被冻结")
    else:
        print("\n❌ 测试失败！模型无法学习")
        print("   → 可能问题:")
        print("      - 模型架构有 bug")
        print("      - 前向传播实现错误")
        print("      - 梯度无法反向传播")
        print("      - 数据预处理有问题")
        print("      - 标签编码错误")
    
    print("="*70)
    
    # 8. 详细诊断（如果失败）
    if best_accuracy < 60:
        print("\n" + "="*70)
        print("详细诊断")
        print("="*70)
        
        # 检查模型输出
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader))
            input_ids = sample_batch['input_ids'].to(DEVICE)
            attention_mask = sample_batch['attention_mask'].to(DEVICE)
            labels = sample_batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask)
            
            print(f"\n样本批次分析:")
            print(f"  输入形状: {input_ids.shape}")
            print(f"  输出形状: {outputs.shape}")
            print(f"  输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"  输出均值: {outputs.mean().item():.4f}")
            print(f"  输出标准差: {outputs.std().item():.4f}")
            
            # 检查预测分布
            _, predicted = torch.max(outputs, 1)
            print(f"\n  真实标签: {labels.cpu().numpy()}")
            print(f"  预测标签: {predicted.cpu().numpy()}")
            print(f"  预测分布: {np.bincount(predicted.cpu().numpy(), minlength=4)}")
        
        # 检查梯度
        print(f"\n梯度检查:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.6f}")
                if grad_norm < 1e-7:
                    print(f"    ⚠️ 梯度过小，可能无法更新")
            else:
                print(f"  {name}: 无梯度 (可能被冻结)")
        
        print("="*70)

if __name__ == '__main__':
    main()
