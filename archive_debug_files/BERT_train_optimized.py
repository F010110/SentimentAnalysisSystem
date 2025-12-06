"""
BERT 大数据集训练 - 基于过拟合测试的优化版本

过拟合测试结果：
✓ 小数据集（100条）43轮达到97%准确率
✓ 学习率 1e-5 + 冻结前6层 + 梯度裁剪 = 成功

大数据集优化策略：
1. 学习率：1e-5（已验证有效）
2. 冻结策略：half（只训练后6层，减少过拟合风险）
3. 梯度裁剪：1.0（防止梯度爆炸）
4. 批次大小：16（保持不变）
5. 训练轮数：30（预计15-20轮收敛）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

sys.path.append(os.path.dirname(__file__))
from BERT_model_final import BertSentimentClassifier, train_epoch, evaluate

# ==================== 配置参数 ====================
MODEL_NAME = 'models/bert-base-uncased'
TRAIN_DATA_PATH = 'dataset/twitter_training_cleaned.csv'
VAL_DATA_PATH = 'dataset/twitter_validation_cleaned.csv'

# 训练超参数 - 基于过拟合测试优化
BATCH_SIZE = 16
LEARNING_RATE = 3e-6          # ✓ 过拟合测试验证有效
NUM_EPOCHS = 30
WARMUP_STEPS = 1000
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0           # ✓ 梯度裁剪
MAX_LENGTH = 128

# 模型参数
FREEZE_STRATEGY = 'embed'     
USE_CLASS_WEIGHTS = True
DROPOUT = 0.3                 # 大数据集可以用更高的dropout
NUM_CLASSES = 4

# 其他参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = None  # 不固定种子

# ==================== 数据集 ====================
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_column='attitude', max_length=128, text_column='cleaned_text'):
        self.texts = dataframe[text_column].values
        self.labels = dataframe[label_column].values
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

def set_seed(seed):
    if seed is None:
        print("⚠️ 未设置固定种子，每次训练将产生不同结果")
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"✓ 随机种子已设置为: {seed}")

def create_dataloader(df, tokenizer, batch_size=32, shuffle=True, max_length=128, text_column='cleaned_text'):
    dataset = SentimentDataset(df, tokenizer, max_length=max_length, text_column=text_column)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def print_config():
    print("\n" + "="*70)
    print("BERT 情感分类 - 优化版本 (基于过拟合测试)")
    print("="*70)
    print(f"\n数据配置:")
    print(f"  训练数据: {TRAIN_DATA_PATH}")
    print(f"  验证数据: {VAL_DATA_PATH}")
    print(f"  文本列: cleaned_text")
    print(f"\n模型配置:")
    print(f"  模型: {MODEL_NAME}")
    print(f"  冻结策略: {FREEZE_STRATEGY} ★ 只训练后6层")
    print(f"  Dropout: {DROPOUT}")
    print(f"\n训练配置:")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE} ★ 过拟合测试验证有效")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  预热步数: {WARMUP_STEPS}")
    print(f"  梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  梯度裁剪: {MAX_GRAD_NORM} ★ 防止梯度爆炸")
    print(f"  类别权重: {USE_CLASS_WEIGHTS}")
    print(f"  设备: {DEVICE}")
    print(f"\n优化说明:")
    print(f"  ✓ 基于100条数据过拟合测试（43轮达97%准确率）")
    print(f"  ✓ 使用相同的学习率和冻结策略")
    print(f"  ✓ 添加梯度裁剪保证训练稳定")
    print("="*70)

# ==================== 主训练函数 ====================
def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 打印配置
    print_config()
    
    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    
    # 标签映射
    label_map = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}
    
    if 'attitude_encoded' not in train_df.columns:
        print(f"  转换标签: 文本 → 整数")
        train_df['attitude_encoded'] = train_df['attitude'].map(label_map)
        val_df['attitude_encoded'] = val_df['attitude'].map(label_map)
        train_df['attitude'] = train_df['attitude_encoded']
        val_df['attitude'] = val_df['attitude_encoded']
    
    print(f"  训练集标签分布: {train_df['attitude'].value_counts().sort_index().to_dict()}")
    
    # 2. 计算类别权重
    if USE_CLASS_WEIGHTS:
        print(f"\n[2/7] 计算类别权重...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1, 2, 3]),
            y=train_df['attitude'].values
        )
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        print(f"  类别权重: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
    
    # 3. 加载 tokenizer
    print(f"\n[3/7] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"  ✓ 加载完成")
    
    # 4. 创建数据加载器
    print(f"\n[4/7] 创建数据加载器...")
    train_loader = create_dataloader(
        train_df, tokenizer, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        max_length=MAX_LENGTH,
        text_column='cleaned_text'
    )
    val_loader = create_dataloader(
        val_df, tokenizer, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        max_length=MAX_LENGTH,
        text_column='cleaned_text'
    )
    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")
    
    # 5. 创建模型
    print(f"\n[5/7] 创建模型...")
    model = BertSentimentClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_strategy=FREEZE_STRATEGY
    ).to(DEVICE)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # 6. 设置优化器和学习率调度
    print(f"\n[6/7] 设置优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"  优化器: AdamW (lr={LEARNING_RATE})")
    print(f"  总训练步数: {total_steps}")
    
    # 7. 训练
    print(f"\n[7/7] 开始训练...")
    print("="*70)
    
    best_val_acc = 0
    patience_counter = 0
    patience = 5
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            epoch=epoch+1
        )
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"★ 新的最佳验证准确率: {best_val_acc:.2f}%")
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_model_optimized.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发 (耐心已用尽: {patience})")
                break
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"\n最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型已保存至: best_model_optimized.pth")
    
    if best_val_acc >= 70:
        print("\n✅ 成功！模型达到预期性能")
    elif best_val_acc >= 60:
        print("\n⚠️ 部分成功，建议:")
        print("   - 增加训练轮数")
        print("   - 调整学习率")
    else:
        print("\n❌ 性能不佳，可能需要:")
        print("   - 检查数据质量")
        print("   - 尝试完全解冻BERT")
        print("   - 调整学习率")

if __name__ == '__main__':
    main()
