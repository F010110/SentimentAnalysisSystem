"""
检查训练过程中的问题
运行: python check_training_process.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path

from BERT_model_final import BertSentimentClassifier
from BERT_main_aggressive import SentimentDataset
from BERT_config import get_model_path

print("="*70)
print("训练过程深度检查")
print("="*70)

MODEL_PATH = './models/bert-base-uncased'
TRAIN_DATA_PATH = './dataset/twitter_training_cleaned.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载数据
print("\n【1. 加载数据】")
df_train = pd.read_csv(TRAIN_DATA_PATH)
print(f"数据大小: {len(df_train)}")

# 建立标签映射
unique_labels = sorted(df_train['attitude'].unique())
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
df_train['attitude'] = df_train['attitude'].map(label_to_id)

print(f"标签映射: {label_to_id}")

# 2. 加载分词器和模型
print("\n【2. 加载模型】")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertSentimentClassifier(MODEL_PATH, freeze_strategy='none')
model.to(DEVICE)
model.eval()

print(f"总参数: {model.get_total_params_count():,}")
print(f"可训练参数: {model.get_trainable_params_count():,}")

# 3. 创建小批次数据
print("\n【3. 创建测试批次】")
small_df = df_train.head(32)  # 32个样本
dataset = SentimentDataset(small_df, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# 4. 检查模型输出
print("\n【4. 检查模型输出分布】")
all_logits = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

all_logits = torch.cat(all_logits, dim=0)
all_probs = torch.cat(all_probs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print(f"\nLogits 统计:")
print(f"  形状: {all_logits.shape}")
print(f"  均值: {all_logits.mean(dim=0).tolist()}")
print(f"  标准差: {all_logits.std(dim=0).tolist()}")
print(f"  最小值: {all_logits.min(dim=0).values.tolist()}")
print(f"  最大值: {all_logits.max(dim=0).values.tolist()}")

print(f"\n概率分布统计:")
for i in range(4):
    print(f"  类别 {i} ({id_to_label[i]}): 平均概率 = {all_probs[:, i].mean():.4f}")

# 5. 检查预测分布
print("\n【5. 检查预测分布】")
predictions = torch.argmax(all_probs, dim=1)
pred_dist = {i: (predictions == i).sum().item() for i in range(4)}

print("预测分布:")
for class_id, count in pred_dist.items():
    print(f"  {id_to_label[class_id]}: {count} / 32 ({100*count/32:.1f}%)")

print("\n真实标签分布:")
label_dist = {i: (all_labels == i).sum().item() for i in range(4)}
for class_id, count in label_dist.items():
    print(f"  {id_to_label[class_id]}: {count} / 32 ({100*count/32:.1f}%)")

# 6. 检查是否所有预测都偏向某一类
print("\n【6. 检查预测偏差】")
if max(pred_dist.values()) > 25:  # 如果超过78%都预测为同一类
    print("⚠️  警告: 模型几乎总是预测同一个类别!")
    dominant_class = max(pred_dist, key=pred_dist.get)
    print(f"   主导类别: {id_to_label[dominant_class]}")
    print("   可能原因:")
    print("   1. 类别不平衡严重")
    print("   2. 损失函数没有使用类别权重")
    print("   3. 模型初始化问题")
    print("   4. 学习率过低")

# 7. 检查损失函数
print("\n【7. 检查损失计算】")
criterion = nn.CrossEntropyLoss()
loss = criterion(all_logits, all_labels)
print(f"当前批次损失: {loss.item():.4f}")

# 计算随机基线
num_classes = 4
random_baseline = -np.log(1/num_classes)
print(f"随机基线损失: {random_baseline:.4f}")

if abs(loss.item() - random_baseline) < 0.1:
    print("⚠️  警告: 损失接近随机基线，模型可能没有学习!")

# 8. 检查单个样本的预测
print("\n【8. 检查单个样本预测】")
for i in range(min(5, len(small_df))):
    text = small_df.iloc[i]['processed_text'][:80]
    true_label = small_df.iloc[i]['attitude']
    pred_label = predictions[i].item()
    probs_i = all_probs[i]
    
    print(f"\n样本 {i+1}:")
    print(f"  文本: {text}...")
    print(f"  真实: {id_to_label[true_label]}")
    print(f"  预测: {id_to_label[pred_label]}")
    print(f"  概率: {[f'{p:.3f}' for p in probs_i.tolist()]}")
    print(f"  正确: {'✓' if true_label == pred_label else '✗'}")

# 9. 测试梯度流
print("\n【9. 测试梯度流】")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 取一个批次
batch = next(iter(dataloader))
input_ids = batch['input_ids'].to(DEVICE)
attention_mask = batch['attention_mask'].to(DEVICE)
labels = batch['labels'].to(DEVICE)

optimizer.zero_grad()
logits = model(input_ids, attention_mask)
loss = criterion(logits, labels)
loss.backward()

print(f"损失: {loss.item():.4f}")

# 检查梯度
grad_norms = {}
for name, param in model.named_parameters():
    if param.grad is not None and param.requires_grad:
        grad_norm = param.grad.norm().item()
        if 'classifier' in name or 'bert.encoder.layer.11' in name or 'bert.encoder.layer.0' in name:
            grad_norms[name] = grad_norm

print("\n关键层梯度范数:")
for name, norm in sorted(grad_norms.items())[:5]:
    print(f"  {name[:50]:50s}: {norm:.6f}")

if all(norm < 1e-6 for norm in grad_norms.values()):
    print("⚠️  警告: 梯度几乎为零，学习可能有问题!")

# 10. 建议
print("\n" + "="*70)
print("【诊断建议】")
print("="*70)

issues = []
if max(pred_dist.values()) > 25:
    issues.append("模型预测偏向单一类别")
if abs(loss.item() - random_baseline) < 0.1:
    issues.append("损失接近随机基线")

if issues:
    print("\n发现的问题:")
    for issue in issues:
        print(f"  ⚠️  {issue}")
    
    print("\n推荐修复:")
    print("  1. 使用类别权重损失函数")
    print("  2. 提高学习率到 5e-5 或 1e-4")
    print("  3. 减小批次大小到 16 或 32")
    print("  4. 增加 warmup 步数")
    print("  5. 尝试不同的优化器设置")
else:
    print("\n✓ 未发现明显问题")

print("\n" + "="*70)
