"""
检查数据和标签的根本性问题
运行: python check_data_labels.py
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*70)
print("数据和标签深度检查")
print("="*70)

# 1. 加载数据
print("\n【1. 加载数据】")
df_train = pd.read_csv('./dataset/twitter_training_cleaned.csv')
df_val = pd.read_csv('./dataset/twitter_validation_cleaned.csv')

print(f"训练集大小: {len(df_train):,}")
print(f"验证集大小: {len(df_val):,}")
print(f"训练集列名: {df_train.columns.tolist()}")

# 2. 检查标签分布
print("\n【2. 标签分布检查】")
print("\n训练集标签分布:")
train_label_dist = df_train['attitude'].value_counts()
print(train_label_dist)
print(f"\n标签比例:")
for label, count in train_label_dist.items():
    print(f"  {label}: {count:>6} ({100*count/len(df_train):>5.2f}%)")

print("\n验证集标签分布:")
val_label_dist = df_val['attitude'].value_counts()
print(val_label_dist)
print(f"\n标签比例:")
for label, count in val_label_dist.items():
    print(f"  {label}: {count:>6} ({100*count/len(df_val):>5.2f}%)")

# 3. 检查标签一致性
print("\n【3. 标签一致性检查】")
train_labels = set(df_train['attitude'].unique())
val_labels = set(df_val['attitude'].unique())
print(f"训练集唯一标签: {sorted(train_labels)}")
print(f"验证集唯一标签: {sorted(val_labels)}")

if train_labels != val_labels:
    print("⚠️  警告: 训练集和验证集的标签不一致!")
    print(f"  只在训练集: {train_labels - val_labels}")
    print(f"  只在验证集: {val_labels - train_labels}")
else:
    print("✓ 标签一致")

# 4. 检查文本质量
print("\n【4. 文本质量检查】")
print(f"\n文本长度统计:")
train_text_lengths = df_train['processed_text'].str.len()
print(f"  平均长度: {train_text_lengths.mean():.1f}")
print(f"  中位数: {train_text_lengths.median():.1f}")
print(f"  最小值: {train_text_lengths.min()}")
print(f"  最大值: {train_text_lengths.max()}")
print(f"  空文本数: {(train_text_lengths == 0).sum()}")

# 5. 随机抽样检查标签是否合理
print("\n【5. 随机样本检查 - 验证标签是否正确】")
print("\n从每个类别随机抽取3个样本:\n")

for label in sorted(df_train['attitude'].unique()):
    samples = df_train[df_train['attitude'] == label].sample(min(3, len(df_train[df_train['attitude'] == label])))
    print(f"标签: {label}")
    print("-" * 70)
    for idx, row in samples.iterrows():
        text = row['processed_text'][:100]  # 只显示前100字符
        print(f"  {text}...")
    print()

# 6. 检查是否有标签混淆
print("\n【6. 潜在标签混淆检查】")
print("\n查找可能被错误标注的样本:")

# 检查带有明显情感词的样本
positive_words = ['love', 'great', 'awesome', 'excellent', 'good', 'best', 'wonderful', 'amazing']
negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'horrible', 'sucks', 'disappointing']

print("\n包含正面词但标注为负面的样本:")
count = 0
for idx, row in df_train.iterrows():
    text = str(row['processed_text']).lower()
    label = row['attitude']
    if any(word in text for word in positive_words) and label in ['Negative', 1]:
        print(f"  [{label}] {text[:80]}...")
        count += 1
        if count >= 5:
            break

print("\n包含负面词但标注为正面的样本:")
count = 0
for idx, row in df_train.iterrows():
    text = str(row['processed_text']).lower()
    label = row['attitude']
    if any(word in text for word in negative_words) and label in ['Positive', 3]:
        print(f"  [{label}] {text[:80]}...")
        count += 1
        if count >= 5:
            break

# 7. 检查类别不平衡程度
print("\n【7. 类别不平衡分析】")
label_counts = df_train['attitude'].value_counts()
max_count = label_counts.max()
min_count = label_counts.min()
imbalance_ratio = max_count / min_count

print(f"最大类别样本数: {max_count}")
print(f"最小类别样本数: {min_count}")
print(f"不平衡比例: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("⚠️  警告: 类别严重不平衡 (>3:1)，建议使用类别权重")
elif imbalance_ratio > 2:
    print("⚠️  注意: 类别不平衡 (>2:1)，可能影响性能")
else:
    print("✓ 类别相对平衡")

# 8. 计算类别权重
print("\n【8. 推荐的类别权重】")
total_samples = len(df_train)
num_classes = len(label_counts)
class_weights = {}

for label, count in label_counts.items():
    weight = total_samples / (num_classes * count)
    class_weights[label] = weight
    print(f"  {label}: {weight:.4f}")

# 9. 检查是否有重复样本
print("\n【9. 重复样本检查】")
duplicates = df_train.duplicated(subset=['processed_text'])
print(f"重复文本数: {duplicates.sum()}")
if duplicates.sum() > 0:
    print(f"重复率: {100*duplicates.sum()/len(df_train):.2f}%")
    print("⚠️  建议去除重复样本")

# 10. 检查文本预处理质量
print("\n【10. 文本预处理质量检查】")
sample_texts = df_train['processed_text'].head(10)
print("\n前10个样本:")
for i, text in enumerate(sample_texts, 1):
    print(f"{i:2d}. {text[:80]}...")

# 检查是否有异常字符
has_special_chars = df_train['processed_text'].str.contains(r'[^\w\s\.\,\!\?\-\']', regex=True)
print(f"\n包含特殊字符的样本数: {has_special_chars.sum()}")

# 检查是否全是小写
is_lowercase = df_train['processed_text'].str.islower()
print(f"全小写样本数: {is_lowercase.sum()} / {len(df_train)} ({100*is_lowercase.sum()/len(df_train):.1f}%)")

print("\n" + "="*70)
print("检查完成")
print("="*70)
