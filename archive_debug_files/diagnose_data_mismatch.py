"""
诊断脚本：对比 SVM (96%) 和 BERT (30%) 使用的数据
目标：找出为什么同样的数据集，SVM能达到96%，BERT只有30%
"""

import pandas as pd
import numpy as np

print("="*70)
print("数据诊断 - SVM vs BERT")
print("="*70)

# 1. 加载数据
train_df = pd.read_csv('dataset/twitter_training_cleaned.csv')
val_df = pd.read_csv('dataset/twitter_validation_cleaned.csv')

print(f"\n训练集: {len(train_df)} 条")
print(f"验证集: {len(val_df)} 条")

# 2. 检查标签分布
print("\n" + "="*70)
print("标签分布检查")
print("="*70)

print("\n训练集标签 (attitude 列):")
print(train_df['attitude'].value_counts().sort_index())
print(f"\n标签类型: {train_df['attitude'].dtype}")
print(f"唯一值: {train_df['attitude'].unique()}")

# 3. 检查是否有 attitude_encoded 列
if 'attitude_encoded' in train_df.columns:
    print("\n⚠️ 发现 attitude_encoded 列!")
    print(train_df['attitude_encoded'].value_counts().sort_index())
    print(f"attitude_encoded 类型: {train_df['attitude_encoded'].dtype}")

# 4. 创建标签映射（和 BERT 代码一致）
label_map = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}

print("\n" + "="*70)
print("标签映射测试")
print("="*70)

# 尝试映射
if train_df['attitude'].dtype == 'object':
    print("\n标签是文本类型，进行映射...")
    mapped_labels = train_df['attitude'].map(label_map)
    print(f"\n映射后的标签:")
    print(mapped_labels.value_counts().sort_index())
    
    # 检查是否有映射失败的
    null_count = mapped_labels.isnull().sum()
    if null_count > 0:
        print(f"\n❌ 映射失败: {null_count} 条数据无法映射!")
        print("无法映射的标签:")
        print(train_df[mapped_labels.isnull()]['attitude'].unique())
else:
    print("\n标签已经是数字类型")
    print(f"数值范围: {train_df['attitude'].min()} ~ {train_df['attitude'].max()}")

# 5. 检查文本列
print("\n" + "="*70)
print("文本列检查")
print("="*70)

print(f"\n可用的列: {train_df.columns.tolist()}")

if 'cleaned_text' in train_df.columns:
    print("\n✓ 找到 cleaned_text 列")
    print(f"样本 1: {train_df['cleaned_text'].iloc[0]}")
    print(f"样本 2: {train_df['cleaned_text'].iloc[1]}")
    print(f"样本 3: {train_df['cleaned_text'].iloc[2]}")
else:
    print("\n❌ 没有找到 cleaned_text 列!")

if 'processed_text' in train_df.columns:
    print("\n✓ 找到 processed_text 列")
    print(f"样本 1: {train_df['processed_text'].iloc[0]}")
    print(f"样本 2: {train_df['processed_text'].iloc[1]}")

# 6. SVM 数据对比
print("\n" + "="*70)
print("SVM vs BERT 数据对比")
print("="*70)

print("\n假设 SVM 使用的是:")
print("- 文本: processed_text (去停用词)")
print("- 标签: 直接的 attitude 列")

print("\nBERT 使用的是:")
print("- 文本: cleaned_text (完整句子)")
print("- 标签: attitude 映射后的整数")

# 7. 关键检查：标签是否正确
print("\n" + "="*70)
print("关键诊断：标签是否一致")
print("="*70)

# 随机抽取10条数据，打印文本和标签
print("\n随机样本检查 (前10条):")
for i in range(min(10, len(train_df))):
    text = train_df['cleaned_text'].iloc[i] if 'cleaned_text' in train_df.columns else train_df['processed_text'].iloc[i]
    label = train_df['attitude'].iloc[i]
    print(f"\n样本 {i+1}:")
    print(f"  文本: {text[:80]}...")
    print(f"  标签: {label}")

# 8. 检查验证集
print("\n" + "="*70)
print("验证集检查")
print("="*70)

print(f"\n验证集标签分布:")
print(val_df['attitude'].value_counts().sort_index())

print(f"\n验证集标签类型: {val_df['attitude'].dtype}")

# 9. 最终诊断
print("\n" + "="*70)
print("诊断结果")
print("="*70)

issues = []

# 检查1: 标签类型
if train_df['attitude'].dtype == 'object':
    issues.append("标签是文本类型，需要映射为整数")
elif train_df['attitude'].max() > 3:
    issues.append(f"标签范围错误: 0-{train_df['attitude'].max()}，应该是 0-3")

# 检查2: 文本列
if 'cleaned_text' not in train_df.columns and 'processed_text' not in train_df.columns:
    issues.append("缺少文本列 (cleaned_text 或 processed_text)")

# 检查3: 数据不平衡
label_counts = train_df['attitude'].value_counts()
max_count = label_counts.max()
min_count = label_counts.min()
if max_count / min_count > 3:
    issues.append(f"数据严重不平衡: 最多{max_count}条，最少{min_count}条")

if issues:
    print("\n发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ 数据格式看起来正常")
    print("\n可能的原因:")
    print("  1. 模型训练参数不当（学习率/冻结策略）")
    print("  2. 数据预处理方式不适合BERT")
    print("  3. 标签质量问题（人工标注错误）")

print("\n" + "="*70)
print("建议操作")
print("="*70)

print("\n1. 运行过拟合测试验证模型能力:")
print("   python test_overfitting_v2.py")

print("\n2. 如果过拟合测试通过，问题在训练策略，尝试:")
print("   - 降低学习率到 5e-6")
print("   - 使用更激进的解冻（只冻结前3层）")
print("   - 延长训练时间（50轮）")

print("\n3. 如果过拟合测试失败，问题在数据，需要:")
print("   - 检查数据标注质量")
print("   - 重新清洗数据")

print("="*70)
