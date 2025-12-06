"""
对比 processed_text 和 cleaned_text 的效果
快速测试脚本
"""

import torch
import pandas as pd
from transformers import AutoTokenizer
from BERT_model_final import BertSentimentClassifier

MODEL_PATH = './models/bert-base-uncased'
DATA_PATH = './dataset/twitter_training_cleaned.csv'

print("="*70)
print("对比 processed_text vs cleaned_text")
print("="*70)

# 加载数据
df = pd.read_csv(DATA_PATH)
print(f"\n数据集大小: {len(df)}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# 取5个样本对比
print("\n【样本对比】\n")
for i in range(5):
    print(f"样本 {i+1} - 标签: {df.iloc[i]['attitude']}")
    print("-" * 70)
    
    # processed_text
    processed = df.iloc[i]['processed_text']
    processed_tokens = tokenizer.tokenize(processed)
    print(f"processed_text: {processed}")
    print(f"Token数: {len(processed_tokens)}, Tokens: {processed_tokens[:10]}...")
    
    # cleaned_text
    cleaned = df.iloc[i]['cleaned_text']
    cleaned_tokens = tokenizer.tokenize(cleaned)
    print(f"cleaned_text:   {cleaned}")
    print(f"Token数: {len(cleaned_tokens)}, Tokens: {cleaned_tokens[:10]}...")
    
    print()

# 统计信息
print("\n【统计对比】\n")

processed_lengths = df['processed_text'].str.len()
cleaned_lengths = df['cleaned_text'].str.len()

print("文本长度:")
print(f"  processed_text - 平均: {processed_lengths.mean():.1f}, 中位数: {processed_lengths.median():.1f}")
print(f"  cleaned_text   - 平均: {cleaned_lengths.mean():.1f}, 中位数: {cleaned_lengths.median():.1f}")

# Token 数量对比
sample_size = min(1000, len(df))
print(f"\n使用 {sample_size} 个样本估算 Token 数量:")

processed_token_counts = []
cleaned_token_counts = []

for i in range(sample_size):
    processed_token_counts.append(len(tokenizer.tokenize(df.iloc[i]['processed_text'])))
    cleaned_token_counts.append(len(tokenizer.tokenize(df.iloc[i]['cleaned_text'])))

import numpy as np
print(f"  processed_text - 平均 Token 数: {np.mean(processed_token_counts):.1f}")
print(f"  cleaned_text   - 平均 Token 数: {np.mean(cleaned_token_counts):.1f}")

# 建议
print("\n" + "="*70)
print("【分析结论】")
print("="*70)

print("\nprocessed_text 特点:")
print("  ✓ 文本更短，训练更快")
print("  ✗ 丢失上下文信息")
print("  ✗ 不适合 BERT (BERT 需要完整句子)")
print("  预期准确率: 50-65%")

print("\ncleaned_text 特点:")
print("  ✓ 保留完整句子结构")
print("  ✓ BERT 可以充分利用预训练知识")
print("  ✓ 更符合 BERT 的设计理念")
print("  预期准确率: 70-80%")

print("\n推荐: 使用 cleaned_text")
print("="*70)
