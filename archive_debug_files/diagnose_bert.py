"""
BERT 模型诊断脚本 - 深度诊断梯度流、数据、参数等问题
在云服务器上运行：python diagnose_bert.py
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from pathlib import Path

# 配置
MODEL_PATH = './models/bert-base-uncased'
TRAIN_PATH = './dataset/twitter_training_cleaned.csv'

def check_gradients(model):
    """检查所有层的梯度情况"""
    print("\n" + "="*60)
    print("梯度流检查")
    print("="*60 + "\n")
    
    total_params = 0
    trainable_params = 0
    frozen_layers = 0
    trainable_layers = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_layers += 1
            status = "✅ 可训练"
            
            # 检查是否有梯度
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_status = f"有梯度 (norm: {grad_norm:.6f})"
            else:
                grad_status = "❌ 无梯度"
        else:
            frozen_layers += 1
            status = "❌ 冻结"
            grad_status = "-"
        
        # 只打印层级（简化输出）
        if 'bert.' in name or name.startswith('fc'):
            print(f"{name:50s} | {status:10s} | {grad_status}")
    
    print("\n" + "-"*60)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"冻结参数数: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
    print(f"可训练层数: {trainable_layers}")
    print(f"冻结层数: {frozen_layers}")
    print("-"*60 + "\n")
    
    if trainable_params / total_params < 0.1:
        print("⚠️  警告: 可训练参数比例太低 (<10%)！")
        print("    这会导致模型学习能力严重不足。\n")
    
    if trainable_params < 1_000_000:
        print("⚠️  警告: 可训练参数太少 (<100万)！")
        print("    模型几乎没有自由度进行学习。\n")


def check_data(df, max_samples=100):
    """检查数据质量"""
    print("\n" + "="*60)
    print("数据质量检查")
    print("="*60 + "\n")
    
    print(f"数据集大小: {len(df):,}")
    print(f"列: {df.columns.tolist()}")
    print(f"\n标签分布:")
    print(df['attitude'].value_counts().sort_index())
    
    # 检查文本
    print(f"\n文本样本检查 (前 {max_samples} 个):")
    print(f"平均长度: {df['processed_text'].str.len().mean():.1f}")
    print(f"最小长度: {df['processed_text'].str.len().min()}")
    print(f"最大长度: {df['processed_text'].str.len().max()}")
    
    print(f"\n文本示例:")
    for i in range(min(3, len(df))):
        text = df['processed_text'].iloc[i]
        label = df['attitude'].iloc[i]
        print(f"  Label {label}: {text[:80]}...")
    
    # 检查缺失值
    print(f"\n缺失值检查:")
    print(df[['processed_text', 'attitude']].isnull().sum())


def test_tokenizer(tokenizer, sample_texts):
    """测试分词器"""
    print("\n" + "="*60)
    print("分词器测试")
    print("="*60 + "\n")
    
    print(f"词汇表大小: {len(tokenizer)}")
    
    for text in sample_texts[:2]:
        print(f"\n原始文本: {text[:80]}...")
        tokens = tokenizer.tokenize(text)
        print(f"Token 数量: {len(tokens)}")
        print(f"Tokens: {tokens[:20]}...")


def test_model_forward(model, tokenizer, sample_texts):
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("模型前向传播测试")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for i, text in enumerate(sample_texts[:2]):
            print(f"\n文本 {i+1}: {text[:80]}...")
            
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
            
            print(f"Input shape: {input_ids.shape}")
            
            # 前向传播
            try:
                logits = model(input_ids, attention_mask)
                print(f"✅ 前向传播成功")
                print(f"Output shape: {logits.shape}")
                print(f"Logits: {logits.squeeze().tolist()}")
                
                probs = torch.softmax(logits, dim=1)
                print(f"概率: {probs.squeeze().tolist()}")
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")


def test_backward(model, tokenizer, sample_text, sample_label):
    """测试反向传播和梯度"""
    print("\n" + "="*60)
    print("反向传播测试")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # 编码
    encoding = tokenizer(
        sample_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    labels = torch.tensor([sample_label]).to(device)
    
    print(f"样本: {sample_text[:80]}...")
    print(f"标签: {sample_label}")
    
    # 清空梯度
    optimizer.zero_grad()
    
    # 前向传播
    logits = model(input_ids, attention_mask)
    print(f"Logits: {logits.squeeze().tolist()}")
    
    # 计算损失
    loss = criterion(logits, labels)
    print(f"初始损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    print(f"✅ 反向传播完成")
    
    # 检查梯度
    print(f"\n梯度检查 (前 5 层):")
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and grad_count < 5:
            grad_norm = param.grad.norm().item()
            print(f"  {name:40s}: grad_norm = {grad_norm:.6f}")
            grad_count += 1
    
    # 优化器步骤
    optimizer.step()
    print(f"\n✅ 优化器步骤完成")
    print(f"学习率: 2e-5")


def main():
    print("\n" + "="*60)
    print("BERT 模型诊断工具")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 1. 加载数据
    print("\n加载数据...")
    try:
        df = pd.read_csv(TRAIN_PATH)
        check_data(df)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 加载分词器
    print("\n加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        sample_texts = df['processed_text'].head(3).tolist()
        test_tokenizer(tokenizer, sample_texts)
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return
    
    # 3. 加载模型
    print("\n加载模型...")
    try:
        bert_model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
        
        # 冻结所有层（模拟当前设置）
        for param in bert_model.parameters():
            param.requires_grad = False
        
        # 分类头
        class_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
        model = nn.Sequential(bert_model, class_head)
        
        check_gradients(model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 4. 测试前向传播
    try:
        sample_texts = df['processed_text'].head(2).tolist()
        test_model_forward(bert_model, tokenizer, sample_texts)
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 测试反向传播
    try:
        sample_text = df['processed_text'].iloc[0]
        sample_label = int(df['attitude'].iloc[0])
        test_backward(bert_model, tokenizer, sample_text, sample_label)
    except Exception as e:
        print(f"❌ 反向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("诊断完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
