import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from pathlib import Path
from transformers import get_linear_schedule_with_warmup

from BERT_model import (
    BertSentimentClassifier, BertTokenizer, device,
    train_epoch, evaluate, plot_training_history, plot_confusion_matrix,
    predict_sentiment, save_model_checkpoint, load_model_checkpoint
)
from BERT_config import print_config, get_model_path


class SentimentDataset(Dataset):
    """情感分析数据集"""
    def __init__(self, dataframe, tokenizer_obj, label_column='attitude'):
        """
        Args:
            dataframe: 包含文本和标签的 DataFrame
            tokenizer_obj: BertTokenizer 对象
            label_column: 标签列名
        """
        self.texts = dataframe['processed_text'].tolist()
        self.labels = dataframe[label_column].tolist()
        self.tokenizer = tokenizer_obj
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        input_ids, attention_mask = self.tokenizer.encode_text(text)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_df, val_df, tokenizer_obj, batch_size=32, num_workers=2):
    """创建数据加载器"""
    train_dataset = SentimentDataset(train_df, tokenizer_obj)
    val_dataset = SentimentDataset(val_df, tokenizer_obj)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def prepare_data(train_path, val_path):
    """加载和准备数据"""
    print("=" * 60)
    print("步骤 1: 加载数据")
    print("=" * 60)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"训练集列: {train_df.columns.tolist()}")
    
    # 数据清理和预处理
    for df in [train_df, val_df]:
        if 'processed_text' not in df.columns:
            if 'text' in df.columns:
                df['processed_text'] = df['text'].astype(str).str.lower().str.strip()
            else:
                raise ValueError("未找到 'text' 或 'processed_text' 列")
        else:
            df['processed_text'] = df['processed_text'].astype(str).str.lower().str.strip()
    
    # 标签映射
    unique_labels = pd.concat([train_df['attitude'], val_df['attitude']]).unique()
    print(f"\n检测到 {len(unique_labels)} 个类别:")
    
    if not np.issubdtype(type(unique_labels[0]), np.number):
        label_to_idx = {lab: i for i, lab in enumerate(sorted(unique_labels))}
        idx_to_label = {i: lab for lab, i in label_to_idx.items()}
        train_df['attitude'] = train_df['attitude'].map(label_to_idx)
        val_df['attitude'] = val_df['attitude'].map(label_to_idx)
    else:
        unique_ints = sorted({int(i) for i in unique_labels})
        if unique_ints != list(range(len(unique_ints))):
            remap = {old: new for new, old in enumerate(unique_ints)}
            train_df['attitude'] = train_df['attitude'].map(remap)
            val_df['attitude'] = val_df['attitude'].map(remap)
            idx_to_label = {new: str(old) for old, new in remap.items()}
        else:
            idx_to_label = {int(i): str(i) for i in unique_ints}
    
    print(f"标签映射: {idx_to_label}")
    print(f"训练集标签分布:\n{train_df['attitude'].value_counts().sort_index()}\n")
    
    return train_df, val_df, idx_to_label


def main():
    """主训练流程"""
    # ==================== 配置 ====================
    print("\n" + "=" * 60)
    print("BERT 情感分析模型 - Kaggle GPU 优化版本")
    print("=" * 60 + "\n")
    
    # 打印模型配置
    print_config()
    
    # 获取模型路径
    model_path = get_model_path()
    
    # 数据路径
    TRAIN_PATH = os.path.join('dataset', 'twitter_training_cleaned.csv')
    VAL_PATH = os.path.join('dataset', 'twitter_validation_cleaned.csv')
    
    # 模型配置 - RTX 4090 优化版本 (24GB VRAM)
    MODEL_NAME = 'bert-base-uncased'  # 可选: 'distilbert-base-uncased', 'roberta-base' 等
    MAX_LENGTH = 128
    NUM_CLASSES = 4
    BATCH_SIZE = 128  # RTX 4090: 大 Batch Size 充分利用显存
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_STEPS = 300  # 根据数据集规模调整
    GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积: 有效 Batch Size = 256
    ENABLE_FP16 = True  # 启用混合精度 (fp16) 加速
    NUM_WORKERS = 4  # 多进程数据加载
    
    # ==================== 准备数据 ====================
    try:
        train_df, val_df, idx_to_label = prepare_data(TRAIN_PATH, VAL_PATH)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请确保数据文件存在:")
        print(f"  {TRAIN_PATH}")
        print(f"  {VAL_PATH}")
        return
    
    # ==================== 初始化分词器 ====================
    print("=" * 60)
    print("步骤 2: 初始化分词器")
    print("=" * 60 + "\n")
    
    tokenizer = BertTokenizer(model_name=model_path, max_length=MAX_LENGTH)
    
    # ==================== 创建数据加载器 ====================
    print("=" * 60)
    print("步骤 3: 创建数据加载器")
    print("=" * 60 + "\n")
    
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, tokenizer,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}\n")
    
    # ==================== 初始化模型 ====================
    print("=" * 60)
    print("步骤 4: 初始化模型")
    print("=" * 60 + "\n")
    
    model = BertSentimentClassifier(
        model_name=model_path,
        num_classes=NUM_CLASSES,
        dropout=0.3
    )
    model.to(device)
    
    # 计算参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}\n")
    
    # ==================== 定义优化器和损失函数 ====================
    print("=" * 60)
    print("步骤 5: 定义优化器和损失函数")
    print("=" * 60 + "\n")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        eps=1e-8
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    print(f"优化器: AdamW")
    print(f"学习率: {LEARNING_RATE}")
    print(f"权重衰减: 0.01")
    print(f"总训练步数: {total_steps}")
    print(f"预热步数: {WARMUP_STEPS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"有效 Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"混合精度 (fp16): {ENABLE_FP16}")
    print(f"数据加载工作进程: {NUM_WORKERS}\n")
    
    # ==================== 训练循环 ====================
    print("=" * 60)
    print("步骤 6: 开始训练")
    print("=" * 60 + "\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            epoch, NUM_EPOCHS,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            enable_fp16=ENABLE_FP16
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, enable_fp16=ENABLE_FP16)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f"\n训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            save_model_checkpoint(model, tokenizer, optimizer, epoch,
                                save_path='bert_model_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: 验证损失连续 {patience} 个 epoch 未改进")
                break
    
    # ==================== 绘制训练历史 ====================
    print("\n" + "=" * 60)
    print("步骤 7: 绘制训练历史")
    print("=" * 60 + "\n")
    
    plot_training_history(history)
    
    # ==================== 最终评估 ====================
    print("\n" + "=" * 60)
    print("步骤 8: 最终评估")
    print("=" * 60 + "\n")
    
    model.eval()
    val_loss, val_acc, all_predictions, all_labels = evaluate(model, val_loader, criterion, enable_fp16=ENABLE_FP16)
    
    print(f"验证集最终准确率: {val_acc:.4f}")
    print(f"验证集最终损失: {val_loss:.4f}\n")
    
    # 详细分类报告
    from sklearn.metrics import classification_report
    target_names = [idx_to_label.get(i, str(i)) for i in range(NUM_CLASSES)]
    print("详细分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    # 混淆矩阵
    plot_confusion_matrix(all_labels, all_predictions, label_names=target_names)
    
    # ==================== 预测示例 ====================
    print("\n" + "=" * 60)
    print("步骤 9: 预测示例")
    print("=" * 60 + "\n")
    
    test_texts = [
        "This product is absolutely wonderful and amazing!",
        "I hate this thing it's terrible",
        "It's okay, not great but not bad either",
        "The quality is good but the price is too high",
        "Worst purchase ever, complete waste of money"
    ]
    
    # 反向标签映射 (数字 -> 文字)
    idx_to_sentiment = {v: k for k, v in enumerate(sorted(set(idx_to_label.values())))}
    sentiment_to_text = {i: name for name, i in idx_to_sentiment.items()}
    
    for text in test_texts:
        sentiment, confidence, probs = predict_sentiment(
            model, tokenizer, text, sentiment_to_text
        )
        print(f"文本: '{text}'")
        print(f"预测情感: {sentiment} (置信度: {confidence:.2%})")
        print(f"所有类别概率: {probs}")
        print("-" * 60)
    
    # ==================== 保存模型 ====================
    print("\n" + "=" * 60)
    print("步骤 10: 保存模型")
    print("=" * 60 + "\n")
    
    save_model_checkpoint(model, tokenizer, optimizer, NUM_EPOCHS - 1,
                        save_path='bert_model_final.pt')
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
