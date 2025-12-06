"""
BERT 训练 - 使用分层学习率 (Layer-wise Learning Rate Decay)
更稳定的训练策略
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from BERT_model_final import BertSentimentClassifier, train_epoch, evaluate, predict_sample
from BERT_main_aggressive import SentimentDataset, set_seed, print_config
from BERT_config import get_model_path

# 配置
TRAIN_DATA_PATH = './dataset/twitter_training_cleaned.csv'
VAL_DATA_PATH = './dataset/twitter_validation_cleaned.csv'
MODEL_PATH = './models/bert-base-uncased'
SAVE_PATH = './saved_model_layerwise_lr'

# 超参数 - 分层学习率策略
BATCH_SIZE = 32              # 适中批次
BASE_LEARNING_RATE = 2e-5    # BERT 基础学习率
CLASSIFIER_LEARNING_RATE = 1e-4  # 分类头更高学习率
NUM_EPOCHS = 20
WARMUP_STEPS = 2000
GRADIENT_ACCUMULATION_STEPS = 2
ENABLE_FP16 = True
FREEZE_STRATEGY = 'none'
USE_CLASS_WEIGHTS = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
DROPOUT = 0.3
SEED = 42

def create_layerwise_optimizer(model, base_lr, classifier_lr):
    """
    创建分层优化器：
    - BERT embedding: 0.1 * base_lr
    - BERT lower layers: base_lr * layer_decay
    - BERT upper layers: base_lr  
    - Classifier head: classifier_lr (更高)
    """
    no_decay = ['bias', 'LayerNorm.weight']
    layer_decay = 0.95  # 每层衰减5%
    
    optimizer_grouped_parameters = []
    
    # 分类头 - 最高学习率
    optimizer_grouped_parameters.append({
        'params': [p for n, p in model.named_parameters() 
                   if 'classifier' in n and not any(nd in n for nd in no_decay)],
        'lr': classifier_lr,
        'weight_decay': 0.01
    })
    optimizer_grouped_parameters.append({
        'params': [p for n, p in model.named_parameters() 
                   if 'classifier' in n and any(nd in n for nd in no_decay)],
        'lr': classifier_lr,
        'weight_decay': 0.0
    })
    
    # BERT 层 - 分层学习率
    num_layers = 12
    for layer in range(num_layers):
        layer_lr = base_lr * (layer_decay ** (num_layers - layer - 1))
        
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() 
                       if f'bert.encoder.layer.{layer}.' in n and not any(nd in n for nd in no_decay)],
            'lr': layer_lr,
            'weight_decay': 0.01
        })
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() 
                       if f'bert.encoder.layer.{layer}.' in n and any(nd in n for nd in no_decay)],
            'lr': layer_lr,
            'weight_decay': 0.0
        })
    
    # Embedding - 最低学习率
    optimizer_grouped_parameters.append({
        'params': [p for n, p in model.named_parameters() 
                   if 'bert.embeddings' in n and not any(nd in n for nd in no_decay)],
        'lr': base_lr * 0.1,
        'weight_decay': 0.01
    })
    optimizer_grouped_parameters.append({
        'params': [p for n, p in model.named_parameters() 
                   if 'bert.embeddings' in n and any(nd in n for nd in no_decay)],
        'lr': base_lr * 0.1,
        'weight_decay': 0.0
    })
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    print(f"\n学习率设置:")
    print(f"  分类头: {classifier_lr}")
    print(f"  BERT 上层: {base_lr}")
    print(f"  BERT 下层: {base_lr * layer_decay:.2e} (衰减)")
    print(f"  Embedding: {base_lr * 0.1:.2e}\n")
    
    return optimizer

def main():
    set_seed(SEED)
    
    print("\n" + "="*60)
    print("BERT 训练 - 分层学习率策略")
    print("="*60)
    
    # 1. 加载数据
    print("\n加载数据...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    
    # 标签映射
    unique_labels = sorted(df_train['attitude'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    df_train['attitude'] = df_train['attitude'].map(label_to_id)
    df_val['attitude'] = df_val['attitude'].map(label_to_id)
    
    print(f"训练数据: {len(df_train):,}")
    print(f"验证数据: {len(df_val):,}")
    print(f"标签映射: {label_to_id}\n")
    
    # 计算类别权重
    if USE_CLASS_WEIGHTS:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(sorted(df_train['attitude'].unique())),
            y=df_train['attitude'].values
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        print(f"类别权重: {class_weights}\n")
    else:
        class_weights_tensor = None
    
    # 2. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = BertSentimentClassifier(MODEL_PATH, freeze_strategy=FREEZE_STRATEGY)
    model.to(DEVICE)
    
    print(f"可训练参数: {model.get_trainable_params_count():,}\n")
    
    # 3. 数据加载器
    from BERT_main_aggressive import create_dataloader
    train_loader = create_dataloader(df_train, tokenizer, BATCH_SIZE, text_column='cleaned_text')
    val_loader = create_dataloader(df_val, tokenizer, BATCH_SIZE, shuffle=False, text_column='cleaned_text')
    
    # 4. 分层优化器
    optimizer = create_layerwise_optimizer(model, BASE_LEARNING_RATE, CLASSIFIER_LEARNING_RATE)
    
    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler() if ENABLE_FP16 else None
    
    # 5. 训练
    print("开始训练...\n")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            scaler=scaler,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            enable_fp16=ENABLE_FP16,
            epoch=epoch
        )
        scheduler.step()
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        
        val_loss, val_acc, val_preds, val_labels_list = evaluate(
            model, val_loader, criterion, DEVICE, enable_fp16=ENABLE_FP16
        )
        
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f'{SAVE_PATH}/best_model.pt')
            print(f"✅ 最佳模型已保存")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发")
                break
        
        print()
    
    print("训练完成!")

if __name__ == "__main__":
    main()
