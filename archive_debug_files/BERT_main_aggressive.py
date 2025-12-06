"""
BERT 情感分类 - 激进修复训练脚本
使用 BERT_model_aggressive.py 的激进优化
在云服务器上运行: python BERT_main_aggressive.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from BERT_model_final import BertSentimentClassifier, train_epoch, evaluate, predict_sample
from BERT_config import get_model_path

# ==================== 配置参数 ====================

# 数据路径
TRAIN_DATA_PATH = './dataset/twitter_training_cleaned.csv'
VAL_DATA_PATH = './dataset/twitter_validation_cleaned.csv'
MODEL_PATH = './models/bert-base-uncased'
SAVE_PATH = './saved_model_aggressive'

# 超参数 - 改进版本 v3 (修复学习率过高问题)
BATCH_SIZE = 16              # 降低批次大小
LEARNING_RATE = 2e-5         # 降低学习率 (完全解冻BERT需要更小LR)
NUM_EPOCHS = 20              # 增加训练轮数
WARMUP_STEPS = 2000          # 增加预热步数
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积
ENABLE_FP16 = True
FREEZE_STRATEGY = 'none'     # 完全解冻 BERT
USE_CLASS_WEIGHTS = True     # 使用类别权重
MAX_GRAD_NORM = 1.0          # 梯度裁剪 (防止梯度爆炸)

# 其他参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
DROPOUT = 0.3
# SEED = 42  # ❌ 取消固定种子，让每次训练结果不同
SEED = None  # ✅ 使用随机种子，每次初始化都不同

# ==================== 工具函数 ====================

class SentimentDataset(Dataset):
    """情感分析数据集"""
    def __init__(self, dataframe, tokenizer, label_column='attitude', max_length=128, text_column='cleaned_text'):
        """
        Args:
            dataframe: 包含文本和标签的 DataFrame (标签应已是整数)
            tokenizer: AutoTokenizer 对象
            label_column: 标签列名
            max_length: 最大序列长度
            text_column: 文本列名 ('cleaned_text' 或 'processed_text')
        """
        # 使用 cleaned_text 而不是 processed_text (保留完整句子结构)
        self.texts = dataframe[text_column].tolist()
        # 标签应该已经是整数，直接转换为列表
        self.labels = dataframe[label_column].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])  # 确保是整数
        
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
    """设置随机种子 (如果 seed=None 则不设置)"""
    if seed is None:
        print("⚠️ 未设置固定种子，每次训练将产生不同结果")
        return
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print(f"✓ 随机种子已设置为: {seed}")

def create_dataloader(df, tokenizer, batch_size=32, shuffle=True, max_length=128, text_column='cleaned_text'):
    """创建数据加载器"""
    dataset = SentimentDataset(df, tokenizer, max_length=max_length, text_column=text_column)
    # 使用 num_workers=0 避免 tokenizer 序列化问题
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def print_config():
    """打印配置信息"""
    print("\n" + "="*60)
    print("BERT 情感分类 - 改进版本 v2")
    print("="*60)
    print(f"\n数据配置:")
    print(f"  训练数据: {TRAIN_DATA_PATH}")
    print(f"  验证数据: {VAL_DATA_PATH}")
    print(f"  文本列: cleaned_text (完整句子) ★ 关键改动")
    print(f"\n模型配置:")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  冻结策略: {FREEZE_STRATEGY} (★ 完全解冻BERT)")
    print(f"  类别数: {NUM_CLASSES}")
    print(f"  Dropout: {DROPOUT}")
    print(f"\n训练配置:")
    print(f"  批次大小: {BATCH_SIZE} (↓ 从 128 降低)")
    print(f"  学习率: {LEARNING_RATE} (↑ 从 2e-5 提高)")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  预热步数: {WARMUP_STEPS}")
    print(f"  梯度累积: {GRADIENT_ACCUMULATION_STEPS}x")
    print(f"  混合精度: {ENABLE_FP16}")
    print(f"  设备: {DEVICE}")
    print("="*60 + "\n")

# ==================== 主训练逻辑 ====================

def main():
    set_seed(SEED)
    print_config()
    
    # 1. 加载数据
    print("加载数据...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    
    print(f"训练数据: {len(df_train):,}")
    print(f"验证数据: {len(df_val):,}")
    print(f"标签原始值示例: {df_train['attitude'].head(3).tolist()}")
    print(f"标签分布:\n{df_train['attitude'].value_counts()}\n")
    
    # 构建标签映射 (处理文本标签)
    unique_labels = sorted(df_train['attitude'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    print(f"标签映射: {label_to_id}")
    print(f"标签数量: {len(label_to_id)}\n")
    
    # 转换标签
    df_train['attitude'] = df_train['attitude'].map(label_to_id)
    df_val['attitude'] = df_val['attitude'].map(label_to_id)
    
    print(f"转换后标签分布:\n{df_train['attitude'].value_counts().sort_index()}\n")
    
    # 计算类别权重 (处理类别不平衡)
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
    
    # 2. 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    # 3. 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_dataloader(df_train, tokenizer, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(df_val, tokenizer, BATCH_SIZE, shuffle=False)
    print(f"数据加载器创建完成\n")
    
    # 4. 初始化模型
    print("初始化模型...")
    model = BertSentimentClassifier(
        model_name=MODEL_PATH,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        freeze_strategy=FREEZE_STRATEGY  # ★ 关键: 使用激进的冻结策略
    )
    model.to(DEVICE)
    
    # 打印参数统计
    total_params = model.get_total_params_count()
    trainable_params = model.get_trainable_params_count()
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)\n")
    
    # 5. 优化器和损失函数
    print("初始化优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    # 使用类别权重的损失函数
    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print("使用加权交叉熵损失函数\n")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失函数\n")
    
    # 学习率调度器
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # 混合精度
    scaler = GradScaler() if ENABLE_FP16 else None
    
    # 7. 训练循环
    print("开始训练...\n")
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            scaler=scaler,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            enable_fp16=ENABLE_FP16,
            epoch=epoch
        )
        scheduler.step()
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, val_preds, val_labels_list = evaluate(
            model, val_loader, criterion, DEVICE,
            enable_fp16=ENABLE_FP16
        )
        
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最好的模型
            Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f'{SAVE_PATH}/best_model.pt')
            print(f"✅ 最佳模型已保存 (损失: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发 (耐心已用尽: {patience})")
                break
        
        print()
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    
    # 8. 测试预测
    print("\n测试预测:")
    
    test_texts = [
        "I love this product, it's wonderful!",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "Best movie ever made!"
    ]
    
    preds, confs = predict_sample(model, tokenizer, test_texts, DEVICE, id_to_label)
    for text, pred, conf in zip(test_texts, preds, confs):
        print(f"  文本: {text}")
        print(f"  预测: {pred} (置信度: {conf:.4f})\n")


if __name__ == "__main__":
    main()
