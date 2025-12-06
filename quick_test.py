"""
快速测试脚本 - 验证所有导入和基本功能
在云服务器上运行: python quick_test.py
"""

import sys
print("Python 版本:", sys.version)

try:
    print("\n✓ 导入 PyTorch...")
    import torch
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 导入 Transformers...")
    from transformers import AutoTokenizer, AutoModel
    print(f"  Transformers 版本: {__import__('transformers').__version__}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 导入 BERT_config...")
    from BERT_config import get_model_path
    model_path = get_model_path()
    print(f"  模型路径: {model_path}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 导入 BERT_model_aggressive...")
    from BERT_model_final import BertSentimentClassifier
    print(f"  BertSentimentClassifier 可用")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 导入 BERT_main_final...")
    from BERT_main_final import  SentimentDataset
    print(f"  BERT_main_final 可用")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 检查数据文件...")
    import pandas as pd
    import os
    if os.path.exists('./dataset/twitter_training_cleaned.csv'):
        df = pd.read_csv('./dataset/twitter_training_cleaned.csv')
        print(f"  训练数据: {len(df)} 行")
        print(f"  列: {df.columns.tolist()}")
        print(f"  标签类型: {df['attitude'].dtype}")
        print(f"  标签分布: {df['attitude'].value_counts().to_dict()}")
    else:
        print("  ✗ 找不到训练数据")
except Exception as e:
    print(f"  ✗ 失败: {e}")

try:
    print("\n✓ 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f"  分词器词汇量: {len(tokenizer)}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 加载数据...")
    df_train = pd.read_csv('./dataset/twitter_training_cleaned.csv')
    print(f"  数据加载成功: {len(df_train)} 条")
    print(f"  标签原始值: {df_train['attitude'].unique()}")
    
    # 建立标签映射（和 BERT_main_final.py 一致）
    label_map = {'Irrelevant': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3}
    if 'attitude_encoded' not in df_train.columns:
        df_train['attitude_encoded'] = df_train['attitude'].map(label_map)
    print(f"  标签映射: {label_map}")
    print(f"  编码后标签: {df_train['attitude_encoded'].unique()}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 创建 Dataset...")
    # 使用前10条数据测试，确保有 cleaned_text 和 attitude_encoded 列
    test_df = df_train.head(10)
    dataset = SentimentDataset(test_df, tokenizer, max_length=128)
    print(f"  Dataset 大小: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  样本键: {sample.keys()}")
    print(f"  input_ids 形状: {sample['input_ids'].shape}")
    print(f"  labels 类型: {type(sample['labels'])}, 值: {sample['labels'].item()}")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 创建模型 (freeze_strategy='none')...")
    model = BertSentimentClassifier(model_path, freeze_strategy='none')
    total_params = model.get_total_params_count()
    trainable_params = model.get_trainable_params_count()
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

try:
    print("\n✓ 测试前向传播...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    text = "I love this product!"
    encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    print(f"  输出形状: {logits.shape}")
    print(f"  ✓ 前向传播成功!")
except Exception as e:
    print(f"  ✗ 失败: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有检查通过！可以开始训练了")
print("="*60)
