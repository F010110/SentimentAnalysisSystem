# 情感分析系统 (Sentiment Analysis System)

一个基于 PyTorch 和 BERT 的 Twitter 文本情感分析系统，支持 CPU/GPU 训练，优化了云服务器部署。

## 📊 项目概览

### 数据集
- **训练集**: 72,142 样本 (twitter_training_cleaned.csv)
- **验证集**: 1,000 样本 (twitter_validation_cleaned.csv)
- **分类类别**: 4 类（Irrelevant, Negative, Neutral, Positive）

### 模型架构
| 模型 | 准确率 | 备注 |
|------|--------|------|
| **BERT-base** | 🔄 训练中 | 110M 参数，微调版本 |
| SVM | 96.2% | 基线模型 |
| Random Forest | 95.9% | 基线模型 |
| XGBoost | 93.2% | 基线模型 |
| KNN | 93.6% | 基线模型 |
| GRU | ~96% | 自定义 GRU 架构 |

## 🚀 快速开始

### 前置要求
```bash
Python >= 3.8
PyTorch >= 1.9
transformers >= 4.20
pandas, numpy, scikit-learn, matplotlib, seaborn
```

### 环境配置

#### 选项 1: 本地 CPU 环境
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas numpy scikit-learn matplotlib seaborn
```

#### 选项 2: GPU 环境（CUDA 11.8）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn matplotlib seaborn
```

#### 选项 3: 使用 conda
```bash
conda create -n sentiment-analysis python=3.10
conda activate sentiment-analysis
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers pandas scikit-learn matplotlib seaborn
```

## 📁 项目结构

```
SentimentAnalysisSystem/
├── README.md                           # 本文件
│
├── 【BERT 模型】
├── BERT_main.py                        # BERT 训练主程序
├── BERT_model.py                       # BERT 模型定义和训练函数
├── BERT_config.py                      # BERT 配置（在线/离线模式）
├── download_models.py                  # 模型下载工具
├── check_model.py                      # 模型诊断工具
│
├── 【RNN 模型】
├── RNN_main.py                         # RNN 训练主程序
├── RNN_model.py                        # RNN 模型定义
├── text_process.py                     # 数据处理和预处理
│
├── 【数据文件】
├── dataset/
│   ├── twitter_training.csv            # 原始训练集
│   ├── twitter_training_cleaned.csv    # 清洗后的训练集
│   ├── twitter_validation.csv          # 原始验证集
│   └── twitter_validation_cleaned.csv  # 清洗后的验证集
│
├── 【模型文件】(运行后生成)
├── models/                             # HuggingFace 模型缓存
│   └── bert-base-uncased/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
│
├── 【输出文件】(训练后生成)
├── bert_model_final.pt                 # BERT 最终模型
├── bert_model_best.pt                  # BERT 最佳模型
├── bert_training_history.png           # 训练历史曲线
├── bert_confusion_matrix.png           # 混淆矩阵
├── sentiment_model.pth                 # RNN 模型检查点
├── predictions.csv                     # 预测结果
└── training_history.png                # RNN 训练曲线
```

## 📋 针对BERT的完整使用流程（其他模型直接运行即可）

### 步骤 1: 下载模型（本地执行）

如果你的服务器**无法访问 HuggingFace**，需要先在本地下载模型，然后上传到服务器。

#### 1.1 本地下载模型
在你的**本地机器**运行（需要网络连接）：
```bash
python download_models.py
```

输出示例：
```
============================================================
HuggingFace 模型离线下载工具 v2
============================================================
模型保存目录: /path/to/models

✅ 网络连接正常

============================================================
正在下载: bert-base-uncased
============================================================
...
✅ 分词器已下载
✅ 模型已下载
保存到本地目录: ./models/bert-base-uncased
✅ 文件已保存
📦 总大小: 417.92 MB

已保存的文件:
  ✓ config.json (0.01 MB)
  ✓ model.safetensors (417.66 MB)
  ✓ tokenizer.json (0.68 MB)
  ...
```

#### 1.2 验证模型完整性
```bash
python check_model.py
```

输出应该显示：
```
✅ 模型已存在 (6 个文件, 417.92 MB)
✅ 分词器加载成功!
✅ 模型加载成功!
✅ 所有检查都通过了！
```

### 步骤 2: 上传到服务器

将以下内容上传到云服务器：
```
SentimentAnalysisSystem/
├── BERT_main.py
├── BERT_model.py
├── BERT_config.py
├── text_process.py
├── dataset/
│   └── twitter_training_cleaned.csv
│   └── twitter_validation_cleaned.csv
└── models/                    ← 关键：包含下载的模型
    └── bert-base-uncased/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

### 步骤 3: 服务器配置

登录到云服务器后，编辑 `BERT_config.py`：

```python
# BERT_config.py

# 选择模式
MODE = 'offline'  # ✅ 使用本地模型（推荐）
# MODE = 'online'   # 如果服务器能访问 HuggingFace，可以改为在线

# 本地模型配置
LOCAL_MODELS_DIR = './models'  # 相对路径或绝对路径
LOCAL_MODEL_NAME = 'bert-base-uncased'
```

### 步骤 4: 训练模型

在服务器上运行训练：

#### 4.1 快速测试（验证代码可用）
```bash
# 修改 BERT_main.py 中的 NUM_EPOCHS = 1 进行快速测试
python BERT_main.py
```

#### 4.2 完整训练
```bash
# 恢复 BERT_main.py 中的 NUM_EPOCHS = 5
python BERT_main.py
```

训练输出示例：
```
============================================================
BERT 情感分析模型 - Kaggle GPU 优化版本
============================================================

============================================================
BERT 模型配置
============================================================
模式: offline
本地模型目录: /root/Workspace/SentimentAnalysisSystem/models/bert-base-uncased
✅ 模型已存在 (6 个文件, 417.92 MB)
============================================================

步骤 1: 加载和预处理数据
============================================================
训练集大小: 72142
验证集大小: 1000
检测到 4 个类别:
标签映射: {0: 'Irrelevant', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

步骤 2: 初始化分词器
============================================================
加载分词器: /root/Workspace/SentimentAnalysisSystem/models/bert-base-uncased
从本地加载分词器...

...

Epoch 1/5
Epoch [1/5] Batch [50/564] Loss: 0.5621
Epoch [1/5] Batch [100/564] Loss: 0.4532
...

训练损失: 0.2145 | 训练准确率: 0.9234
验证损失: 0.2856 | 验证准确率: 0.9156
```

### 步骤 5: 查看结果

训练完成后会生成以下文件：
- `bert_training_history.png` - 训练曲线
- `bert_confusion_matrix.png` - 混淆矩阵
- `bert_model_final.pt` - 最终模型
- 控制台输出详细的分类报告

## 🎯 关键配置参数

编辑 `BERT_main.py` 中的 `main()` 函数：

```python
# 模型配置 - RTX 4090 优化版本 (24GB VRAM)
MODEL_NAME = 'bert-base-uncased'           # 模型选择
MAX_LENGTH = 128                           # 最大序列长度
NUM_CLASSES = 4                            # 分类类别数
BATCH_SIZE = 128                           # 批大小（RTX 4090）
LEARNING_RATE = 2e-5                       # 学习率
NUM_EPOCHS = 5                             # 训练轮数
WARMUP_STEPS = 300                         # 预热步数
GRADIENT_ACCUMULATION_STEPS = 2            # 梯度累积
ENABLE_FP16 = True                         # 混合精度
NUM_WORKERS = 4                            # 数据加载进程数
```

### 根据硬件调整 Batch Size

| 硬件 | 推荐 Batch Size | 显存占用 |
|------|-----------------|---------|
| RTX 4090 (24GB) | 128 | ~8GB |
| RTX 3090 (24GB) | 96-128 | ~18GB |
| RTX 3080 (10GB) | 32-64 | ~8GB |
| Tesla V100 (16GB) | 64-96 | ~12GB |
| Kaggle (16GB) | 32-64 | ~10GB |
| CPU | 8-16 | 内存 |

## 📊 性能对比

### 硬件性能

| 设备 | 每个 Epoch | 5 Epochs 总时间 |
|------|-----------|-----------------|
| **RTX 4090** (BS=128) | ~0.5 分钟 | **~2.5 分钟** ⚡ |
| RTX 3090 (BS=96) | ~1 分钟 | ~5 分钟 |
| RTX 3080 (BS=64) | ~2 分钟 | ~10 分钟 |
| Tesla V100 (BS=64) | ~1.5 分钟 | ~7.5 分钟 |
| Kaggle GPU (BS=32) | ~3 分钟 | ~15 分钟 |
| CPU | ~30 分钟 | ~150 分钟 |

### 模型性能（参考）

- **BERT-base**: 准确率 ~92-93% (微调后)
- **蒸馏 BERT**: 准确率 ~90-91% (更轻量)
- **RoBERTa-base**: 准确率 ~93-94% (更强)

## 🔧 故障排查

### 问题 1: 模型加载失败
```
FileNotFoundError: 本地模型路径不存在
```
**解决方案:**
1. 确认 `models/bert-base-uncased/` 文件夹存在
2. 运行 `python check_model.py` 检查模型文件
3. 确认 `BERT_config.py` 中 `MODE = 'offline'`

### 问题 2: HuggingFace 超时
```
ReadTimeoutError: HTTPSConnectionPool timeout
```
**解决方案:**
1. 使用离线模式（在本地下载后上传）
2. 或增加超时时间：编辑 `BERT_config.py` 中的 `HF_TIMEOUT`

### 问题 3: 显存不足
```
CUDA out of memory
```
**解决方案:**
1. 减少 `BATCH_SIZE`（如从 128 改为 64）
2. 减少 `MAX_LENGTH`（如从 128 改为 96）
3. 设置 `GRADIENT_ACCUMULATION_STEPS = 4`

### 问题 4: 数据加载缓慢
**解决方案:**
1. 增加 `NUM_WORKERS`（如从 4 改为 8）
2. 确保使用 `pin_memory=True`

## 💾 数据格式

### 训练/验证数据
必须包含以下列：
- `processed_text` (或 `text`): 处理后的文本
- `attitude`: 情感标签（支持字符串或数字）

示例：
```csv
processed_text,attitude
"this movie is great",Positive
"i hate this",Negative
"it is okay",Neutral
"not relevant",Irrelevant
```

## 🌐 在线模式配置

如果你的服务器**可以访问 HuggingFace**，可以使用在线模式：

```python
# BERT_config.py
MODE = 'online'  # 改为在线模式

# HuggingFace 模型选项
HUGGINGFACE_MODEL_NAME = 'bert-base-uncased'
# 其他选项:
# 'distilbert-base-uncased' - 轻量版（参数少 40%）
# 'roberta-base' - 更强版本
# 'albert-base-v2' - 轻量版
```

在线模式会自动从 HuggingFace 下载，首次运行需要网络。

## 📚 相关资源

- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch 文档](https://pytorch.org/docs/)

## ✨ 优化技术

本模型（BERT）使用以下优化技术：

- **混合精度训练 (FP16)**: 加速 20-30%，节省 50% 显存
- **梯度累积**: 有效增大 Batch Size，不额外占用显存
- **梯度裁剪**: 防止梯度爆炸
- **预热 (Warmup)**: 稳定学习率变化
- **多进程数据加载**: 充分利用 CPU
- **内存锁定 (Pin Memory)**: 加快 CPU-GPU 数据传输

## 📄 许可证

MIT License

---

**最后更新**: 2025-12-05  
**版本**: 2.0 (BERT + RNN)

