# 情感分析系统 (Sentiment Analysis System)

一个基于 PyTorch 和 BERT 的 Twitter 文本情感分析系统，实现了 **98% 验证准确率**，支持 CPU/GPU 训练。

## 📊 项目概览

### 数据集
- **训练集**: 72,142 样本 (twitter_training_cleaned.csv)
- **验证集**: 1,000 样本 (twitter_validation_cleaned.csv)
- **分类类别**: 4 类（Irrelevant, Negative, Neutral, Positive）

### 模型性能
| 模型 | 准确率 | 参数量 | 备注 |
|------|--------|--------|------|
| **BERT-base (最终)** | **98.0%** ✅ | 43.6M (可训练) | 冻结前6层，微调后6层 |
| SVM | 96.2% | - | 基线模型 |
| GRU | ~96% | - | 自定义 GRU 架构 |
| Random Forest | 95.9% | - | 基线模型 |
| KNN | 93.6% | - | 基线模型 |
| XGBoost | 93.2% | - | 基线模型 |

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
├── README.md                           # 项目说明文档
├── LICENSE                             # 开源协议
├── .gitignore                          # Git 配置
├── down_models.py                      # 下载模型
|
├── 【BERT 模型 - 最终版本】
├── BERT_main_final.py                  # ⭐ BERT 训练主程序（98%准确率）
├── BERT_model_final.py                 # ⭐ BERT 模型定义和训练函数
├── BERT_config.py                      # BERT 配置（本地/在线模式）
│
├── 【RNN 基线模型】
├── RNN_main.py                         # RNN 训练主程序
├── RNN_model.py                        # RNN 模型定义
│
├── 【数据处理】
├── text_process.py                     # 文本预处理工具
├── data_cleaning.ipynb                 # 数据清洗过程记录
├── baseline_model.ipynb                # 基线模型实验记录
│
├── 【数据集】
├── dataset/
│   ├── twitter_training_cleaned.csv    # ⭐ 清洗后的训练集（72,142条）
│   ├── twitter_validation_cleaned.csv  # ⭐ 清洗后的验证集（1,000条）
│   ├── twitter_training.csv            # 原始训练集
│   └── twitter_validation.csv          # 原始验证集
│
├── 【预训练模型】
├── models/
│   └── bert-base-uncased/              # ⭐ BERT 预训练模型（运行down_models.py）
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── vocab.txt
│
├── 【训练输出】（运行后生成）
├── best_model_simple.pth               # ⭐ 训练好的最佳模型（98%准确率）
├── training_log.txt                    # 完整训练日志
│
├── 【项目维护】
├── cleanup_final.py                    # 项目清理脚本
├── PROJECT_CLEANUP_FINAL.md            # 清理指南
└── archive_debug_files/                # 调试文件归档（可选）
    ├── test_overfitting*.py            # 过拟合测试
    ├── diagnose_*.py                   # 诊断脚本
    ├── check_*.py                      # 检查工具
    └── BERT_*_old.py                   # 旧版本训练代码
```

## 📋 使用指南

### 快速开始：BERT 模型训练

**前置条件**：
- 已准备好数据集：`dataset/twitter_training_cleaned.csv` 和 `dataset/twitter_validation_cleaned.csv`
- 已运行down_models.py，下载 BERT 模型到 `models/bert-base-uncased/`

**一键训练**：
```bash
python BERT_main_final.py
```

**训练配置（已优化到最佳性能）**：
- 学习率：`1e-5`
- Batch Size：`8`
- 训练轮数：`50`（通常15-20轮即可收敛）
- 冻结策略：`half`（冻结前6层，训练后6层）
- Dropout：`0.1`
- 梯度裁剪：`1.0`

**预期结果**：
- Epoch 1：验证准确率 ~69%
- Epoch 5-10：验证准确率 ~85-90%
- Epoch 15-20：验证准确率 ~95-98%

**训练完成后**：
- 最佳模型保存在：`best_model_simple.pth`
- 训练日志保存在：`training_log.txt`

### RNN（实际上是GRU） 基线模型（可选）

```bash
python RNN_main.py
```

### 下载 BERT 预训练模型（首次使用）

如果 `models/bert-base-uncased/` 文件夹不存在，需要先下载模型：

```python
# 方法1：使用 transformers 库自动下载（需要网络）
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('bert-base-uncased', cache_dir='./models')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models')

# 方法2：运行down_models.py（推荐，需要网络）

```



## 🔧 环境配置

### 前置要求
```bash
Python >= 3.8
PyTorch >= 1.9
transformers >= 4.20
pandas, numpy, scikit-learn
```

### 安装依赖

#### 选项 1: GPU 环境（推荐，用于训练）
```bash
# 安装 PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install transformers pandas numpy scikit-learn
```

#### 选项 2: CPU 环境（用于推理）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas numpy scikit-learn
```

#### 选项 3: 使用 conda
```bash
conda create -n sentiment python=3.10
conda activate sentiment
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers pandas scikit-learn
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

## 🎯 关键技术要点

### BERT 模型优化策略

**最终成功配置**（98%准确率）：
```python
BATCH_SIZE = 8              # 小批次，更频繁的权重更新
LEARNING_RATE = 1e-5        # 低学习率，稳定训练
FREEZE_STRATEGY = 'half'    # 冻结前6层，只训练后6层
DROPOUT = 0.1               # 低dropout，充分学习
MAX_GRAD_NORM = 1.0         # 梯度裁剪，防止梯度爆炸
NUM_EPOCHS = 50             # 充足的训练轮数
```

### 训练失败原因总结（供参考）

经过多次实验，发现以下配置会导致训练失败（准确率停留在25-30%）：

❌ **错误配置**：
- 梯度累积步数过大（>1）：导致有效batch size过大（64+），更新频率太低
- 学习率过高（>2e-5）：导致梯度爆炸
- Dropout过高（>0.3）：训练时丢弃过多信息
- 完全解冻BERT：110M参数在小数据集上容易过拟合

✅ **成功要素**：
1. **小batch size（8）+ 无梯度累积** → 频繁更新权重
2. **低学习率（1e-5）** → 稳定的梯度下降
3. **部分冻结（half）** → 减少可训练参数到43.6M
4. **低dropout（0.1）** → 保留更多训练信息
5. **梯度裁剪（1.0）** → 防止梯度爆炸

### 性能对比

| 配置 | Epoch 1 准确率 | 最终准确率 | 收敛轮数 |
|------|----------------|-----------|---------|
| 失败配置 | ~30% | ~30% | 不收敛 |
| 成功配置 | **69%** | **98%** | 15-20轮 |

## 💡 使用建议

### 训练监控

训练过程中关注以下指标：
- **Epoch 1 验证准确率**：应该在 60-70% 之间
- **Loss 下降趋势**：应该平稳下降，不应该震荡或上升
- **训练/验证准确率差距**：差距过大说明过拟合

### 早停策略

代码内置早停机制：
```python
patience = 10  # 连续10轮无提升则停止
```

### 模型推理

加载训练好的模型进行预测：

```python
import torch
from BERT_model_final import BertSentimentClassifier
from transformers import AutoTokenizer

# 加载模型
model = BertSentimentClassifier(
    model_name='models/bert-base-uncased',
    num_classes=4,
    dropout=0.1,
    freeze_strategy='half'
)
model.load_state_dict(torch.load('best_model_simple.pth'))
model.eval()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/bert-base-uncased', local_files_only=True)

# 预测
text = "I love this product!"
encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

with torch.no_grad():
    output = model(encoding['input_ids'], encoding['attention_mask'])
    prediction = torch.argmax(output, dim=1).item()

label_map = {0: 'Irrelevant', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}
print(f"预测结果: {label_map[prediction]}")
```

## 🔧 故障排查

### 问题 1: 训练准确率停在 25-30%
**原因**: 梯度累积过大或学习率不当  
**解决方案**: 使用 `BERT_main_final.py` 的默认配置

### 问题 2: 显存不足
```
CUDA out of memory
```
**解决方案**:
- 减少 `BATCH_SIZE` (8 → 4)
- 减少 `MAX_LENGTH` (128 → 96)

### 问题 3: 模型加载失败
```
FileNotFoundError: models/bert-base-uncased
```
**解决方案**: 下载 BERT 模型到 `models/` 文件夹

## 📚 数据格式

训练数据必须包含以下列：
- `cleaned_text`: 清洗后的文本
- `attitude`: 情感标签（Irrelevant/Negative/Neutral/Positive 或 0/1/2/3）

示例：
```csv
cleaned_text,attitude
im getting on borderlands and i will murder you all,Positive
this movie is great,Positive
i hate this,Negative
it is okay,Neutral
not relevant,Irrelevant
```

## 📊 项目成果

### 最终性能指标
- **验证准确率**: 98.0%
- **训练准确率**: 99.5%
- **训练时长**: ~2小时（RTX 4090，20 epochs）
- **收敛速度**: 15-20 epochs

### 与基线模型对比
- 超越 SVM (96.2%) **+1.8%**
- 超越 GRU (~96%) **+2.0%**
- 超越 Random Forest (95.9%) **+2.1%**

## 📚 参考资源

- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch 文档](https://pytorch.org/docs/)

## 📝 项目维护

### 清理调试文件

项目包含大量调试和测试文件，已移至 `archive_debug_files/`：

```bash
# 查看归档内容
ls archive_debug_files/

# 如需恢复某个文件
cp archive_debug_files/test_overfitting_v2.py .
```

### 核心文件说明

| 文件 | 用途 | 是否必需 |
|------|------|---------|
| `BERT_main_final.py` | 训练脚本 | ✅ 必需 |
| `BERT_model_final.py` | 模型定义 | ✅ 必需 |
| `best_model_simple.pth` | 训练好的模型 | ✅ 必需（推理用）|
| `models/bert-base-uncased/` | 预训练模型 | ✅ 必需 |
| `dataset/*.csv` | 数据集 | ✅ 必需（训练用）|
| `archive_debug_files/` | 调试文件 | ❌ 可选 |

## 📄 许可证

MIT License

## 🙏 致谢

- HuggingFace 提供预训练 BERT 模型
- PyTorch 深度学习框架
- Kaggle 提供数据集

---

**项目状态**: ✅ 已完成 (2025-12-06)  
**最终准确率**: 98.0%  
**作者**: F010110

**最后更新**: 2025-12-06  
**版本**: 3.0 (BERT + GRU)

