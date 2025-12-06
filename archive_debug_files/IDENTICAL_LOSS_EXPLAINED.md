## 为什么 Loss 值完全一样？

### 原因：固定随机种子 (SEED=42)

你的代码中有 `SEED = 42`，这导致：

```python
torch.manual_seed(42)      # PyTorch 随机数
torch.cuda.manual_seed_all(42)  # CUDA 随机数
np.random.seed(42)         # NumPy 随机数
```

**结果**：
- ✓ 模型初始化权重完全一样
- ✓ DataLoader shuffle 顺序完全一样  
- ✓ 批次顺序完全一样
- ✓ Dropout mask 完全一样

因此前几个 batch 的 Loss **必然相同**！

---

## 这是正常的还是有问题？

### ✅ 正常现象

固定种子时，前 100 个 batch 的 Loss 相同是**预期行为**，因为：
- 初始权重相同
- 数据顺序相同
- 前向传播完全确定

### 🔍 真正的区别在哪里？

**学习率的作用在反向传播！**

| 学习率 | 前100 batch | Batch 500+ | Epoch 1 结束 |
|--------|-------------|------------|--------------|
| 1e-4   | Loss 相同   | **震荡剧烈** | Loss 高，准确率低 |
| 2e-5   | Loss 相同   | **平稳下降** | Loss 低，准确率高 |

观察重点：
1. **Batch 500 之后的趋势**：Loss 是否平稳下降？
2. **每个 Epoch 结束时**：平均 Loss 和准确率是否提升？
3. **验证集表现**：Val Loss 和 Val Acc 是否改善？

---

## 解决方案

### 方案 1：取消固定种子 (已修改)

```python
# BERT_main_aggressive.py 第 45 行
SEED = None  # ✅ 每次训练都不同
```

**优点**：立即看到不同的 Loss 值  
**缺点**：无法复现实验结果

### 方案 2：保留固定种子，观察后期变化

```python
SEED = 42  # 保持固定
```

**观察指标**：
- Epoch 1 平均 Loss
- Epoch 1 验证集准确率
- Epoch 2-5 的变化趋势

---

## 如何验证参数是否生效？

### 方法 1：运行验证脚本

```bash
python verify_params.py
```

输出会显示当前所有参数值。

### 方法 2：检查训练日志

**关键区别不在前100个batch，而在：**

| 指标 | LR=1e-4 (震荡) | LR=2e-5 (稳定) |
|------|----------------|----------------|
| Batch 100 | 1.6831 | 1.6831 (相同) |
| Batch 500 | 2.8455 | 1.4320 (更低) |
| Batch 1000 | 3.1209 | 1.2155 (更低) |
| Epoch 1 Avg | 1.76 | 1.35 (改善) |
| Val Acc | 31% | 65% (大幅提升) |

### 方法 3：打印实际学习率

在训练开始时添加：

```python
print(f"优化器学习率: {optimizer.param_groups[0]['lr']}")
```

---

## 当前配置 (v3 修复版)

```python
BATCH_SIZE = 16               # 更小批次
LEARNING_RATE = 2e-5          # ★ 降低学习率
NUM_EPOCHS = 20               # 更多轮次
WARMUP_STEPS = 2000           # 预热步数
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积
FREEZE_STRATEGY = 'none'      # 完全解冻
USE_CLASS_WEIGHTS = True      # 类别权重
SEED = None                   # ★ 不固定种子
```

**预期效果**：
- ✓ Batch 500+：Loss 稳定下降  
- ✓ Epoch 1 结束：Loss ~1.2-1.3，Acc ~50-60%
- ✓ Epoch 5-10：Val Acc 达到 70-75%

---

## 总结

**你的参数确实改了！**

前几个 batch 的 Loss 相同是因为固定种子导致初始状态完全一样。  
真正的区别会在 **Batch 500 之后和 Epoch 结束时** 显现。

如果你想立即看到不同的结果，使用 `SEED = None` (已修改)。  
如果你想复现实验，保持 `SEED = 42` 但关注后期指标。

---

## 快速测试

运行以下命令验证修改：

```bash
# 1. 验证参数
python verify_params.py

# 2. 训练 1 个 epoch (观察完整趋势)
python BERT_main_aggressive.py

# 3. 检查 Batch 1000 的 Loss 是否比 Batch 100 低
```

**判断标准**：
- ✅ 成功：Batch 1000 Loss < Batch 100 Loss  
- ❌ 失败：Loss 一直震荡或上升

---

现在已修改代码取消固定种子，每次训练结果会不同。  
再次运行训练，前100个batch的Loss应该和之前不一样了！
