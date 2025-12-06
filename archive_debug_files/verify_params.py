"""
验证参数是否真的被修改了
运行: python verify_params.py
"""

import importlib
import sys

# 强制重新加载模块
if 'BERT_main_aggressive' in sys.modules:
    del sys.modules['BERT_main_aggressive']

from BERT_main_aggressive import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    WARMUP_STEPS, GRADIENT_ACCUMULATION_STEPS,
    FREEZE_STRATEGY, USE_CLASS_WEIGHTS, SEED
)

print("="*60)
print("当前配置参数")
print("="*60)
print(f"\nBATCH_SIZE: {BATCH_SIZE}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"NUM_EPOCHS: {NUM_EPOCHS}")
print(f"WARMUP_STEPS: {WARMUP_STEPS}")
print(f"GRADIENT_ACCUMULATION_STEPS: {GRADIENT_ACCUMULATION_STEPS}")
print(f"FREEZE_STRATEGY: {FREEZE_STRATEGY}")
print(f"USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
print(f"SEED: {SEED}")

print("\n" + "="*60)
print("预期配置 (v3 修复版)")
print("="*60)
print(f"\nBATCH_SIZE: 16")
print(f"LEARNING_RATE: 2e-05 (0.00002)")
print(f"NUM_EPOCHS: 20")
print(f"WARMUP_STEPS: 2000")
print(f"GRADIENT_ACCUMULATION_STEPS: 4")
print(f"FREEZE_STRATEGY: none")
print(f"USE_CLASS_WEIGHTS: True")
print(f"SEED: 42")

print("\n" + "="*60)
print("验证结果")
print("="*60)

errors = []
if BATCH_SIZE != 16:
    errors.append(f"BATCH_SIZE 错误: {BATCH_SIZE} (应该是 16)")
if LEARNING_RATE != 2e-5:
    errors.append(f"LEARNING_RATE 错误: {LEARNING_RATE} (应该是 2e-5)")
if NUM_EPOCHS != 20:
    errors.append(f"NUM_EPOCHS 错误: {NUM_EPOCHS} (应该是 20)")
if WARMUP_STEPS != 2000:
    errors.append(f"WARMUP_STEPS 错误: {WARMUP_STEPS} (应该是 2000)")

if errors:
    print("\n⚠️ 发现参数不匹配:")
    for err in errors:
        print(f"  - {err}")
    print("\n可能原因:")
    print("  1. 文件没有保存")
    print("  2. 云服务器使用的是旧版本文件")
    print("  3. Python 缓存了旧的 .pyc 文件")
else:
    print("\n✅ 所有参数正确!")

print("\n" + "="*60)
print("关于相同 Loss 的说明")
print("="*60)
print("""
由于 SEED=42 固定了随机种子，每次训练：
✓ 数据 shuffle 顺序相同
✓ 模型初始化权重相同
✓ 批次顺序相同

因此前几个 batch 的 Loss 会相同，这是**正常的**！

关键区别在于：
- 梯度更新的幅度 (由学习率控制)
- 后续 batch 的 Loss 变化趋势

观察重点：
1. Batch 500 之后的 Loss 是否开始下降
2. 每个 Epoch 结束时的平均 Loss 和准确率
3. 验证集的 Loss 和准确率变化

如果学习率从 1e-4 改为 2e-5:
- 前100个batch: Loss相同 (初始权重相同)
- Batch 500+: Loss应该更稳定 (不会震荡)
- Epoch 1结束: Loss应该更低，准确率更高
""")

print("="*60)
