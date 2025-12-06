"""
自动清理脚本 - 移动调试文件到归档文件夹
执行前请确保已备份项目！

使用方法：
python cleanup_final.py
"""

import os
import shutil
from pathlib import Path

# 要移动到归档的文件
DEBUG_FILES = [
    # 失败的训练版本
    'BERT_main_aggressive.py',
    'BERT_train_layerwise_lr.py',
    'BERT_train_optimized.py',
    
    # 测试脚本
    'test_overfitting.py',
    'test_overfitting_v2.py',
    'check_data_labels.py',
    'check_model.py',
    'check_training_process.py',
    'diagnose_bert.py',
    'diagnose_data_mismatch.py',
    'compare_text_columns.py',
    'verify_params.py',
    'quick_test.py',
    
    # 临时文档
    'AGGRESSIVE_FIX_GUIDE.md',
    'ANALYSIS_50_PERCENT.md',
    'DATA_COLUMN_ANALYSIS.txt',
    'EMERGENCY_FIX.txt',
    'IDENTICAL_LOSS_EXPLAINED.md',
    'CLEANUP_GUIDE.txt',
    
    # 辅助脚本
    'cleanup_project.py',
    'download_models.py',
]

# 保留的核心文件（用于检查）
KEEP_FILES = [
    'BERT_train_simple.py',
    'BERT_model_aggressive.py',
    'BERT_config.py',
    'text_process.py',
    'README.md',
    'LICENSE',
    '.gitignore',
]

def main():
    print("="*70)
    print("项目清理脚本 - 移动调试文件到归档")
    print("="*70)
    
    # 创建归档文件夹
    archive_dir = Path('archive_debug_files')
    archive_dir.mkdir(exist_ok=True)
    print(f"\n✓ 创建归档文件夹: {archive_dir}")
    
    # 统计
    moved_count = 0
    not_found_count = 0
    
    print(f"\n开始移动文件...")
    print("-" * 70)
    
    for file in DEBUG_FILES:
        file_path = Path(file)
        
        if file_path.exists():
            try:
                dest_path = archive_dir / file
                shutil.move(str(file_path), str(dest_path))
                print(f"✓ 已移动: {file}")
                moved_count += 1
            except Exception as e:
                print(f"✗ 移动失败: {file} - {e}")
        else:
            print(f"- 未找到: {file}")
            not_found_count += 1
    
    print("-" * 70)
    print(f"\n统计:")
    print(f"  已移动: {moved_count} 个文件")
    print(f"  未找到: {not_found_count} 个文件")
    
    # 检查核心文件
    print(f"\n检查核心文件...")
    print("-" * 70)
    
    all_exist = True
    for file in KEEP_FILES:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ 缺失: {file} ⚠️")
            all_exist = False
    
    print("-" * 70)
    
    if all_exist:
        print("\n✅ 清理完成！所有核心文件完整。")
    else:
        print("\n⚠️ 警告：部分核心文件缺失，请检查！")
    
    # 检查模型文件
    print(f"\n检查关键文件...")
    model_path = Path('best_model_simple.pth')
    bert_path = Path('models/bert-base-uncased')
    
    if model_path.exists():
        print(f"✓ 训练好的模型: {model_path}")
    else:
        print(f"⚠️ 未找到训练好的模型: {model_path}")
    
    if bert_path.exists():
        print(f"✓ BERT预训练模型: {bert_path}")
    else:
        print(f"✗ 未找到BERT模型: {bert_path}")
    
    print("\n" + "="*70)
    print("清理完成！")
    print("="*70)
    print(f"\n调试文件已移动到: {archive_dir}/")
    print("如需恢复，可以从归档文件夹中移回。")
    print("\n建议下一步:")
    print("  1. 测试运行: python BERT_train_simple.py")
    print("  2. 更新 README.md")
    print("  3. Git提交: git add . && git commit -m 'Project cleanup'")

if __name__ == '__main__':
    main()
