# 项目文件整理脚本
# 运行: python cleanup_project.py

import os
import shutil
from pathlib import Path

# 当前项目根目录
ROOT = Path(__file__).parent

# 定义文件分类
FILES_TO_KEEP = {
    "核心训练文件": [
        "BERT_main_aggressive.py",  # 激进修复的训练脚本 (推荐使用)
        "BERT_model_aggressive.py",  # 激进修复的模型
        "BERT_config.py",            # 配置文件
        "text_process.py",           # 数据处理
    ],
    "原始文件(备份)": [
        "BERT_main.py",              # 原始训练脚本 (有问题，但保留)
        "BERT_model.py",             # 原始模型 (有问题，但保留)
        "RNN_main.py",               # RNN训练脚本 (备用)
        "RNN_model.py",              # RNN模型 (备用)
    ],
    "工具脚本": [
        "quick_test.py",             # 快速测试脚本
        "diagnose_bert.py",          # 诊断脚本
        "download_models.py",        # 下载模型脚本
        "check_model.py",            # 检查模型脚本
    ],
    "数据处理": [
        "data_cleaning.ipynb",       # 数据清洗 Notebook
        "baseline_model.ipynb",      # 基线模型 Notebook
    ],
    "文档": [
        "README.md",                 # 项目说明
        "LICENSE",                   # 许可证
        "AGGRESSIVE_FIX_GUIDE.md",   # 激进修复指南
        ".gitignore",                # Git忽略文件
    ],
    "临时/日志": [
        "log.txt",                   # 训练日志
        "FIX_SUMMARY.txt",           # 修复摘要
    ]
}

FILES_TO_DELETE = [
    "log.txt",                       # 临时日志
    "FIX_SUMMARY.txt",               # 临时修复说明
    "check_model.py",                # 不再需要
]

FOLDERS_TO_KEEP = [
    "dataset/",                      # 数据集
    "models/",                       # 预训练模型
    ".git/",                         # Git版本控制
    "__pycache__/",                  # Python缓存 (可删除但会自动生成)
]

FOLDERS_TO_DELETE = [
    "saved_model/",                  # 旧的训练模型 (如果有)
    "__pycache__/",                  # Python缓存 (可选)
]


def create_backup():
    """创建备份文件夹"""
    backup_dir = ROOT / "backup_old_files"
    backup_dir.mkdir(exist_ok=True)
    return backup_dir


def cleanup():
    """执行清理"""
    print("="*60)
    print("项目文件整理工具")
    print("="*60)
    
    # 1. 创建备份
    backup_dir = create_backup()
    print(f"\n✓ 创建备份文件夹: {backup_dir}")
    
    # 2. 移动要删除的文件到备份
    print("\n移动临时文件到备份:")
    for filename in FILES_TO_DELETE:
        filepath = ROOT / filename
        if filepath.exists():
            backup_path = backup_dir / filename
            shutil.move(str(filepath), str(backup_path))
            print(f"  ✓ {filename} -> backup/")
        else:
            print(f"  - {filename} (不存在)")
    
    # 3. 删除临时文件夹
    print("\n清理临时文件夹:")
    for foldername in FOLDERS_TO_DELETE:
        folderpath = ROOT / foldername
        if folderpath.exists():
            choice = input(f"  删除 {foldername}? (y/n): ")
            if choice.lower() == 'y':
                shutil.rmtree(folderpath)
                print(f"  ✓ 已删除: {foldername}")
            else:
                print(f"  - 跳过: {foldername}")
        else:
            print(f"  - {foldername} (不存在)")
    
    # 4. 显示保留的文件
    print("\n" + "="*60)
    print("保留的文件:")
    print("="*60)
    for category, files in FILES_TO_KEEP.items():
        print(f"\n{category}:")
        for filename in files:
            filepath = ROOT / filename
            if filepath.exists():
                size = filepath.stat().st_size / 1024  # KB
                print(f"  ✓ {filename:30s} ({size:>8.1f} KB)")
            else:
                print(f"  ✗ {filename:30s} (缺失)")
    
    print("\n" + "="*60)
    print("整理完成!")
    print("="*60)
    print(f"\n已删除的文件保存在: {backup_dir}")
    print("如果需要恢复，请从备份文件夹复制回来。")


if __name__ == "__main__":
    print("\n警告: 此脚本将删除临时文件并整理项目结构。")
    confirm = input("确认继续? (yes/no): ")
    
    if confirm.lower() == 'yes':
        cleanup()
    else:
        print("已取消。")
