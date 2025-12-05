"""
BERT 模型配置文件
支持在线和离线两种模式
"""

import os
from pathlib import Path

# ==================== 模型配置 ====================

# 选择模式: 'online' 或 'offline'
# online: 从 HuggingFace 在线下载 (需要网络连接)
# offline: 使用本地预下载的模型
MODE = 'offline'  # 改为 'online' 以使用在线模式

# ==================== 本地模型配置 ====================
# 当 MODE='offline' 时使用

# 本地模型目录路径
# 相对路径: 相对于当前脚本所在的目录
# 绝对路径: 完整的文件系统路径
LOCAL_MODELS_DIR = './models'

# 本地模型名称（与目录名称相同）
LOCAL_MODEL_NAME = 'bert-base-uncased'

# ==================== 在线模型配置 ====================
# 当 MODE='online' 时使用

# HuggingFace 模型名称
HUGGINGFACE_MODEL_NAME = 'bert-base-uncased'

# 其他可选模型:
# 'distilbert-base-uncased' - 轻量级，参数少 40%
# 'roberta-base'
# 'albert-base-v2'

# ==================== 网络配置 ====================

# HuggingFace API 超时时间（秒）
HF_TIMEOUT = 30

# 重试次数
HF_MAX_RETRIES = 5

# ==================== 工具函数 ====================

def get_model_path():
    """获取模型路径"""
    if MODE == 'offline':
        # 获取当前脚本所在目录
        current_dir = Path(__file__).parent
        model_path = current_dir / LOCAL_MODELS_DIR / LOCAL_MODEL_NAME
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"本地模型路径不存在: {model_path.absolute()}\n"
                f"请先运行 download_models.py 下载模型"
            )
        
        # 返回绝对路径字符串
        return str(model_path.absolute())
    
    elif MODE == 'online':
        return HUGGINGFACE_MODEL_NAME
    
    else:
        raise ValueError(f"未知的模式: {MODE}")


def get_model_config():
    """获取模型配置字典"""
    model_path = get_model_path()
    
    return {
        'model_name': model_path,
        'mode': MODE,
        'max_length': 128,
        'num_classes': 4,
        'dropout': 0.3,
        'cache_dir': None if MODE == 'online' else LOCAL_MODELS_DIR
    }


def print_config():
    """打印当前配置"""
    print("\n" + "="*60)
    print("BERT 模型配置")
    print("="*60)
    print(f"模式: {MODE}")
    
    if MODE == 'offline':
        current_dir = Path(__file__).parent
        model_path = current_dir / LOCAL_MODELS_DIR / LOCAL_MODEL_NAME
        print(f"本地模型目录: {model_path.absolute()}")
        
        if model_path.exists():
            files = list(model_path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            print(f"✅ 模型已存在 ({file_count} 个文件, {total_size_mb:.2f} MB)")
        else:
            print(f"❌ 模型不存在，请先运行 python check_model.py 检查")
            print(f"   或运行 python download_models.py 下载")
    
    else:  # online
        print(f"在线模型: {HUGGINGFACE_MODEL_NAME}")
        print(f"网络超时: {HF_TIMEOUT}s")
        print(f"重试次数: {HF_MAX_RETRIES}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()
