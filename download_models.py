"""
下载 HuggingFace 模型和分词器到本地
在本地运行此脚本，然后将 models 文件夹上传到服务器
"""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# 创建本地模型存储目录
MODELS_DIR = Path('./models')
MODELS_DIR.mkdir(exist_ok=True)

# 需要下载的模型列表
MODELS_TO_DOWNLOAD = [
    'bert-base-uncased',
    # 可选: 其他轻量化模型
    # 'distilbert-base-uncased',
    # 'roberta-base',
]

def download_model(model_name):
    """下载模型和分词器"""
    print(f"\n{'='*60}")
    print(f"正在下载: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 模型保存路径
        model_save_path = MODELS_DIR / model_name
        
        # 如果目录已存在，先删除它
        if model_save_path.exists():
            print(f"清理旧的文件...")
            shutil.rmtree(model_save_path)
        
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 下载分词器
        print(f"下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ 分词器已下载")
        
        # 下载模型
        print(f"下载模型 (这可能需要几分钟)...")
        model = AutoModel.from_pretrained(model_name)
        print(f"✅ 模型已下载")
        
        # 保存到本地目录
        print(f"保存到本地目录: {model_save_path}")
        tokenizer.save_pretrained(str(model_save_path))
        model.save_pretrained(str(model_save_path))
        print(f"✅ 文件已保存")
        
        # 获取模型文件大小
        total_size = sum(
            f.stat().st_size for f in model_save_path.rglob('*') if f.is_file()
        )
        total_size_mb = total_size / (1024 * 1024)
        print(f"📦 总大小: {total_size_mb:.2f} MB")
        
        # 列出已保存的文件
        print(f"\n已保存的文件:")
        files = sorted([f for f in model_save_path.rglob('*') if f.is_file()])
        for f in files:
            rel_path = f.relative_to(model_save_path)
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  ✓ {rel_path} ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("HuggingFace 模型离线下载工具 v2")
    print("="*60)
    print(f"模型保存目录: {MODELS_DIR.absolute()}\n")
    
    # 检查网络连接
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        print("✅ 网络连接正常\n")
    except Exception as e:
        print(f"❌ 网络连接失败: {e}")
        print("请检查网络连接后重试\n")
        return
    
    # 下载所有模型
    success_count = 0
    for model_name in MODELS_TO_DOWNLOAD:
        if download_model(model_name):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"下载完成: {success_count}/{len(MODELS_TO_DOWNLOAD)} 个模型")
    print(f"{'='*60}")
    print(f"\n📁 文件夹结构:")
    print(f"models/")
    for model_name in MODELS_TO_DOWNLOAD:
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            files = list(model_path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            total_size_mb = total_size / (1024 * 1024)
            print(f"  └─ {model_name}/ ({file_count} 个文件, {total_size_mb:.2f} MB)")
    
    print(f"\n📝 后续步骤:")
    print(f"1. 验证文件完整性: python check_model.py")
    print(f"2. 将 'models' 文件夹上传到云服务器")
    print(f"3. 在云服务器上修改 BERT_config.py 中的 MODE = 'offline'")
    print(f"4. 运行 BERT_main.py，代码会使用本地模型\n")


if __name__ == "__main__":
    main()

