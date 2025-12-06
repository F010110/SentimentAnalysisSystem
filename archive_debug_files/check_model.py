"""
诊断脚本：检查模型文件是否完整
"""

import os
from pathlib import Path

def check_model_structure():
    """检查本地模型文件结构"""
    print("\n" + "="*60)
    print("BERT 模型文件结构检查")
    print("="*60 + "\n")
    
    # 获取当前脚本目录
    current_dir = Path(__file__).parent
    models_dir = current_dir / 'models' / 'bert-base-uncased'
    
    print(f"检查目录: {models_dir}")
    print(f"目录存在: {models_dir.exists()}\n")
    
    if not models_dir.exists():
        print("❌ 模型目录不存在!")
        print("请先运行: python download_models.py\n")
        return False
    
    # 需要的关键文件
    required_files = [
        'config.json',
        'tokenizer.json',
        'vocab.txt',
    ]
    
    # 模型文件（两种格式都可以，新版本用 safetensors）
    model_files = [
        'pytorch_model.bin',      # 旧版本
        'model.safetensors',      # 新版本
    ]
    
    print("检查关键文件:")
    all_exist = True
    for file_name in required_files:
        file_path = models_dir / file_name
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_name}")
        if not exists:
            all_exist = False
    
    # 检查模型文件
    print("\n检查模型文件 (需要至少一个):")
    has_model = False
    for file_name in model_files:
        file_path = models_dir / file_name
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file_name}")
        if exists:
            has_model = True
    
    if not has_model:
        print("\n❌ 没有找到模型文件！")
        all_exist = False
    
    # 列出所有文件
    print(f"\n所有文件列表:")
    if models_dir.exists():
        all_files = list(models_dir.rglob('*'))
        file_list = [f for f in all_files if f.is_file()]
        for f in sorted(file_list):
            relative_path = f.relative_to(models_dir)
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {relative_path} ({size_mb:.2f} MB)")
    
    print("\n" + "="*60)
    if all_exist:
        print("✅ 模型文件完整！")
        print("="*60 + "\n")
        return True
    else:
        print("❌ 模型文件不完整!")
        print("="*60 + "\n")
        return False


def test_load_tokenizer():
    """测试加载分词器"""
    print("="*60)
    print("测试加载分词器")
    print("="*60 + "\n")
    
    try:
        from transformers import AutoTokenizer
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        model_path = current_dir / 'models' / 'bert-base-uncased'
        
        print(f"模型路径: {model_path}")
        print(f"路径类型: {type(model_path)}")
        print(f"绝对路径: {model_path.absolute()}")
        
        print(f"\n正在加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path.absolute()),
            local_files_only=True
        )
        
        print("✅ 分词器加载成功!")
        print(f"Vocab size: {len(tokenizer)}")
        
        # 测试分词
        test_text = "This is a test sentence."
        tokens = tokenizer.tokenize(test_text)
        print(f"\n测试分词: '{test_text}'")
        print(f"Tokens: {tokens}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_load_model():
    """测试加载模型"""
    print("="*60)
    print("测试加载 BERT 模型")
    print("="*60 + "\n")
    
    try:
        from transformers import AutoModel
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        model_path = current_dir / 'models' / 'bert-base-uncased'
        
        print(f"正在加载模型...")
        model = AutoModel.from_pretrained(
            str(model_path.absolute()),
            local_files_only=True
        )
        
        print("✅ 模型加载成功!")
        print(f"模型配置: {model.config}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("BERT 模型诊断工具")
    print("="*60 + "\n")
    
    # 检查文件结构
    structure_ok = check_model_structure()
    
    if structure_ok:
        # 测试加载分词器
        tokenizer_ok = test_load_tokenizer()
        
        # 测试加载模型
        model_ok = test_load_model()
        
        if tokenizer_ok and model_ok:
            print("="*60)
            print("✅ 所有检查都通过了！")
            print("="*60)
            print("你可以现在运行: python BERT_main.py\n")
        else:
            print("❌ 模型加载失败，请检查上面的错误信息\n")


if __name__ == "__main__":
    main()
