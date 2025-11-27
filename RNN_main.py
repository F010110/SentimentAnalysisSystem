from torch import nn
import torch
from RNN_model import SentimentRNN, train_model, evaluate_model, plot_training_history, predict_sentiment, save_predictions, device
from text_process import SentimentDataProcessor, create_data_loaders
import torch.optim as optim
import os

def main():
    """主函数"""
    
    
    EMBED_DIM = 128
    HIDDEN_DIM = 64
    OUTPUT_DIM = 4  # Irrelevant, Negative, Neutral, Positive
    N_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    
    TRAIN_PATH = os.path.join('dataset', 'twitter_training_cleaned.csv')
    VAL_PATH = os.path.join('dataset', 'twitter_validation_cleaned.csv')
    
    
    
    
    
    
    
    
    
    # 1. 加载和预处理数据
    print("步骤1: 加载和预处理数据")
    processor = SentimentDataProcessor()
    
    try:
        train_df, test_df = processor.load_data(TRAIN_PATH, VAL_PATH)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保数据集路径正确:")
        print(f"训练集: {TRAIN_PATH}")
        print(f"测试集: {VAL_PATH}")
        return
    
    # 2. 创建数据加载器
    print("\n步骤2: 创建数据加载器")
    train_loader, test_loader = create_data_loaders(train_df, test_df, processor, BATCH_SIZE)
    
    # 3. 创建模型
    print("\n步骤3: 创建模型")
    model = SentimentRNN(
        vocab_size=processor.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 5. 训练模型
    print("\n步骤4: 训练模型")
    history = train_model(model, train_loader, optimizer, criterion, epochs=15)
    
    # 6. 绘制训练历史
    plot_training_history(history)
    
    # 7. 评估模型
    print("\n步骤5: 评估模型")
    predictions, true_labels, texts, probabilities = evaluate_model(model, test_loader, processor)
    
    # 8. 保存预测结果
    results_df = save_predictions(predictions, true_labels, texts, probabilities, processor)
    
    # 9. 预测示例
    print("\n步骤6: 预测示例")
    test_texts = [
        "This product is absolutely wonderful and amazing!",
        "I hate this thing it's terrible",
        "It's okay, not great but not bad either",
        "The quality is good but the price is too high",
        "Worst purchase ever, complete waste of money"
    ]
    
    print("\n预测示例:")
    for text in test_texts:
        sentiment, confidence, cleaned_text = predict_sentiment(model, processor, text)
        print(f"文本: '{text}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2%})")
        print("-" * 50)
    
    # 10. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor': processor,
        'model_params': {
            'vocab_size': processor.vocab_size,
            'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT
        }
    }, 'sentiment_model.pth')
    
    print("\n模型已保存到: sentiment_model.pth")

# 运行主函数
if __name__ == "__main__":
    main()