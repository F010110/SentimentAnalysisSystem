from torch import nn
import torch
from RNN_model import SentimentRNN, train_model, evaluate_model, plot_training_history, predict_sentiment, save_predictions, device
from text_process import SentimentDataProcessor, create_data_loaders
import torch.optim as optim
import os

def main():
    """主函数"""
    
    
    EMBED_DIM = 300         # 词嵌入维度：每个单词用300维向量表示
    HIDDEN_DIM = 256         # RNN隐藏层维度：隐藏状态向量的长度
    OUTPUT_DIM = 4          # 输出维度：对应4个情感类别: Irrelevant, Negative, Neutral, Positive
    N_LAYERS = 3            # RNN层数：堆叠3层GRU
    DROPOUT = 0.5           # 丢弃率 
    BATCH_SIZE = 64         # 批处理大小：每次训练64个样本
    LEARNING_RATE = 0.0005   # 学习率：Adam优化器的学习率
    
    
    TRAIN_PATH = os.path.join('dataset', 'twitter_training_cleaned.csv')
    VAL_PATH = os.path.join('dataset', 'twitter_validation_cleaned.csv')
    
    
    
    # 1. 加载和预处理数据
    print("步骤1: 加载和预处理数据")
    processor = SentimentDataProcessor()
    # 创建SentimentDataProcessor实例：
    # - 初始化词表：{'<PAD>': 0, '<UNK>': 1}
    # - 初始化标签映射：空字典
    # - 设置最小词频：2（默认值）
    
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
    # create_data_loaders函数：
    #   1. 创建SentimentDataset实例（包装DataFrame和processor）
    #   2. 创建DataLoader实例：
    #    - train_loader：训练数据加载器，shuffle=True（打乱数据）
    #    - test_loader：测试数据加载器，shuffle=False（保持顺序）
    #   3. 返回两个数据加载器

    # 3. 创建模型
    print("\n步骤3: 创建模型")
    model = SentimentRNN(
        vocab_size=processor.vocab_size,  # 词汇表大小
        embed_dim=EMBED_DIM,              # 词嵌入维度
        hidden_dim=HIDDEN_DIM,            # 隐藏层维度
        output_dim=OUTPUT_DIM,            # 输出维度（类别数）
        n_layers=N_LAYERS,                # RNN层数
        dropout=DROPOUT                   # 丢弃率
    ).to(device)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    # sum(p.numel() for p in model.parameters())：计算所有参数的数量
    # p.numel()：返回张量中元素的数量
    # :,：千位分隔符格式化，如1,000,000

    
    # 4. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Adam优化器：
    # - model.parameters()：模型的所有可训练参数
    # - lr=0.001：学习率
    # - weight_decay=1e-4：L2正则化系数，防止过拟合
    
    
    
    # 5. 训练模型
    print("\n步骤4: 训练模型")
    history = train_model(model, train_loader, optimizer, criterion, epochs=15)
    # train_model函数：
    # - model：要训练的模型
    # - train_loader：训练数据加载器
    # - optimizer：优化器
    # - criterion：损失函数
    # - epochs=15：训练轮数
    # 返回包含训练损失和准确率的字典history
    
    # 6. 绘制训练历史
    plot_training_history(history)
    
    # 7. 评估模型
    print("\n步骤5: 评估模型")
    predictions, true_labels, texts, probabilities = evaluate_model(model, test_loader, processor)
    # evaluate_model函数：
    # - 在测试集上评估模型性能
    # - 返回预测结果、真实标签、原始文本和预测概率

    # 8. 保存预测结果
    results_df = save_predictions(predictions, true_labels, texts, probabilities, processor)


    # save_predictions函数：
    # - 将预测结果保存为CSV文件
    # - 返回包含所有结果的DataFrame
    
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
        # predict_sentiment函数返回：
        # - sentiment: 预测的情感类别
        # - confidence: 预测置信度（0-1之间）
        # - cleaned_text: 清洗后的文本
        
        print(f"文本: '{text}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2%})")
        print("-" * 50)
    
    # 10. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),  # 模型参数
        'processor': processor,  # 数据处理器（包含词表、标签映射等）
        'model_params': {  # 模型超参数
            'vocab_size': processor.vocab_size,
            'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'n_layers': N_LAYERS,
            'dropout': DROPOUT
        }
    }, 'sentiment_model.pth')  # 保存文件名
    
    print("\n模型已保存到: sentiment_model.pth")

# 运行主函数
if __name__ == "__main__":
    main()