from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler

# GPU 优化设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 自动选择最优 cuDNN 算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32 加速
    print(f"GPU 已启用: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("使用 CPU 训练")

class SentimentRNN(nn.Module):
    """四分类情感分析RNN模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(SentimentRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 词嵌入层，将索引映射为密集向量
        # 经过嵌入层: 每个索引变成embed_dim维向量
        # 输出维度:(batch_size, seq_len, embed_dim)

        
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout, bidirectional=True)
        # GRU层，设置batch_first=True表示输入张量的第一个维度是批次大小，
        # bidirectional=True表示双向GRU
        
        # n_layers=2表示堆叠2层GRU
        # 第二层GRU就是把第一层的输出(隐藏层)作为输入
        # 输入序列: x1  -> x2  -> x3  -> x4
        #           ↓      ↓      ↓      ↓
        # 第1层RNN: h1¹ -> h2¹ -> h3¹ -> h4¹
        #           ↓      ↓      ↓      ↓  
        # 第2层RNN: h1² -> h2² -> h3² -> h4²
        
        

        
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  
        # 全连接层，因为双向GRU，所以隐藏层维度是hidden_dim * 2

        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # x形状: (batch_size, sequence_length)
        
        # 嵌入层
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN层
        rnn_output, hidden = self.rnn(embedded)
        # rnn_output - 所有时间步的输出
        #   对于多层rnn: rnn_output只包含最后一层的输出
        # hidden - 最后时间步的隐藏状态
        
        # 使用双向GRU的最后隐藏状态
        # rnn_output形状: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden形状: (num_layers(RNN层数) * num_directions(方向数), batch_size, hidden_dim(隐藏状态维度))
        # hidden的4个维度分别对应:
        #   [0]: 第一层前向最后隐藏状态
        #   [1]: 第一层后向最后隐藏状态  
        #   [2]: 第二层前向最后隐藏状态
        #   [3]: 第二层后向最后隐藏状态
        hidden_forward = hidden[-2, :, :]   # 前向最后层
        hidden_backward = hidden[-1, :, :]  # 后向最后层
        # 取最后两个隐藏状态（因为双向，所以前向和后向各最后一个），拼接后作为全连接层的输入
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # 全连接层
        output = self.fc(self.dropout(hidden_concat))
        
        return output

def train_model(model, train_loader, optimizer, criterion, epochs=10):
    """训练模型"""
    train_losses = []      # 存储每个epoch的训练损失
    train_accuracies = []  # 存储每个epoch的训练准确率
    
    print("\n开始训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        # - 启用dropout
        # - 启用batch normalization
        """Batch Normalization是一种对神经网络中间层输出进行标准化处理的技术,
        它通过规范化每一层的输入, 使数据保持相对稳定的分布,
        从而加速训练过程并提高模型稳定性。"""
        
        # - 计算梯度


        train_loss = 0      # 累计损失
        train_correct = 0   # 正确预测的数量
        train_total = 0     # 总样本数量
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)  # 将序列数据移动到GPU/CPU
            labels = batch['label'].to(device)        # 将标签数据移动到GPU/CPU
            
            # 前向传播
            outputs = model(sequences)  # 调用模型的forward方法
            loss = criterion(outputs, labels)  # 计算损失
            
            # 反向传播
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 计算当前梯度
            
            # 梯度裁剪 (限制反向传播过程中梯度的大小，防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # clip_grad_norm_：裁剪梯度范数
            # model.parameters()：所有模型参数
            # max_norm=1.0：最大梯度范数

            optimizer.step()  # 更新参数
            
            # 统计
            train_loss += loss.item()  # 累加损失值
            # loss.item()：从张量中提取标量值

            _, predicted = torch.max(outputs.data, 1)
            # torch.max()：返回最大值和对应的索引
            # outputs.data：模型输出张量
            # dim=1：在类别维度上取最大值
            # _：最大值（不需要）
            # predicted：预测的类别索引
            
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total  # 准确率百分比
        avg_train_loss = train_loss / len(train_loader)     # 平均损失
        # len(train_loader)：批次数

        train_losses.append(avg_train_loss)      # 记录损失
        train_accuracies.append(train_accuracy)  # 记录准确率
        
        # 输出训练进度
        print(f'Epoch [{epoch+1:2d}/{epochs}] | '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='训练损失', color='blue')
    ax1.set_title('训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()    # 显示图例
    ax1.grid(True)  # 显示网格
    
    # 准确率曲线
    ax2.plot(history['train_accuracies'], label='训练准确率', color='green')
    ax2.set_title('训练准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, processor):
    """评估模型"""
    model.eval()  # 设置模型为评估模式：
    # - 禁用dropout
    # - 禁用batch normalization的统计更新
    # - 不计算梯度

    all_predictions = []     # 存储所有预测结果
    all_labels = []          # 存储所有真实标签
    all_texts = []           # 存储所有原始文本
    all_probabilities = []   # 存储所有预测概率
    
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']  # 原始文本，不需要移动到设备
            
            outputs = model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            # torch.softmax()：将输出转换为概率分布
            # dim=1：在类别维度上计算softmax

            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            
            all_predictions.extend(predicted.cpu().numpy())
            # .cpu()：将张量移动到CPU
            # .numpy()：转换为numpy数组
            # .extend()：将列表元素逐个添加到目标列表

            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(texts)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    
    
    # 分类报告
    print("\n=== 测试集评估结果 ===")
    print("\n详细分类报告:")
    
    target_names = [processor.idx_to_label.get(i, str(i)) for i in range(len(processor.idx_to_label))]
    
    print(classification_report(all_labels, all_predictions, 
                              target_names=target_names,
                              digits=4))
    # classification_report：生成分类报告，包括精确率、召回率、F1分数等
    # all_labels：真实标签
    # all_predictions：预测标签
    # target_names：类别名称
    # digits=4：保留4位小数

    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    # confusion_matrix：计算混淆矩阵
    label_names = [processor.idx_to_label.get(i, str(i)) for i in range(len(processor.idx_to_label))]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    # sns.heatmap：绘制热力图
    # cm：混淆矩阵数据
    # annot=True：在单元格中显示数值
    # fmt='d'：数值格式为整数
    # cmap='Blues'：颜色映射为蓝色系
    # xticklabels, yticklabels：坐标轴标签

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    # 保存图片，dpi=300（分辨率），bbox_inches='tight'（紧凑布局）
    plt.show()
    
    # 计算准确率
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    # np.array(): 将列表转换为numpy数组
    # np.array(all_predictions) == np.array(all_labels): 逐元素比较，返回布尔数组
    # np.sum(): 计算True的数量（正确预测的数量）
    # len(all_labels): 总样本数
    print(f"测试集准确率: {accuracy:.4f}%")
    
    # 各类别准确率
    class_accuracies = []  # 存储每个类别的准确率
    num_classes = len(processor.idx_to_label)  # processor.idx_to_label 是类别索引到语义标签的映射字典
    for i in range(num_classes):
        class_mask = np.array(all_labels) == i  # 创建布尔掩码，筛选出所有真实标签为类别 i 的样本
        if np.sum(class_mask) > 0:  # 确保当前类别有样本
            
            # 计算类别准确率
            class_acc = 100 * np.sum(np.array(all_predictions)[class_mask] == i) / np.sum(class_mask)
            # np.array(all_predictions)[class_mask]: 只选择当前类别的预测结果
            # == i: 检查预测是否正确
            # np.sum(): 计算正确预测的数量
            # np.sum(class_mask): 当前类别的总样本数
            
            class_accuracies.append(class_acc)
            label_name = processor.idx_to_label.get(i, str(i))
            # get(i, str(i))：获取类别名称，若不存在则返回字符串形式的索引

            print(f"{label_name}类别准确率: {class_acc:.4f}%")
    
    return all_predictions, all_labels, all_texts, all_probabilities

def predict_sentiment(model, processor, text):
    """预测单条文本情感"""
    model.eval()  # 设置模型为评估模式
    
    # 清洗文本
    cleaned_text = text.lower().strip()
    sequence = processor.text_to_sequence(cleaned_text)  # 将文本转换为数字序列
    sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    # torch.tensor(sequence): 将列表转换为PyTorch张量
    # dtype=torch.long: 指定数据类型为长整型
    # .unsqueeze(0): 在0维度增加一维，从[length]变为[1, length]
    #   因为模型期望批处理输入，即使只有1个样本也要有批次维度
    # .to(device): 移动到GPU或CPU
    
    with torch.no_grad():
        output = model(sequence_tensor)
        
        probabilities = torch.softmax(output, dim=1)
        # torch.softmax(): 将输出转换为概率分布
        # dim=1: 在类别维度上计算softmax
        
        predicted_class = torch.argmax(output, dim=1).item()
        # torch.argmax(): 返回最大值的索引
        # dim=1: 在类别维度上寻找最大值
        # .item(): 从单元素张量中提取Python标量

        confidence = probabilities[0][predicted_class].item()
        # probabilities[0]: 第一个样本的概率分布（因为批次大小为1）
        # [predicted_class]: 预测类别的概率
        # .item(): 提取标量值
    
    sentiment = processor.idx_to_label[predicted_class]  # 获取情感标签名称
    
    return sentiment, confidence, cleaned_text

def save_predictions(predictions, labels, texts, probabilities, processor, filename='predictions.csv'):
    """保存预测结果"""
    num_classes = len(processor.idx_to_label)  # 获取类别数量

    # 创建每个类别的概率列
    prob_cols = {f'class_{processor.idx_to_label.get(i, str(i))}_prob': 
                 [prob[i] for prob in probabilities] for i in range(num_classes)}
    # 字典推导式：为每个类别创建一列概率值
    # f'class_{...}_prob': 列名，如'class_Positive_prob'
    # [prob[i] for prob in probabilities]: 列表推导式，提取每个样本对当前类别的概率
    
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': [processor.idx_to_label.get(label, str(label)) for label in labels],
        'predicted_label': [processor.idx_to_label.get(pred, str(pred)) for pred in predictions],
        'is_correct': [pred == true for pred, true in zip(predictions, labels)]
        # zip(predictions, labels): 将两个列表配对
    })
    
    # 添加每个类别的概率
    for col_name, col_values in prob_cols.items():  # 遍历概率列字典
        results_df[col_name] = col_values  # 添加新列
    
    results_df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n预测结果已保存到: {filename}")
    return results_df

def load_trained_model(model_path):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    processor = checkpoint['processor']
    model_params = checkpoint['model_params']
    
    model = SentimentRNN(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, processor

def demo_trained_model():
    """演示使用训练好的模型"""
    print("=== 使用训练好的模型进行预测 ===")
    
    # 加载模型
    try:
        model, processor = load_trained_model('sentiment_model.pth')
        print("模型加载成功!")
    except FileNotFoundError:
        print("模型文件不存在，请先训练模型")
        return
    
    # 批量预测
    new_texts = [
        "This movie is fantastic and I really enjoyed it!",
        "The service was terrible and the quality is poor.",
        "It's an average product, nothing special.",
        "Amazing quality and excellent customer service!",
        "Very disappointed with this purchase.",
        "It's okay, I guess. Not what I expected but usable.",
        "Absolutely love it! Best purchase ever!",
        "Worst experience of my life, never again."
    ]
    
    print("\n批量预测结果:")
    results = []
    for text in new_texts:
        sentiment, confidence, cleaned_text = predict_sentiment(model, processor, text)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': f"{confidence:.2%}"
        })
        print(f"『{text}』")
        print(f"  → 情感: {sentiment} (置信度: {confidence:.2%})")
        print()
    
    # 保存演示结果
    demo_df = pd.DataFrame(results)
    demo_df.to_csv('demo_predictions.csv', index=False, encoding='utf-8')
    print("演示预测结果已保存到: demo_predictions.csv")