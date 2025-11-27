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
    """三分类情感分析RNN模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(SentimentRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, 
                         batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向所以*2
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
        
        # 使用双向GRU的最后隐藏状态
        # hidden形状: (num_layers * num_directions, batch_size, hidden_dim)
        hidden_forward = hidden[-2, :, :]  # 前向最后层
        hidden_backward = hidden[-1, :, :]  # 后向最后层
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # 全连接层
        output = self.fc(self.dropout(hidden_concat))
        
        return output


def train_model(model, train_loader, optimizer, criterion, epochs=10):
    """训练模型"""
    train_losses = []
    train_accuracies = []
    
    print("\n开始训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
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
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_accuracies'], label='训练准确率', color='green')
    ax2.set_title('训练准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_loader, processor):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_texts = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']
            
            outputs = model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
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
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    label_names = [processor.idx_to_label.get(i, str(i)) for i in range(len(processor.idx_to_label))]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算准确率
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    print(f"测试集准确率: {accuracy:.4f}%")
    
    # 各类别准确率
    class_accuracies = []
    num_classes = len(processor.idx_to_label)
    for i in range(num_classes):
        class_mask = np.array(all_labels) == i
        if np.sum(class_mask) > 0:
            class_acc = 100 * np.sum(np.array(all_predictions)[class_mask] == i) / np.sum(class_mask)
            class_accuracies.append(class_acc)
            label_name = processor.idx_to_label.get(i, str(i))
            print(f"{label_name}类别准确率: {class_acc:.4f}%")
    
    return all_predictions, all_labels, all_texts, all_probabilities

def predict_sentiment(model, processor, text):
    """预测单条文本情感"""
    model.eval()
    
    # 清洗文本
    cleaned_text = text.lower().strip()
    sequence = processor.text_to_sequence(cleaned_text)
    sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    sentiment = processor.idx_to_label[predicted_class]
    
    return sentiment, confidence, cleaned_text

def save_predictions(predictions, labels, texts, probabilities, processor, filename='predictions.csv'):
    """保存预测结果"""
    num_classes = len(processor.idx_to_label)
    prob_cols = {f'class_{processor.idx_to_label.get(i, str(i))}_prob': [prob[i] for prob in probabilities]
                 for i in range(num_classes)}
    
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': [processor.idx_to_label.get(label, str(label)) for label in labels],
        'predicted_label': [processor.idx_to_label.get(pred, str(pred)) for pred in predictions],
        'is_correct': [pred == true for pred, true in zip(predictions, labels)]
    })
    
    # 添加每个类别的概率
    for col_name, col_values in prob_cols.items():
        results_df[col_name] = col_values
    
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

