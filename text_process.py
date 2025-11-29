from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SentimentDataProcessor:
    """简单的数据处理器：构建词表、文本->序列、加载数据等"""
    def __init__(self, min_freq=2):
        self.min_freq = min_freq  # 最小词频，出现次数少于min_freq的单词将被忽略
        self.max_length = 0       # 最大序列长度，初始为0
        '''特殊标记：
        - `<PAD>`:表示填充(padding)。当处理批次数据时,句子长度可能不同,
        所以我们用`<PAD>`来填充较短的句子，使所有句子长度一致。
        - `<UNK>`:表示未知单词(unknown)。当遇到训练数据中未出现的单词时,用`<UNK>`代替。'''
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}  # 单词到索引的映射，初始化包含特殊标记
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}  # 索引到单词的映射
        self.vocab_size = len(self.word_to_idx)  # 词汇表大小，初始为2
        self.label_to_idx = {}    # 标签到索引的映射（用于将字符串标签转换为数字）
        self.idx_to_label = {}    # 索引到标签的映射

    def _build_vocab(self, texts):
        """构建词汇表"""
        word_freq = Counter()  # 计数器，用于统计词频

        for text in texts:
            text = str(text)  # 将文本转换为字符串，确保兼容性
            words = text.split()  # 按空格分割成单词列表
            word_freq.update(words)  # 更新词频计数
            self.max_length = max(self.max_length, len(words))  # 更新最大序列长度

        idx = max(self.word_to_idx.values()) + 1  # 当前最大索引+1
        for word, freq in word_freq.items():
            if freq >= self.min_freq and word not in self.word_to_idx:
                # 如果词频达到阈值且不在词汇表中，则添加
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

        self.vocab_size = len(self.word_to_idx)  # 更新词汇表大小
        print(f"\n词汇表统计:")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"最大序列长度: {self.max_length}")
        print(f"前10个词汇: {list(self.word_to_idx.items())[:10]}")

    def text_to_sequence(self, text, max_length=None):
        """文本转序列"""
        if max_length is None:
            max_length = self.max_length  # 如果未指定，使用最大序列长度

        words = str(text).split()  # 分割文本为单词列表
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        # 将每个单词转换为索引，如果单词不在词汇表中，使用<UNK>的索引

        # 填充或截断序列
        if len(sequence) < max_length:
            sequence = sequence + [self.word_to_idx['<PAD>']] * (max_length - len(sequence))
            # 如果序列长度不足，用<PAD>填充
        else:
            sequence = sequence[:max_length]  # 如果序列过长，截断
            

        return sequence

    def load_data(self, train_path, val_path):
        """加载 CSV 数据；返回 (train_df, val_df)。
        要求 CSV 中有列 'processed_text' 和 'attitude' 。
        """
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # 兼容列名
        for df in (train_df, val_df):
            if 'processed_text' not in df.columns and 'text' in df.columns:
                df['processed_text'] = df['text'].astype(str).str.lower().str.strip()
            elif 'processed_text' in df.columns:
                df['processed_text'] = df['processed_text'].astype(str).str.lower().str.strip()
            else:
                # 如果没有文本列，强制转换所有列为字符串并拼接
                df['processed_text'] = df.astype(str).agg(' '.join, axis=1).str.lower().str.strip()

        # 构建标签映射（如果标签不是数字）
        unique_labels = pd.concat([train_df['attitude'], val_df['attitude']]).unique()
        print(f"DEBUG: Unique labels found: {sorted(unique_labels)}")
        print(f"DEBUG: Number of unique labels: {len(unique_labels)}")
        
        if not np.issubdtype(type(unique_labels[0]), np.number):
             # 如果标签不是数字类型
            self.label_to_idx = {lab: i for i, lab in enumerate(sorted(unique_labels))}
            self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
            # 字典推导式：创建标签到索引的映射
            # enumerate(sorted(unique_labels)): 对排序后的唯一标签进行枚举
            print(f"DEBUG: Label mapping: {self.label_to_idx}")
            train_df['attitude'] = train_df['attitude'].map(self.label_to_idx)
            val_df['attitude'] = val_df['attitude'].map(self.label_to_idx)
            # .map(): 根据映射字典替换值
        else:
            # 假定已经是整数索引，确保它们从0开始且连续
            unique_ints = sorted({int(i) for i in unique_labels})
            print(f"DEBUG: Unique integer labels: {unique_ints}")
            
            # 如果标签不是从0开始的连续整数，需要重新映射
            if unique_ints != list(range(len(unique_ints))):
                print(f"WARNING: Labels are not 0-indexed. Remapping from {unique_ints} to {list(range(len(unique_ints)))}")
                remap = {old: new for new, old in enumerate(unique_ints)}
                # 创建重新映射字典
                train_df['attitude'] = train_df['attitude'].map(remap)
                val_df['attitude'] = val_df['attitude'].map(remap)
                self.label_to_idx = {new: str(old) for old, new in remap.items()}
                self.idx_to_label = {new: str(old) for old, new in remap.items()}
            else:
                self.idx_to_label = {int(i): str(i) for i in unique_ints}

        # 构建词表（只用训练集）
        self._build_vocab(train_df['processed_text'].astype(str).tolist())

        return train_df, val_df


class SentimentDataset(Dataset):
    """情感分析数据集"""
    def __init__(self, dataframe, processor, max_length=None):
        self.dataframe = dataframe  # 存储数据的DataFrame
        self.processor = processor  # 数据处理器实例
        self.max_length = max_length or processor.max_length  # 序列最大长度

    def __len__(self):
        return len(self.dataframe)
        # 返回数据集大小，PyTorch Dataset必须实现的方法

    def __getitem__(self, idx):
        # 获取单个样本，PyTorch Dataset必须实现的方法
        
        row = self.dataframe.iloc[idx]
        # .iloc[idx]: 按整数位置获取行数据
        
        text = row.get('processed_text', '')
        # 获取文本，如果列不存在则返回空字符串
        
        label = int(row['attitude'])
        # 获取标签并转换为整数

        sequence = self.processor.text_to_sequence(text, self.max_length)
        # 将文本转换为数字序列

        return {  # 返回字典格式的数据
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_data_loaders(train_df, val_df, processor, batch_size=32):
    """创建数据加载器"""
    train_dataset = SentimentDataset(train_df, processor)
    # 创建训练数据集实例
    val_dataset = SentimentDataset(val_df, processor)
    # 创建验证数据集实例

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # DataLoader参数：
    # - dataset: 数据集实例
    # - batch_size: 批处理大小
    # - shuffle=True: 打乱数据顺序（训练集）
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # shuffle=False: 验证集不打乱顺序，便于结果分析

    return train_loader, val_loader