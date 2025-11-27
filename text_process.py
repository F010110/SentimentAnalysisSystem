from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SentimentDataProcessor:
    """简单的数据处理器：构建词表、文本->序列、加载数据等"""
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.max_length = 0
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = len(self.word_to_idx)
        self.label_to_idx = {}
        self.idx_to_label = {}

    def _build_vocab(self, texts):
        """构建词汇表"""
        word_freq = Counter()

        for text in texts:
            text = str(text)
            words = text.split()
            word_freq.update(words)
            self.max_length = max(self.max_length, len(words))

        idx = max(self.word_to_idx.values()) + 1
        for word, freq in word_freq.items():
            if freq >= self.min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1

        self.vocab_size = len(self.word_to_idx)
        print(f"\n词汇表统计:")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"最大序列长度: {self.max_length}")
        print(f"前10个词汇: {list(self.word_to_idx.items())[:10]}")

    def text_to_sequence(self, text, max_length=None):
        """文本转序列"""
        if max_length is None:
            max_length = self.max_length

        words = str(text).split()
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]

        if len(sequence) < max_length:
            sequence = sequence + [self.word_to_idx['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]

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
            self.label_to_idx = {lab: i for i, lab in enumerate(sorted(unique_labels))}
            self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}
            print(f"DEBUG: Label mapping: {self.label_to_idx}")
            train_df['attitude'] = train_df['attitude'].map(self.label_to_idx)
            val_df['attitude'] = val_df['attitude'].map(self.label_to_idx)
        else:
            # 假定已经是整数索引，确保它们从0开始且连续
            unique_ints = sorted({int(i) for i in unique_labels})
            print(f"DEBUG: Unique integer labels: {unique_ints}")
            
            # 如果标签不是从0开始的连续整数，需要重新映射
            if unique_ints != list(range(len(unique_ints))):
                print(f"WARNING: Labels are not 0-indexed. Remapping from {unique_ints} to {list(range(len(unique_ints)))}")
                remap = {old: new for new, old in enumerate(unique_ints)}
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
        self.dataframe = dataframe
        self.processor = processor
        self.max_length = max_length or processor.max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row.get('processed_text', '')
        label = int(row['attitude'])

        sequence = self.processor.text_to_sequence(text, self.max_length)

        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_data_loaders(train_df, val_df, processor, batch_size=32):
    """创建数据加载器"""
    train_dataset = SentimentDataset(train_df, processor)
    val_dataset = SentimentDataset(val_df, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader