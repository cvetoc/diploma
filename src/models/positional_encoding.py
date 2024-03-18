import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, maxlen, emb_size, dropout=0.1):  # , maxlen=5000
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size
        # TODO: Реализуйте конструтор https://pytorch.org/tutorials/beginner/translation_transformer.html
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        # TODO: Реализуйте сложение эмбединнгов токенов с позиционными эмбеддингами
        token_embedding = token_embedding * math.sqrt(self.emb_size)
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# https://weiliu2k.github.io/CITS4012/transformer/position_encoding.html
# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len, d_model):
#         super(PositionalEncoding, self).__init__()
#         self.d_model = d_model
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * angular_speed) # even dimensions
#         pe[:, 1::2] = torch.cos(position * angular_speed) # odd dimensions
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         # x is N, L, D
#         # pe is 1, maxlen, D
#         scaled_x = x * np.sqrt(self.d_model)
#         encoded = scaled_x + self.pe[:, :x.size(1), :]
#         return encoded
