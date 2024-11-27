import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

'''
    base transformer encoderdecoder architecture
'''
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    # src source vector
    # mask don't let this word see behind words
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decoder(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encoder(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decoder(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# x means a vector [batchsize, seq_len, features]
# 层标准化
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    # features always means the feature length
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # Parameter 多维矩阵
        self.a_2 = nn.Parameter(torch.ones(features)) # 初始化为1
        self.b_2 = nn.Parameter(torch.zeros(features)) # 初始化为0
        self.eps = eps

    def forward(self, x):
        "对于最后一个维度进行求平均和"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 归一化(x - mean) 均值为0， (std + self.eps)：对标准差加上 eps，避免标准差为 0 的情况下出现除零错误。
        # x - mean / std：将输入除以标准差，使其标准差为 1
        # self.a_2：对归一化后的值进行缩放（乘以 γ）
        # self.b_2：对归一化后的值进行平移（加上 β）
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 子层之间的连接 处理单个Sublayer的输出，该输出将继续被输入下一个Sublayer
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout) # 随机丢弃层

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    "N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)




class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, mask, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, mask, src_mask, tgt_mask)
            return self.norm(x)

"""
单层decoder与单层encoder相比，decoder还有第三个子层，该层对encoder的输出执行attention：
即encoder-decoder-attention层，q向量来自decoder上一层的输出，k和v向量是encoder最后层的输出向量。
与encoder类似，我们在每个子层再采用残差连接，然后进行层标准化。
"""
class DecoderLayer(nn.Module):
    "Decoder is made up of self-attn, src-attn and feed forward."
    # self_attn 自注意力机制
    # src_attn 源注意力机制模块，将解码器和编码器输入结合
    # feed_forward 前馈神经网络
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # src 源序列的注意力掩码
    # tgt 目标序列的注意力掩码
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # 匿名函数
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # triangle upper 对角线上元素及上方元素设置为1，其他设置为0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 将矩阵转换为张量返回
    return torch.from_numpy(subsequent_mask) == 0

# attention 矩阵 q k v 矩阵
def attention(query, key, value, mask=None, dropout=None):
    # embedding length
    d_k = query.size(-1)
    # key.transpose(-2, -1) 对于-2 -1维度进行转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # 0 的位置设置为-1e9
    # vector A
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(dropout)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "mask 矩阵"
        if mask is not None:
            mask = mask.unsqueeze(1) # 在第二个维度增加一个维度，用于应用所有的注意力头
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # batch_size n h d_k 通过l线性层进行变换 - batch_size h n d_k
        # 此时三个矩阵形状都是(batch_size h n d_k) 然后最终合并为 batch_size, n, h * dk
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 将展平后的张量通过最后一个线性层
        return self.linears[-1](x)
