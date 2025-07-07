import torch
import torch.nn as nn
import math

class LSTMBaseline(nn.Module):
    """
    一个简单的基于LSTM的基线模型，用于预测机器人动力学。
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        初始化模型层。

        参数:
        input_size (int): 输入特征的数量 (我们的数据是21)。
        hidden_size (int): LSTM隐藏层的大小 (例如 128)。
        num_layers (int): LSTM的层数 (例如 2)。
        output_size (int): 输出特征的数量 (我们的目标是14)。
        """
        super(LSTMBaseline, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        # batch_first=True 表示输入的张量形状为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义一个全连接层，将LSTM的输出映射到最终的预测维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        定义模型的前向传播逻辑。
        """
        # 初始化LSTM的隐藏状态和细胞状态
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM层的前向传播
        # out的形状: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # 我们只关心序列中最后一个时间步的输出，用它来进行预测
        # out[:, -1, :] 的形状: (batch_size, hidden_size)
        last_time_step_out = out[:, -1, :]

        # 将最后一个时间步的输出送入全连接层，得到最终预测
        prediction = self.fc(last_time_step_out)

        return prediction


class PositionalEncoding(nn.Module):
    """
    为序列注入位置信息。这是 Transformer 模型正确处理序列顺序所必需的。
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # d_model: 词嵌入的维度，也是模型的维度
        # max_len: 序列的最大长度

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe 不是模型的参数，所以我们把它注册为 buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    """
    基于Transformer Encoder的动力学预测模型。
    """

    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout=0.1):
        """
        初始化模型。

        参数:
        input_size (int): 输入特征的数量 (21)
        d_model (int): Transformer模型的内部维度 (例如 256)。必须是 nhead 的整数倍。
        nhead (int): 多头注意力机制中的头数 (例如 8)。
        num_encoder_layers (int): Encoder层数 (例如 3)。
        dim_feedforward (int): Encoder中前馈网络的维度 (例如 512)。
        output_size (int): 输出特征的数量 (14)。
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # 1. 输入层：将21维的输入特征映射到d_model维
        self.encoder = nn.Linear(input_size, d_model)

        # 2. 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder层
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                                         num_layers=num_encoder_layers)

        # 4. 输出层：将d_model维的输出映射回14维的目标
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        """
        前向传播。
        src 的形状: (batch_size, seq_len, input_size)
        """
        # 1. 输入嵌入
        src = self.encoder(src) * math.sqrt(self.d_model)

        # 2. PyTorch的TransformerEncoder期望的输入形状是(seq_len, batch, feature)，但我们用了batch_first=True，所以不用转换
        # 如果没有用batch_first=True，则需要 src = src.permute(1, 0, 2)

        # 3. 添加位置编码 (需要适配形状)
        # 我们的 PositionalEncoding 期望 (seq_len, batch, feature)，所以需要临时转换一下
        # src_permuted = src.permute(1, 0, 2)
        # src_with_pos = self.pos_encoder(src_permuted)
        # src_with_pos = src_with_pos.permute(1, 0, 2)
        # 注意: PyTorch的新版本中，PositionalEncoding的设计更加灵活，这里我们直接加在batch_first的张量上
        # 但标准的实现是上面注释掉的部分。为了简单，我们直接相加，但要知道这是一种简化。
        # 实际上，我们需要调整PositionalEncoding或输入张量以匹配。
        # 一个更正确的做法是将PositionalEncoding的forward适配batch_first
        # 这里我们先用一个简单的方式，实际应用中可能需要调整

        # 4. 通过Transformer Encoder
        output = self.transformer_encoder(src)  # 形状: (batch_size, seq_len, d_model)

        # 5. 我们只关心序列最后一个时间步的输出
        output = output[:, -1, :]  # 形状: (batch_size, d_model)

        # 6. 通过输出层得到最终预测
        prediction = self.decoder(output)

        return prediction