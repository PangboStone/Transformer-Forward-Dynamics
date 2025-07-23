import torch
import torch.nn as nn
import math


# --- (Positional Encoding) ---
class PositionalEncoding(nn.Module):
    """
    Injects position information into the sequence
    Adapts the input format of batch_first=True.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        参数:
        d_model (int): model dimension
        dropout (float): Dropout ratio
        max_len (int): Maximum sequence length of pre-calculated position codes
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe 的形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to pe so that its shape is (1, max_len, d_model)
        # Add to the input of (batch_size, seq_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
        返回:
            torch.Tensor: 添加了位置编码的输出张量
        """
        # x.size(1) 是序列长度 (seq_len)
        # self.pe[:, :x.size(1)] 会取出所需长度的位置编码
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- 模块 2: Transformer 模型 ---
class TransformerModel(nn.Module):

    def __init__(self, input_size: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, output_size: int, dropout: float = 0.1):
        """
        初始化模型。
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        #  (Input Embedding)
        # Linear mapping of 21-dimensional input features to the d_model dimension
        self.input_embedding = nn.Linear(input_size, d_model)

        # Positional Encoding Layer
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        #  Transformer Encoder Layer
        #  Use batch_first=True to make the shape of the input and output more intuitive
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output Layer)
        # Mapping the output of the d_model dimension back to the 14-dimensional target
        self.output_decoder = nn.Linear(d_model, output_size)

        # Initialisation weights
        self.init_weights()

    def init_weights(self) -> None:

        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_decoder.weight.data.uniform_(-initrange, initrange)
        self.output_decoder.bias.data.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        参数:
            src (torch.Tensor): The source sequence, in the shape of (batch_size, seq_len, input_size)
        返回:
            torch.Tensor: The prediction results in the shape of (batch_size, output_size)
        """
        # 1. Input embedding and scaling
        # src : (batch_size, seq_len, input_size) -> (batch_size, seq_len, d_model)
        src = self.input_embedding(src) * math.sqrt(self.d_model)

        # 2. Add positional encoding
        # src : (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)

        # 3. Transformer Encoder
        # output : (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src)

        # 4. Focus on the output of the last time step of the sequence and use it to make a prediction
        # output : (batch_size, seq_len, d_model) -> (batch_size, d_model)
        output = output[:, -1, :]

        # 5. The final prediction is obtained through the output layer
        # prediction 形状: (batch_size, d_model) -> (batch_size, output_size)
        prediction = self.output_decoder(output)

        return prediction


