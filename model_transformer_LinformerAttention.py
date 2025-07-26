import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- Module 1: Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- Module 2: Linformer Attention Mechanism Model ---
class LinformerAttention(nn.Module):
    "Linformer Implementation"
    def __init__(self, d_model: int, nhead: int, seq_len:int, k: int, dropout: float = 0.1):
        # k (int): The projected dimension for keys and values (the Linformer's bottleneck).#
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.seq_len = seq_len
        self.k = k
        self.batch_first = True

        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # The core of Linformer: projection layers for K and V
        # These project the sequence length dimension (seq_len) down to k
        self.e_proj = nn.Linear(self.seq_len, self.k)
        self.f_proj = nn.Linear(self.seq_len, self.k)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
        torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        # 1. Project Q, K, V from input x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, nhead, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)

        # 3. Apply Linformer's projection on K and V
        # The projection is applied on the sequence length dimension.
        # k: (batch_size, nhead, seq_len, d_head) -> (batch_size * nhead, d_head, seq_len)
        k = k.permute(0, 1, 3, 2).reshape(batch_size * self.nhead, self.d_head, self.seq_len)
        # 将多头注意力的每一头“摊平”到一个维度上，一次性用矩阵操作并行完成所有头的计算，投影到较小的矩阵中。
        # k_proj: (batch_size * nhead, d_head, seq_len) -> (batch_size * nhead, d_head, k)
        k_proj = self.e_proj(k)

        v = v.permute(0, 1, 3, 2).reshape(batch_size * self.nhead, self.d_head, self.seq_len)
        v_proj = self.e_proj(v)

        # Reshape back for attention calculation
        # k_proj: (batch_size, nhead, d_head, k) -> (batch_size, nhead, k, d_head)
        k_proj = k_proj.view(batch_size, self.nhead, self.d_head, self.k).permute(0, 1, 3, 2)
        # v_proj: (batch_size, nhead, d_head, k) -> (batch_size, nhead, k, d_head)
        v_proj = v_proj.view(batch_size, self.nhead, self.d_head, self.k).permute(0, 1, 3, 2)

        # 4. Perform scaled dot-product attention
        # q: (batch, nhead, seq_len, d_head), k_proj.transpose: (batch, nhead, d_head, k)
        # scores: (batch, nhead, seq_len, k)
        # matmul 矩阵乘法
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 5. Apply attention to projected V and output: (batch, nhead, seq_len, d_head)
        output = torch.matmul(attn_probs, v_proj)

        # 6. concatenate（并列）heads and apply final linear layer
        # ->(batch, seq_len, d_model)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        # contiguous 保证张量在内存中是连续的，
        return output

# Moduel 3: Linformer Encoder Layer
class LinformerEncoderLayer(nn.Module):
    # A single Linformer encoder layer, combining LinformerAttention and a Feed-Forward network.
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 seq_len:int,
                 k: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = LinformerAttention(d_model, nhead, seq_len, k, dropout)
        # 两层前馈网络
        # 两种linear 映射相反
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # def forward(self, src:torch.Tensor) -> torch.Tensor:
    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, **kwargs) -> torch.Tensor:
        # Self-attentiono Block
        src2 = self.self_attn(src)
        # 残差连接 并 归一化
        src = src +self.dropout1(src2)
        src = self.norm1(src)

        # Feed Forward block
        # 前馈网络，先通过linear 映射到高维，使用ReLU激活，
        # 而后Dropout，再通过linear 2 映射回原始维度
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # 同样进行残差链接和归一化
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Module 4 Transformer with Lineformer
class LineformerTransformerModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 output_size: int,
                 seq_len:int,
                 k:int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 3. Transformer Encoder Stack
        # Use the custom LineformerEncoderLayer
        encoder_layer = LinformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            k = k,
            seq_len = seq_len,
            dropout=dropout,
        )
        # Stack defined layers on standard nn.TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        # Output Layer
        self.output_decoder = nn.Linear(d_model, output_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_decoder.weight.data.uniform_(-initrange, initrange)
        self.output_decoder.bias.data.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src (torch.Tensor): The source sequence is of shape (batch_size, seq_len, input_size).
        Return torch.Tensor: The prediction results, of shape (batch_size, output_size).
        """
        # Input embedding and scaling.
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        # Add positional encoding.
        src = self.pos_encoder(src)

        # Pass through the Transformer Encoder stack.
        output = self.transformer_encoder(src)

        # Withdraw output of the last time step.
        output = output[:, -1, :]

        # Final prediction via the output layer.
        prediction = self.output_decoder(output)

        return prediction
