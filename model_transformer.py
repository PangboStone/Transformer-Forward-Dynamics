import torch
import torch.nn as nn
import math


# --- Module 1: Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input sequence embeddings.
    This implementation is designed for inputs with batch_first=True.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimension of the model (embedding size).
            dropout (float): The dropout probability.
            max_len (int): The maximum pre-computed sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a position tensor of shape (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions:  1 / (10000^(2i / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Initialize the positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Apply sine to even indices in the d_model dimension
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the d_model dimension
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to "pe" so its shape is (1, max_len, d_model),
        # which is corresponded to the input tensor of shape (batch_size, seq_len, d_model).
        # 'register_buffer' makes 'pe' a part of the model's state, but not a parameter to be trained.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PositionalEncoding.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: The output tensor with positional information added.
        """
        # x.size(1) is the sequence length (seq_len) of the current batch,
        # slice the pre-computed 'pe' to match the input's sequence length.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- Module 2: Transformer Model ---
class TransformerModel(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, output_size: int, dropout: float = 0.1):
        """
        Args:
            input_size (int): The number of input features.
            d_model (int): The dimension of the model (must be divisible by nhead).
            nhead (int): The number of heads in the multi-head attention mechanism.
            num_encoder_layers (int): The number of stacked encoder layers.
            dim_feedforward (int): The dimension of the feedforward network model.
            output_size (int): The number of output features (e.g., number of classes).
            dropout (float): The dropout probability.
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        # 1. Input Embedding Layer
        # Linearly maps the input features of 'input_size' to the model's dimension 'd_model'.
        self.input_embedding = nn.Linear(input_size, d_model)
        # 2. Positional Encoding Layer
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # 3. Transformer Encoder Stack
        # Define a single encoder layer. batch_first=True makes the input/output shape more intuitive.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Crucial for (batch, seq, feature) format
        )
        # Stack multiple encoder layers.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        # 4. Output Layer
        # Maps the final d_model dimensional representation to the desired output_size.
        self.output_decoder = nn.Linear(d_model, output_size)
        # Initialize weights for better training stability.
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_decoder.weight.data.uniform_(-initrange, initrange)
        self.output_decoder.bias.data.zero_()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        src (torch.Tensor): The source sequence is of shape (batch_size, seq_len, input_size).
        Return torch.Tensor: The prediction results, of shape (batch_size, output_size).
        """
        # Step 1: Input embedding and scaling.
        # convert src shape: (batch_size, seq_len, input_size) -> (batch_size, seq_len, d_model)
        # Scaling by sqrt(d_model) is a common practice to balance embedding and positional encoding magnitudes.
        src = self.input_embedding(src) * math.sqrt(self.d_model)

        # Step 2: Add positional encoding.
        # src shape remains: (batch_size, seq_len, d_model)
        src = self.pos_encoder(src)

        # Step 3: Pass through the Transformer Encoder stack.
        # output shape: (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src)

        # Step 4: Withdraw output of the last time step.
        # This vector is considered the final representation of the entire sequence.
        # output shape: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        output = output[:, -1, :]

        # Step 5: Final prediction via the output layer.
        # prediction shape: (batch_size, d_model) -> (batch_size, output_size)
        prediction = self.output_decoder(output)

        return prediction
