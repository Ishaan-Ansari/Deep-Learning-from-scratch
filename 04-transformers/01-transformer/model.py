import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        """
        d_model: Dimension of the model
        seq_len: Maximum length of the input sequences
        dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of of shape (max_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (Seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # multiply by denominator by log term for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sine function to even indices
        pe[:, 0::2] = torch.sin(position * div_term)   
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape (1, seq_len, d_model)
        self.register_buffer('pe', pe) # Not a model parameter, but should be saved in the state_dict

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)  # Add positional encoding to input embeddings
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6)-> None:   # we define eps as 0.000001 to avoid division by zero
        super().__init__()
        self.eps = eps
        # We define alpha as a trainable parameter and initialize it with ones
        self.alpha = nn.Parameter(torch.ones(1))    # One-dimensional tensor that will be used to scale the input data

        # We define bias as a trainable parameter and initialize it with zeros
        self.bias = nn.Parameter(torch.zeros(1)) # One-dimensional tensor that will be added to the input data


    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # Computing the mean of the input data. Keeping the number of dimensions unchanged
        std = x.std(-1, keepdim=True)   # Computing the standard deviation of the input data. Keep the number of dimensions unchanged

        # Returning the normalized input
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # First linear transformation
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 & b1
        self.dropout = nn.Dropout(dropout)      # Dropout to prevent overfitting
        # Second linear transformation
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply the first linear layer, followed by ReLU activation and dropout
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1)-> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model) # Final linear layer to combine heads
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1) # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        if dropout is not None:
            attention_output = dropout(attention_output)

        attention_output = attention_weights @ value # Shape: (batch_size, num_heads, seq_len, d_k)

        return attention_output, attention_weights

    def forward(self, query, key, value, mask=None):
        query = self.w_q(query) # Shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(key)
        value = self.w_v(value)

        # batch, seq_len, d_model -> batch, num_heads, seq_len, d_k -> batch, num_heads, seq_len, d_k
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) # Concatenate heads and put back in (batch_size, seq_len, d_model)
        x = self.w_o(x) # Final linear layer
        
        return x

# Building residual connection
class ResidualConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout to prevent overfitting
        self.norm = LayerNormalization()  # Normalization layer

    def forward(self, x, sublayer):
        # We normalize the input and add it to the original input 'x'. This creates the residual connection process.
        return x + self.dropout(sublayer(self.norm(x))) 
    

# Encoder block
class EncoderBlock(nn.Module):
    # This blocks takes the MultiHeadAttention, FeedForward, as well as the dropout rate for the residual connection
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 2 Residual Connections with dropout

    def forward(self, x, src_mask):
        # Applying the first residual connection around the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Three x's are query, key, value plus the source mask

        # Applying the second residual connection around the feed-forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x # Output tensor after applying self-attention and feed-forward layers with residual connections
    

# Building encoder 
class Encoder(nn.Module):
    def __init__(self,  layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()  # Final layer normalization

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)  # Applying each EncoderBlock to the input tensor 'x'

        return self.norm(x)  # Normalizing output
    
    