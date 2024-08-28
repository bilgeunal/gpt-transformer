import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        """
        Initialize the Scaled Dot-Product Attention module.
        
        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Define linear layers for query, key, and value transformations
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x, mask=None):
        """
        Perform the forward pass of scaled dot-product attention.
        
        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, embed_dim).
            mask (Tensor, optional): The mask tensor to apply during attention calculation. Shape (batch_size, seq_len, seq_len).
        
        Returns:
            Tensor: The output tensor after applying attention. Shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()

        # Apply linear transformations to get Q, K, V
        Q = self.query_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for multi-head attention: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate attention heads and pass through a linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return attention_output
