import torch
import torch.nn as nn
import math

### Input embeddings 
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

### Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ### Creating matrices for the positional encoding of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        ### Create a vector of shape (seq_len, 1)
        position = torch.arrange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        ### Apply sin to every even position
        pe[:, 0::2] = torch.sin(position * div_term)
        ### Apply cos to every odd position
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # it will become a tensor of shape (1, seq_len, d_model)

        """ when we have a tensor that we want to save the file of module, we should register as a buffer"""  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
