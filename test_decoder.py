from math import sqrt

import torch
import torch.nn.functional as F

from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoConfig


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = AutoConfig.from_pretrained("bert-based-uncased")
inputs_text = "time flies like an arrow"


# Step 01. Tokenization
inputs_token = tokenizer(inputs_text, return_tensors='pt', add_special_tokens=False)


# Step 02. Embedding
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        token_embed = self.token_embeddings(input_ids)
        pos_embed = self.position_embeddings(position_ids)
        embed = token_embed + pos_embed
        embed = self.ln(embed)
        embed = self.dropout(embed)
        return embed


# Step 03. Masked Multi-Head Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.fc_query = nn.Linear(embed_dim, head_dim)
        self.fc_key = nn.Linear(embed_dim, head_dim)
        self.fc_value = nn.Linear(embed_dim, head_dim)
    
    def forward(self, hidden_state, mask=None):
        query = self.fc_query(hidden_state)
        key = self.fc_key(hidden_state)
        value = self.fc_value(hidden_state)
        attn_outputs = scaled_dot_product_attention(query, key, value, mask)
        return attn_outputs
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, hidden_state, mask=None):
        x = torch.cat([head(hidden_state, mask) for head in self.heads], dim=-1)
        x = self.fc_out(x)
        return x


# Step 04. Feed Forward Neural Network
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Step 05. Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.ffnn = FeedForwardNeuralNetwork(config)
        
    def forward(self, x, mask=None):
        hidden_state = self.ln1(x)
        x = x + self.attention(hidden_state, mask)
        x = x + self.ffnn(self.ln2(x))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
