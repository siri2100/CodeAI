"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer


''' TODO
    01. CodeAI_v10_Alpha 코드 작성 (아래의 MarianCG Model config 참고)

        MarianConfig {
        "_name_or_path": "AhmedSSoliman/MarianCG-CoNaLa-Large",
        "_num_labels": 3,
        "activation_dropout": 0.0,
        "activation_function": "swish",
        "add_bias_logits": false,
        "add_final_layer_norm": false,
        "architectures": [
            "MarianMTModel"
        ],
        "attention_dropout": 0.0,
        "bad_words_ids": [
            [
            67027
            ]
        ],
        "bos_token_id": 0,
        "classif_dropout": 0.0,
        "classifier_dropout": 0.0,
        "d_model": 512,
        "decoder_attention_heads": 8,
        "decoder_ffn_dim": 2048,
        "decoder_layerdrop": 0.0,
        "decoder_layers": 6,
        "decoder_start_token_id": 67027,
        "decoder_vocab_size": 67028,
        "dropout": 0.1,
        "encoder_attention_heads": 8,
        "encoder_ffn_dim": 2048,
        "encoder_layerdrop": 0.0,
        "encoder_layers": 6,
        "eos_token_id": 0,
        "forced_eos_token_id": 0,
        "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1",
            "2": "LABEL_2"
        },
        "init_std": 0.02,
        "is_encoder_decoder": true,
        "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2
        },
        "max_length": 512,
        "max_position_embeddings": 512,
        "model_type": "marian",
        "normalize_before": false,
        "normalize_embedding": false,
        "num_beams": 4,
        "num_hidden_layers": 6,
        "pad_token_id": 67027,
        "scale_embedding": true,
        "share_encoder_decoder_embeddings": true,
        "static_position_embeddings": true,
        "torch_dtype": "float32",
        "transformers_version": "4.30.0.dev0",
        "use_cache": true,
        "vocab_size": 67028
        }
'''


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Model_V10_Alpha(nn.Module):
    def __init__(self,
                 device,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 dim_feedforward: int=512,
                 dropout: float=0.1,
                ):
        super(Model_V10_Alpha, self).__init__()
        self.device = device
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True,
                                       )
        self.generator = nn.Linear(emb_size, dst_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.dst_tok_emb = TokenEmbedding(dst_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, dst):
        src_emb = self.src_tok_emb(src)
        dst_emb = self.dst_tok_emb(dst)
        src_emb = self.positional_encoding(src_emb)
        dst_emb = self.positional_encoding(dst_emb)
        out = self.transformer(src_emb, dst_emb)
        return self.generator(out)

    def encode(self, src:Tensor, src_mask:Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, dst:Tensor, memory:Tensor, dst_mask:Tensor):
        return self.transformer.decoder(self.positional_encoding(self.dst_tok_emb(dst)), memory, dst_mask)
    