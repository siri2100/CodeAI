import torch
import torch.nn as nn


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


class CodeAI_V10_Alpha(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x