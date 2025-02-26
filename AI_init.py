import math
import torch
from torch import nn
import text_init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0,
                                max_seq_length,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,
                                          d_model,
                                          2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 0::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(-1)]


class Transformer(torch.nn.Module):
    """Initialization AI architecture
    """
    def __init__(self, num_tokens, seq_length, d_model):
        super(Transformer, self).__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_emb = PositionalEncoding(d_model, seq_length)

        self.transformer = torch.nn.Transformer(num_encoder_layers=2,
                                                num_decoder_layers=2,
                                                dim_feedforward=1024,
                                                dropout=0.3,
                                                d_model=d_model,
                                                nhead=4)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, src, tgt):
        src = self.token_emb(src)
        src = src + self.pos_emb(src)

        tgt = self.token_emb(tgt)
        tgt = tgt + self.pos_emb(tgt)

        result = self.transformer(src, tgt)
        result = self.ff(result)

        return result


# device = torch.device('cuda')
transformer = Transformer(num_tokens=len(text_init.learn_list),
                          seq_length=1,
                          d_model=text_init.TEXT_LENTH)
optimizer = torch.optim.AdamW(transformer.parameters(),
                              lr=0.001)
loss = torch.nn.MSELoss()
