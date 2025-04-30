import math
import torch
from torch import nn
import text_init


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0,
                                seq_length,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             .float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(torch.nn.Module):
    """Initialization AI architecture
    """
    def __init__(self, num_tokens, seq_length, d_model):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.seq_lenth = seq_length
        self.num_tokens = num_tokens

        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_emb = PositionalEncoding(d_model, seq_length)

        self.transformer = torch.nn.Transformer(num_encoder_layers=2,
                                                num_decoder_layers=2,
                                                dim_feedforward=256,
                                                dropout=0.3,
                                                d_model=d_model,
                                                nhead=4,
                                                batch_first=True)

        self.ff = nn.Linear(d_model, num_tokens)

    def forward(self, src, tgt):
        src = self.token_emb(src)
        src = self.pos_emb(src)

        tgt = self.token_emb(tgt)
        tgt = self.pos_emb(tgt)

        result = self.transformer(src, tgt)
        result = self.ff(result)

        return result

    def generate(self, src):
        tgt = src + [0 for _ in range(self.seq_lenth - len(src))]
        tgt = torch.tensor(tgt)
        src = tgt

        for word_index in range(len(src) - self.seq_lenth, self.seq_lenth - 1):
            output = self.forward(src,
                                  tgt.long())
            output = torch.argmax(output, dim=-1)
            tgt[word_index] = output[0][word_index]

        return output


transformer = Transformer(num_tokens=len(text_init.learn_dict),
                          seq_length=text_init.TEXT_LENTH,
                          d_model=512)
optimizer = torch.optim.AdamW(transformer.parameters(),
                              lr=0.001)
loss = torch.nn.CrossEntropyLoss()
