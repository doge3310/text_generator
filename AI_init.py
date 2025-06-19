import math
import torch
from torch import nn
from torchinfo import summary
import text_init


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)

        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(p=0.2),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff),
                                          nn.Dropout(p=0.2),
                                          nn.GELU(),
                                          nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


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
        self.num_layers = 2

        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_emb = PositionalEncoding(d_model, seq_length)
        self.decoder_layer = nn.ModuleList([DecoderLayer(d_model=d_model,
                                           num_heads=2,
                                           d_ff=512,
                                           dropout=0.2)
                                            for _ in range(self.num_layers)])

        self.encoder_layer = nn.ModuleList([EncoderLayer(d_model=d_model,
                                           num_heads=2,
                                           d_ff=512,
                                           dropout=0.2)
                                            for _ in range(self.num_layers)])

        self.ff = nn.Linear(d_model, num_tokens)

    def generate_mask(self, src, tgt):
        nopeak_mask = (1 - torch.triu(torch.ones(1,
                                                 self.seq_lenth,
                                                 self.seq_lenth),
                                      diagonal=1)).bool()
        tgt_mask = tgt & nopeak_mask

        return src, tgt_mask.squeeze(0)

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src = self.pos_emb(self.token_emb(src_mask))
        tgt = self.pos_emb(self.token_emb(tgt_mask))

        for index in range(self.num_layers):
            src = self.encoder_layer[index](src, src_mask)

        for index in range(self.num_layers):
            tgt = self.decoder_layer[index](tgt, src, src_mask, tgt_mask)

        result = self.ff(tgt)

        return result


transformer = Transformer(num_tokens=len(text_init.learn_dict),
                          seq_length=text_init.TEXT_LENTH,
                          d_model=128)
optimizer = torch.optim.AdamW(transformer.parameters(),
                              lr=3e-4)
loss = torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    random_data = torch.randint(low=0, high=2000, size=(20, 20))
    summary(transformer, input_data=(random_data, random_data))
