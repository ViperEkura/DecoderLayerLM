import math
import torch
import torch.nn as nn
from config import config


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.dim = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros((self.max_len, self.dim), requires_grad=False)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, dtype=torch.float) * (-math.log(10000.0) / self.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()
        query = self.q_proj(q)
        key = self.k_proj(k)
        value = self.v_proj(v)
        query = query.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        output = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask) \
            if mask is not None else nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        # (batch_size, num_heads, seq_len, head_dim)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.dim)
        output = self.o_proj(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_feed_forward):
        super(FeedForward, self).__init__()
        self.gate_proj = nn.Linear(d_model, d_feed_forward)
        self.input_proj = nn.Linear(d_model, d_feed_forward)
        self.output_proj = nn.Linear(d_feed_forward, d_model)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = nn.functional.silu(gate)
        x = self.input_proj(x)
        out = self.output_proj(x * gate)
        out = nn.functional.relu(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(d_model))
        self.bias = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        center = x - m
        variance = center.pow(2).mean(-1, keepdim=True)
        x = center * torch.rsqrt(variance + self.eps)
        out = self.weight * x + self.bias
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.feedforward_dim = config.feedforward_dim

        self.attention_0 = Attention(self.d_model, self.n_head)
        self.norm_0 = LayerNorm(self.d_model)
        self.ffn = FeedForward(self.d_model, self.feedforward_dim)
        self.norm_1 = LayerNorm(self.d_model)

    def forward(self, input_tensor):
        attn_tensor = self.attention_0(input_tensor, input_tensor, input_tensor)
        residual = input_tensor + attn_tensor
        norm = self.norm_0(residual)

        ffn = self.ffn(norm)
        out = ffn + norm
        out = self.norm_1(out)
        return out


class DecoderTransformerLM(nn.Module):
    def __init__(self):
        super(DecoderTransformerLM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = PositionalEncoding(config.max_len, config.d_model, config.drop_rate)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layer)])
        self.linear = nn.Linear(config.d_model, config.vocab_size)
        self.linear.weight = self.embedding.weight  # share weight

    def forward(self, input_ids):
        input_tensor = self.embedding(input_ids)
        x = self.positional_embedding(input_tensor)
        for layer in self.decoder_layers:
            x = layer(x)
        logits = self.linear(x)
        return logits


if __name__ == '__main__':
    model = DecoderTransformerLM()
    print(model)
