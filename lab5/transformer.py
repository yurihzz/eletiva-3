"""
transformer.py
==============
Arquitetura completa do Transformer construída do zero em PyTorch.
Baseada no paper "Attention Is All You Need" (Vaswani et al., 2017).

Módulos implementados manualmente (sem uso de nn.Transformer):
  - MultiHeadAttention
  - PositionwiseFeedForward
  - PositionalEncoding
  - EncoderLayer / Encoder
  - DecoderLayer / Decoder
  - Transformer (modelo completo)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 1. ATENÇÃO MULTI-CABEÇA
# ──────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Implementação manual da Multi-Head Attention.

    Para cada cabeça h:
        head_i = Attention(Q * WQ_i, K * WK_i, V * WV_i)
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) * WO

    Parâmetros
    ----------
    d_model : int   – dimensão do modelo (embedding)
    num_heads : int – número de cabeças de atenção
    dropout : float – taxa de dropout
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensão por cabeça

        # Projeções lineares WQ, WK, WV, WO
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Attention(Q,K,V) = softmax(QK^T / √d_k) * V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, S_q, S_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)  # (B, h, S_q, d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = query.size(0)

        # Projeção e divisão em cabeças
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # Atenção escalada
        x = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenação: (B, h, S, d_k) → (B, S, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.W_o(x)


# ──────────────────────────────────────────────────────────────
# 2. FEED-FORWARD POSICIONAL
# ──────────────────────────────────────────────────────────────

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Parâmetros
    ----------
    d_model : int – dimensão de entrada/saída
    d_ff    : int – dimensão interna (tipicamente 4 * d_model)
    dropout : float
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ──────────────────────────────────────────────────────────────
# 3. CODIFICAÇÃO POSICIONAL
# ──────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Adiciona informação de posição ao embedding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────
# 4. CAMADA DO ENCODER
# ──────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Uma camada do Encoder:
      Sub-layer 1: Multi-Head Self-Attention + Add & Norm
      Sub-layer 2: Feed-Forward               + Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Sub-layer 1
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ──────────────────────────────────────────────────────────────
# 5. ENCODER COMPLETO
# ──────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Empilha N camadas de EncoderLayer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


# ──────────────────────────────────────────────────────────────
# 6. CAMADA DO DECODER
# ──────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Uma camada do Decoder:
      Sub-layer 1: Masked Multi-Head Self-Attention + Add & Norm
      Sub-layer 2: Cross-Attention (Encoder-Decoder) + Add & Norm
      Sub-layer 3: Feed-Forward                      + Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Sub-layer 1: self-attention mascarada
        sa_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))

        # Sub-layer 2: cross-attention com saída do encoder
        ca_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(ca_out))

        # Sub-layer 3: feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


# ──────────────────────────────────────────────────────────────
# 7. DECODER COMPLETO
# ──────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Empilha N camadas de DecoderLayer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x


# ──────────────────────────────────────────────────────────────
# 8. MODELO COMPLETO
# ──────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Modelo Transformer completo (Encoder-Decoder).

    Parâmetros
    ----------
    src_vocab_size : int   – tamanho do vocabulário da língua fonte
    tgt_vocab_size : int   – tamanho do vocabulário da língua destino
    d_model        : int   – dimensão de embedding (128 para o lab)
    num_heads      : int   – número de cabeças de atenção (4 para o lab)
    num_layers     : int   – número de camadas enc/dec (2 para o lab)
    d_ff           : int   – dimensão interna do FFN (4 * d_model)
    max_seq_len    : int   – comprimento máximo de sequência
    dropout        : float – taxa de dropout
    pad_idx        : int   – índice do token de padding
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Inicialização Xavier para as projeções lineares."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Máscara de padding para o encoder.
        Forma: (batch, 1, 1, src_len)
        """
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Máscara causal + padding para o decoder.
        Forma: (batch, 1, tgt_len, tgt_len)
        """
        B, T = tgt.size()
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T,T)
        return pad_mask & causal_mask  # broadcasting → (B,1,T,T)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        src : (batch, src_len)   – tokens da língua fonte
        tgt : (batch, tgt_len)   – tokens da língua destino (teacher forcing)

        Retorna logits: (batch, tgt_len, tgt_vocab_size)
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.output_projection(dec_output)

        return logits
