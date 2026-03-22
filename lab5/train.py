"""
train.py
========
Tarefa 3 – O Motor de Otimização (Training Loop)

IMPORTANTE: O fluxo de Forward/Backward neste módulo interage
EXCLUSIVAMENTE com as classes construídas manualmente nos laboratórios
anteriores (Transformer, Encoder, Decoder, MultiHeadAttention, etc.),
conforme exigido pelo enunciado.

Fluxo de cada iteração:
  1. Forward  → modelo(src, tgt_input)  →  logits
  2. Loss     → CrossEntropyLoss(logits, tgt_output)  [ignore_index=PAD]
  3. Backward → loss.backward()
  4. Step     → optimizer.step()
"""

import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.transformer import Transformer


# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÕES DO LABORATÓRIO
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "d_model"   : 128,    # dimensão do modelo
    "num_heads" : 4,      # cabeças de atenção
    "num_layers": 2,      # camadas encoder/decoder
    "d_ff"      : 512,    # dimensão interna do FFN (4 × d_model)
    "dropout"   : 0.1,
    "max_seq_len": 64,
    "lr"        : 1e-3,   # taxa de aprendizado Adam
    "epochs"    : 15,     # número de épocas
}


# ──────────────────────────────────────────────────────────────
# INSTANCIAÇÃO DO MODELO
# ──────────────────────────────────────────────────────────────

def build_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    pad_idx: int,
    config: dict = None,
) -> Transformer:
    """
    Instancia o Transformer com as dimensões viáveis para o laboratório.

    Parâmetros
    ----------
    src_vocab_size : tamanho do vocabulário fonte
    tgt_vocab_size : tamanho do vocabulário destino
    pad_idx        : índice do token de padding
    config         : dicionário de hiperparâmetros (DEFAULT_CONFIG se None)

    Retorna
    -------
    Transformer inicializado (weights Xavier)
    """
    cfg = config or DEFAULT_CONFIG

    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = cfg["d_model"],
        num_heads      = cfg["num_heads"],
        num_layers     = cfg["num_layers"],
        d_ff           = cfg["d_ff"],
        max_seq_len    = cfg["max_seq_len"],
        dropout        = cfg["dropout"],
        pad_idx        = pad_idx,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Transformer instanciado | parâmetros treináveis: {n_params:,}")
    return model


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────────

def train(
    model     : Transformer,
    dataloader: DataLoader,
    pad_idx   : int,
    config    : dict = None,
    device    : str  = "cpu",
) -> List[float]:
    """
    Executa o Training Loop completo com Backpropagation e Adam.

    Passos por epoch:
      1. Forward Pass   → logits = model(src, tgt_input)
      2. Loss           → CrossEntropyLoss(logits, tgt_output)
      3. Backward Pass  → loss.backward()
      4. Optimizer Step → optimizer.step()

    Parâmetros
    ----------
    model      : instância do Transformer (src.transformer)
    dataloader : DataLoader com pares (src, tgt)
    pad_idx    : índice de padding (ignorado na loss)
    config     : hiperparâmetros
    device     : 'cpu' ou 'cuda'

    Retorna
    -------
    Lista com o valor médio da loss por época
    """
    cfg = config or DEFAULT_CONFIG
    model = model.to(device)

    # ── Função de Perda ──────────────────────────────────────
    # ignore_index=pad_idx: a rede não é penalizada por errar
    # tokens de padding artificiais.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # ── Otimizador Adam ──────────────────────────────────────
    # Mesmo otimizador do paper original (Vaswani et al., 2017)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9)

    loss_history: List[float] = []

    print("\n" + "="*60)
    print("  TRAINING LOOP – Transformer Lab 05")
    print("="*60)
    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  Batches/ep  : {len(dataloader)}")
    print(f"  Optimizer   : Adam (lr={cfg['lr']})")
    print(f"  Device      : {device}")
    print("="*60 + "\n")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)          # (B, src_len)
            tgt = tgt.to(device)          # (B, tgt_len)  inclui <BOS> e <EOS>

            # ── Teacher Forcing ──────────────────────────────
            # tgt_input : tudo exceto o último token  (<BOS> … último_token)
            # tgt_output: tudo exceto o primeiro token (primeiro_token … <EOS>)
            # O modelo aprende: dado X tokens anteriores, prever o próximo.
            tgt_input  = tgt[:, :-1]     # (B, T-1)
            tgt_output = tgt[:, 1:]      # (B, T-1)

            # ── 1. FORWARD PASS ──────────────────────────────
            # Passa src pelo Encoder e tgt_input pelo Decoder.
            # logits shape: (B, T-1, tgt_vocab_size)
            logits = model(src, tgt_input)

            # ── 2. CÁLCULO DA LOSS ───────────────────────────
            # CrossEntropyLoss espera (N, C) e (N,)
            # Reshape: (B*(T-1), vocab) vs (B*(T-1),)
            B, T, V = logits.shape
            loss = criterion(
                logits.reshape(B * T, V),
                tgt_output.reshape(B * T),
            )

            # Conta tokens reais (não-padding) para logging
            n_tokens = (tgt_output != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            # ── 3. BACKWARD PASS ─────────────────────────────
            optimizer.zero_grad()
            loss.backward()   # calcula gradientes de WQ, WK, WV, etc.

            # Gradient clipping (estabilidade numérica)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── 4. ATUALIZAÇÃO DOS PESOS ─────────────────────
            optimizer.step()  # atualiza WQ, WK, WV, WO, W1, W2, ...

        avg_loss = total_loss / max(total_tokens, 1)
        loss_history.append(avg_loss)
        elapsed = time.time() - t0

        print(
            f"  Época [{epoch:02d}/{cfg['epochs']}]  |  "
            f"Loss: {avg_loss:.4f}  |  "
            f"Tempo: {elapsed:.1f}s"
        )

    print("\n[train] Treinamento concluído.")
    return loss_history


# ──────────────────────────────────────────────────────────────
# LOOP AUTO-REGRESSIVO (INFERÊNCIA / GERAÇÃO)
# ──────────────────────────────────────────────────────────────

def greedy_decode(
    model     : Transformer,
    src_tensor: torch.Tensor,
    bos_id    : int,
    eos_id    : int,
    max_len   : int = 50,
    device    : str = "cpu",
) -> List[int]:
    """
    Loop auto-regressivo: gera a tradução token a token (greedy search).

    Algoritmo:
      1. Codifica a frase fonte com o Encoder.
      2. Inicializa a sequência destino com [<BOS>].
      3. A cada passo, passa a sequência atual pelo Decoder
         e seleciona o token com maior probabilidade (argmax).
      4. Repete até gerar <EOS> ou atingir max_len.

    Parâmetros
    ----------
    model      : Transformer treinado
    src_tensor : (1, src_len) – frase fonte tokenizada
    bos_id     : ID do token de início
    eos_id     : ID do token de fim
    max_len    : comprimento máximo da geração
    device     : dispositivo

    Retorna
    -------
    Lista de IDs gerados (sem <BOS>)
    """
    model.eval()
    src = src_tensor.to(device)

    with torch.no_grad():
        # Codifica toda a frase fonte de uma vez
        src_mask   = model.make_src_mask(src)
        enc_output = model.encoder(src, src_mask)

        # Inicializa a sequência destino com <BOS>
        generated = [bos_id]

        for _ in range(max_len):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_mask   = model.make_tgt_mask(tgt_tensor)

            # Decodificação auto-regressiva
            dec_output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
            logits     = model.output_projection(dec_output)  # (1, t, vocab)

            # Greedy: pega o token com maior logit na última posição
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)

            if next_token == eos_id:
                break

    return generated[1:]  # remove <BOS> inicial


# ──────────────────────────────────────────────────────────────
# SALVAMENTO / CARREGAMENTO DE CHECKPOINT
# ──────────────────────────────────────────────────────────────

def save_checkpoint(model: Transformer, path: str, extra: dict = None):
    """Salva pesos do modelo em disco."""
    state = {"model_state": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"[checkpoint] Modelo salvo em '{path}'")


def load_checkpoint(model: Transformer, path: str) -> dict:
    """Carrega pesos do modelo de disco."""
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    print(f"[checkpoint] Modelo carregado de '{path}'")
    return state
