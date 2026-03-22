"""
overfit_test.py
===============
Tarefa 4 – A Prova de Fogo (Overfitting Test)

Técnica clássica de debugging de redes neurais:
  Forçar o modelo a memorizar um conjunto ínfimo de dados para provar
  que os gradientes estão fluindo corretamente e que a arquitetura
  consegue aprender.

Fluxo:
  1. Seleciona N frases do conjunto de treino (N = 5 a 10)
  2. Treina o modelo APENAS nessas frases por muitas épocas
  3. Usa o Loop Auto-regressivo para gerar a tradução das mesmas frases
  4. Compara saída gerada com a tradução esperada
  5. Se o modelo "decorou" → arquitetura e gradientes estão corretos ✓
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.transformer import Transformer
from src.dataset import TranslationDataset, Tokenizer, tokenize_pairs, collate_fn
from src.train import greedy_decode


# ──────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────

OVERFIT_SAMPLES = 8    # número de frases para memorizar
OVERFIT_EPOCHS  = 300  # muitas épocas para garantir memorização
OVERFIT_LR      = 5e-4


# ──────────────────────────────────────────────────────────────
# TREINAMENTO DE OVERFITTING
# ──────────────────────────────────────────────────────────────

def run_overfit_test(
    model     : Transformer,
    raw_pairs : List[Tuple[str, str]],
    tokenizer : Tokenizer,
    pad_idx   : int,
    n_samples : int = OVERFIT_SAMPLES,
    epochs    : int = OVERFIT_EPOCHS,
    lr        : float = OVERFIT_LR,
    device    : str   = "cpu",
) -> None:
    """
    Executa o Overfitting Test completo.

    Passos:
      1. Pega as primeiras n_samples frases do conjunto de treino
      2. Treina o modelo exclusivamente nessas frases
      3. Roda o loop auto-regressivo nas mesmas frases
      4. Imprime comparação esperado vs gerado

    Parâmetros
    ----------
    model      : Transformer (já instanciado, pode ser reinicializado)
    raw_pairs  : lista de pares (src_text, tgt_text) originais
    tokenizer  : instância de Tokenizer
    pad_idx    : índice de padding
    n_samples  : quantas frases para memorizar
    epochs     : épocas de overfitting
    lr         : taxa de aprendizado
    device     : dispositivo de treino
    """

    print("\n" + "="*60)
    print("  OVERFITTING TEST – Prova de Fogo")
    print("="*60)

    # ── 1. Seleciona subconjunto minúsculo ────────────────────
    tiny_pairs = raw_pairs[:n_samples]
    print(f"\n[overfit] Frases selecionadas para memorização ({n_samples}):")
    for i, (src, tgt) in enumerate(tiny_pairs):
        print(f"  [{i+1}] EN: {src}")
        print(f"       DE: {tgt}")

    # ── 2. Tokeniza e cria DataLoader ─────────────────────────
    tokenized = tokenize_pairs(tiny_pairs, tokenizer)
    dataset   = TranslationDataset(tokenized)
    loader    = DataLoader(
        dataset,
        batch_size=n_samples,  # 1 batch com todas as frases
        shuffle=False,
        collate_fn=collate_fn(pad_idx),
    )

    # ── 3. Redefine otimizador (Adam com lr maior) ────────────
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── 4. Loop de overfitting ────────────────────────────────
    print(f"\n[overfit] Iniciando {epochs} épocas de memorização...\n")

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            B, T, V = logits.shape

            loss = criterion(
                logits.reshape(B * T, V),
                tgt_output.reshape(B * T),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss = loss.item()

        loss_history.append(total_loss)

        # Imprime a cada 50 épocas
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Época [{epoch:03d}/{epochs}]  |  Loss: {total_loss:.6f}")

    # ── 5. Avalia memorização ─────────────────────────────────
    print("\n" + "-"*60)
    print("  RESULTADO DA PROVA DE FOGO")
    print("-"*60)

    model.eval()
    for i, (src_text, tgt_text) in enumerate(tiny_pairs):
        # Tokeniza a frase fonte
        src_ids = tokenizer.encode_src(src_text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long)

        # Loop auto-regressivo
        generated_ids = greedy_decode(
            model     = model,
            src_tensor= src_tensor,
            bos_id    = tokenizer.bos_id,
            eos_id    = tokenizer.eos_id,
            max_len   = 50,
            device    = device,
        )

        generated_text = tokenizer.decode(generated_ids)

        print(f"\n  Amostra [{i+1}]")
        print(f"    Fonte (EN)   : {src_text}")
        print(f"    Esperado (DE): {tgt_text}")
        print(f"    Gerado   (DE): {generated_text}")

        # Verifica similaridade simples (tokens sobrepostos)
        exp_tokens = set(tgt_text.lower().split())
        gen_tokens = set(generated_text.lower().split())
        overlap = len(exp_tokens & gen_tokens) / max(len(exp_tokens), 1)
        status = "✓ MEMORIZADO" if overlap > 0.5 else "✗ ainda aprendendo"
        print(f"    Sobreposição : {overlap:.0%}  {status}")

    print("\n[overfit] Teste concluído.")
    print("[overfit] Se a Loss chegou próximo de 0 e as traduções batem,")
    print("[overfit] os gradientes estão fluindo corretamente. ✓")

    return loss_history
