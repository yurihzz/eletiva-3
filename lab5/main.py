"""
main.py
=======
Ponto de entrada principal do Laboratório 05.

Executa em sequência:
  → Tarefa 1: Carrega dataset real (Hugging Face)
  → Tarefa 2: Tokenização com bert-base-multilingual-cased
  → Tarefa 3: Training Loop (Forward → Loss → Backward → Step)
  → Tarefa 4: Overfitting Test (Prova de Fogo)

Uso:
  python main.py                  # execução padrão (CPU)
  python main.py --epochs 20      # customizar épocas
  python main.py --skip-overfit   # pula Tarefa 4
"""

import argparse
import os
import sys

import torch

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import build_dataloader, Tokenizer, MAX_SUBSET
from src.train import build_model, train, DEFAULT_CONFIG, save_checkpoint
from src.overfit_test import run_overfit_test


# ──────────────────────────────────────────────────────────────
# ARGUMENTOS DE LINHA DE COMANDO
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Lab 05 – Transformer End-to-End Training")
    parser.add_argument("--epochs",       type=int,   default=15,       help="Épocas de treinamento")
    parser.add_argument("--batch-size",   type=int,   default=32,       help="Tamanho do batch")
    parser.add_argument("--lr",           type=float, default=1e-3,     help="Taxa de aprendizado")
    parser.add_argument("--max-samples",  type=int,   default=MAX_SUBSET, help="Subconjunto do dataset")
    parser.add_argument("--skip-overfit", action="store_true",           help="Pular Tarefa 4")
    parser.add_argument("--save-model",   action="store_true",           help="Salvar modelo após treino")
    parser.add_argument("--device",       type=str,   default="auto",   help="'cpu', 'cuda' ou 'auto'")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Dispositivo
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n[main] Dispositivo: {device.upper()}")

    # ── TAREFA 1 + 2: Dataset e Tokenização ──────────────────
    print("\n" + "="*60)
    print("  TAREFA 1 + 2: Dataset e Tokenização")
    print("="*60)

    tokenizer = Tokenizer()

    config = {**DEFAULT_CONFIG, "epochs": args.epochs, "lr": args.lr}

    dataloader, tokenizer, raw_pairs = build_dataloader(
        split       = "train",
        max_samples = args.max_samples,
        batch_size  = args.batch_size,
        shuffle     = True,
        tokenizer   = tokenizer,
    )

    print(f"\n[main] Exemplo de tokenização:")
    src_ex, tgt_ex = raw_pairs[0]
    src_ids = tokenizer.encode_src(src_ex)
    tgt_ids = tokenizer.encode_tgt(tgt_ex)
    print(f"  EN texto : {src_ex}")
    print(f"  EN IDs   : {src_ids[:10]}...")
    print(f"  DE texto : {tgt_ex}")
    print(f"  DE IDs   : {tgt_ids[:10]}...  (<BOS>={tokenizer.bos_id}, <EOS>={tokenizer.eos_id})")

    # ── TAREFA 3: Training Loop ───────────────────────────────
    print("\n" + "="*60)
    print("  TAREFA 3: Training Loop")
    print("="*60)

    model = build_model(
        src_vocab_size = tokenizer.vocab_size,
        tgt_vocab_size = tokenizer.vocab_size,
        pad_idx        = tokenizer.pad_id,
        config         = config,
    )

    loss_history = train(
        model      = model,
        dataloader = dataloader,
        pad_idx    = tokenizer.pad_id,
        config     = config,
        device     = device,
    )

    # Resumo da convergência
    print("\n[main] Resumo da convergência:")
    print(f"  Loss inicial  : {loss_history[0]:.4f}")
    print(f"  Loss final    : {loss_history[-1]:.4f}")
    queda = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
    print(f"  Queda total   : {queda:.1f}%")

    if loss_history[-1] < loss_history[0]:
        print("  ✓ A Loss caiu! A arquitetura está aprendendo corretamente.")
    else:
        print("  ✗ A Loss não caiu. Verifique hiperparâmetros.")

    # Salva modelo
    if args.save_model:
        os.makedirs("outputs", exist_ok=True)
        save_checkpoint(model, "outputs/transformer_lab05.pt", {
            "loss_history": loss_history,
            "vocab_size"  : tokenizer.vocab_size,
        })

    # ── TAREFA 4: Overfitting Test ────────────────────────────
    if not args.skip_overfit:
        print("\n" + "="*60)
        print("  TAREFA 4: Overfitting Test (Prova de Fogo)")
        print("="*60)

        # Reinicializa modelo fresco para o teste de overfitting
        model_overfit = build_model(
            src_vocab_size = tokenizer.vocab_size,
            tgt_vocab_size = tokenizer.vocab_size,
            pad_idx        = tokenizer.pad_id,
            config         = config,
        )

        run_overfit_test(
            model      = model_overfit,
            raw_pairs  = raw_pairs,
            tokenizer  = tokenizer,
            pad_idx    = tokenizer.pad_id,
            device     = device,
        )

    print("\n" + "="*60)
    print("  Lab 05 concluído com sucesso!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
