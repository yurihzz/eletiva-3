"""
dataset.py
==========
Tarefa 1 – Preparação do Dataset (Hugging Face)
Tarefa 2 – Tokenização Básica

NOTA: As Tarefas 1 e 2 foram implementadas com auxílio de IA Generativa
(Claude) para facilitar a manipulação do dataset e a tokenização,
conforme permitido pelo enunciado do laboratório.

Responsabilidades deste módulo:
  - Carregar o dataset bentrevett/multi30k (pares EN→DE)
  - Selecionar um subconjunto de até 1.000 frases
  - Tokenizar usando AutoTokenizer (bert-base-multilingual-cased)
  - Adicionar tokens especiais <BOS>/<EOS> e aplicar padding
  - Expor a classe TranslationDataset (torch.utils.data.Dataset)
"""

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ──────────────────────────────────────────────────────────────────────────────
# NOTA: as bibliotecas abaixo são carregadas com lazy import para que o módulo
# possa ser importado mesmo sem internet (útil em testes offline).
# ──────────────────────────────────────────────────────────────────────────────

def _load_hf():
    from datasets import load_dataset
    return load_dataset

def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


# ──────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────

DATASET_NAME   = "bentrevett/multi30k"
TOKENIZER_NAME = "bert-base-multilingual-cased"
MAX_SUBSET     = 1000   # primeiras N frases conforme enunciado
MAX_SEQ_LEN    = 64     # comprimento máximo após truncamento
PAD_IDX        = 0      # índice de padding (padrão BERT: [PAD] = 0)


# ──────────────────────────────────────────────────────────────
# 1. CARREGAMENTO DO DATASET
# ──────────────────────────────────────────────────────────────

def load_raw_pairs(split: str = "train", max_samples: int = MAX_SUBSET) -> List[Tuple[str, str]]:
    """
    Carrega o dataset multi30k e retorna uma lista de pares (en, de).

    Parâmetros
    ----------
    split       : partição do dataset ('train', 'validation', 'test')
    max_samples : número máximo de pares a carregar

    Retorna
    -------
    list of (source_str, target_str)
    """
    load_dataset = _load_hf()

    print(f"[dataset] Carregando {DATASET_NAME} – split='{split}' (máx {max_samples} amostras)...")
    dataset = load_dataset(DATASET_NAME, split=split)

    # Seleciona subconjunto
    subset = dataset.select(range(min(max_samples, len(dataset))))

    pairs = [(row["en"], row["de"]) for row in subset]
    print(f"[dataset] {len(pairs)} pares carregados.")
    return pairs


# ──────────────────────────────────────────────────────────────
# 2. TOKENIZAÇÃO
# ──────────────────────────────────────────────────────────────

class Tokenizer:
    """
    Wrapper em torno do AutoTokenizer do Hugging Face.

    Tokens especiais relevantes (bert-base-multilingual-cased):
      [PAD] → 0
      [UNK] → 100
      [CLS] → 101   ← usamos como <BOS>
      [SEP] → 102   ← usamos como <EOS>
    """

    def __init__(self, model_name: str = TOKENIZER_NAME):
        print(f"[tokenizer] Carregando '{model_name}'...")
        self.tok = _load_tokenizer(model_name)
        self.pad_id = self.tok.pad_token_id   # 0
        self.bos_id = self.tok.cls_token_id   # 101 ([CLS] → <BOS>)
        self.eos_id = self.tok.sep_token_id   # 102 ([SEP] → <EOS>)
        self.vocab_size = self.tok.vocab_size
        print(f"[tokenizer] vocab_size={self.vocab_size} | PAD={self.pad_id} | BOS={self.bos_id} | EOS={self.eos_id}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Converte texto em lista de IDs (sem truncamento aqui)."""
        return self.tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_src(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        """
        Codifica a frase fonte (Encoder).
        Sem tokens especiais de início/fim (somente padding será aplicado).
        """
        ids = self.encode(text, add_special_tokens=False)
        return ids[:max_len]

    def encode_tgt(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        """
        Codifica a frase destino (Decoder).
        Adiciona <BOS> no início e <EOS> no fim, conforme enunciado.
        """
        ids = self.encode(text, add_special_tokens=False)
        ids = [self.bos_id] + ids[: max_len - 2] + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converte lista de IDs de volta em texto."""
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


def tokenize_pairs(
    pairs: List[Tuple[str, str]],
    tokenizer: Tokenizer,
    max_len: int = MAX_SEQ_LEN,
) -> List[Dict[str, List[int]]]:
    """
    Itera pelos pares (src, tgt) e converte em listas de inteiros.

    Retorna
    -------
    Lista de dicts: {"src": [int, ...], "tgt": [int, ...]}
    """
    tokenized = []
    for src_text, tgt_text in pairs:
        src_ids = tokenizer.encode_src(src_text, max_len)
        tgt_ids = tokenizer.encode_tgt(tgt_text, max_len)
        if len(src_ids) > 0 and len(tgt_ids) > 2:  # filtra frases vazias
            tokenized.append({"src": src_ids, "tgt": tgt_ids})
    print(f"[tokenizer] {len(tokenized)} pares tokenizados.")
    return tokenized


# ──────────────────────────────────────────────────────────────
# 3. DATASET PYTORCH
# ──────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Dataset PyTorch para pares de tradução tokenizados.

    Cada item retorna:
      src : Tensor de IDs da língua fonte
      tgt : Tensor de IDs da língua destino (inclui <BOS> e <EOS>)
    """

    def __init__(self, tokenized_pairs: List[Dict[str, List[int]]]):
        self.data = tokenized_pairs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        src = torch.tensor(item["src"], dtype=torch.long)
        tgt = torch.tensor(item["tgt"], dtype=torch.long)
        return src, tgt


def collate_fn(pad_idx: int = PAD_IDX):
    """
    Função de collate para o DataLoader.

    Aplica padding (preenchimento com zeros) para que todas as frases
    do batch tenham o mesmo comprimento matemático.
    """
    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        srcs, tgts = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        return src_padded, tgt_padded
    return _collate


# ──────────────────────────────────────────────────────────────
# 4. FUNÇÃO AUXILIAR: PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────

def build_dataloader(
    split: str = "train",
    max_samples: int = MAX_SUBSET,
    batch_size: int = 32,
    shuffle: bool = True,
    tokenizer: Tokenizer = None,
) -> Tuple[DataLoader, Tokenizer, List[Tuple[str, str]]]:
    """
    Pipeline completo: carrega dataset → tokeniza → cria DataLoader.

    Retorna
    -------
    (dataloader, tokenizer, raw_pairs)
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    raw_pairs = load_raw_pairs(split=split, max_samples=max_samples)
    tokenized = tokenize_pairs(raw_pairs, tokenizer)

    dataset = TranslationDataset(tokenized)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn(tokenizer.pad_id),
    )
    return loader, tokenizer, raw_pairs
