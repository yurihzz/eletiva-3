# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

> **Disciplina:** Tópicos em Inteligência Artificial  
> **Instituição:** Instituto de Ensino Superior iCEV  
> **Domínio escolhido:** Suporte Técnico de TI


## Objetivo

Pipeline completo de fine-tuning do modelo **Llama 2 7B** utilizando:

- **QLoRA** — quantização 4-bit (NF4) via `bitsandbytes` para reduzir o uso de VRAM
- **LoRA (PEFT)** — treino apenas de matrizes de baixo rank, congelando os pesos originais
- **SFTTrainer (trl)** — orquestração do treinamento supervisionado
- **Dataset sintético** — 50+ pares instrução/resposta gerados via API da OpenAI

---

## Estrutura do Repositório

```
lab07-qlora/
├── step1_generate_dataset.py     # Geração do dataset sintético (OpenAI API)
├── step2_3_4_qlora_finetune.py   # Fine-tuning completo com QLoRA
├── train_dataset.jsonl           # Dataset de treino (90%)
├── test_dataset.jsonl            # Dataset de teste (10%)
└── README.md                     # Este arquivo
```

---

## Pré-requisitos

```bash
pip install transformers peft trl bitsandbytes accelerate datasets openai
```

**Hardware recomendado:** GPU com ≥ 16 GB VRAM (ex: A100, RTX 3090/4090).  
Para GPUs menores (8–12 GB), reduza `per_device_train_batch_size` para `1`.

---

## Como Executar

### Passo 1 — Gerar o Dataset

```bash
export OPENAI_API_KEY="sua-chave-aqui"
python step1_generate_dataset.py
```

Isso gera `train_dataset.jsonl` e `test_dataset.jsonl`.

### Passo 2, 3 e 4 — Fine-tuning com QLoRA

```bash
python step2_3_4_qlora_finetune.py
```

O adaptador LoRA será salvo em `./llama2-it-support-qlora/adapter/`.

---

## Hiperparâmetros Principais

| Componente | Parâmetro | Valor |
|---|---|---|
| **Quantização** | Tipo | NF4 (4-bit) |
| **Quantização** | Compute dtype | float16 |
| **LoRA** | Rank (r) | 64 |
| **LoRA** | Alpha | 16 |
| **LoRA** | Dropout | 0.1 |
| **Otimizador** | Tipo | paged_adamw_32bit |
| **Scheduler** | Tipo | cosine |
| **Scheduler** | Warmup ratio | 0.03 |
| **Treino** | Épocas | 3 |
| **Treino** | Learning rate | 2e-4 |

---

## Uso de IA

> **Partes geradas/complementadas com IA, revisadas por Yuri Ribeiro Estrela.**

Ferramentas de IA (Claude/ChatGPT) foram utilizadas para auxiliar na estrutura inicial dos scripts e na documentação. Todo o código foi revisado criticamente, compreendido e validado conforme os requisitos do laboratório.

---

## Release

A versão final avaliada está marcada como **v1.0**.
