"""
Passos 2, 3 e 4: Fine-tuning com QLoRA (Quantização + LoRA + SFTTrainer)
Modelo base: NousResearch/Llama-2-7b-hf (variante open-access do Llama 2 7B)

Pré-requisitos:
    pip install transformers peft trl bitsandbytes accelerate datasets

Hardware recomendado: GPU com >= 16 GB VRAM (ex: A100, RTX 3090/4090)
Para GPUs menores (8–12 GB), reduza per_device_train_batch_size para 1.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# ─────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ─────────────────────────────────────────────
# Modelo base (sem necessidade de aceite de termos)
BASE_MODEL = "NousResearch/Llama-2-7b-hf"
OUTPUT_DIR = "./llama2-it-support-qlora"
TRAIN_FILE = "train_dataset.jsonl"
TEST_FILE = "test_dataset.jsonl"

# ─────────────────────────────────────────────
# PASSO 2: CONFIGURAÇÃO DA QUANTIZAÇÃO (QLoRA)
# ─────────────────────────────────────────────
# Carrega o modelo em 4-bits usando NormalFloat4 (nf4) para economizar memória de GPU.
# O compute_dtype define a precisão dos cálculos durante o forward pass.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Carrega pesos em 4-bit
    # Tipo de quantização: NormalFloat 4-bit
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,       # Precisão de cálculo: float16
    # Double quantization para maior economia
    bnb_4bit_use_double_quant=True,
)

# ─────────────────────────────────────────────
# PASSO 3: ARQUITETURA DO LoRA
# ─────────────────────────────────────────────
# O LoRA congela os pesos originais e injeta matrizes de baixo rank treináveis
# nas camadas de atenção (q_proj, v_proj, etc.), reduzindo drásticamente os
# parâmetros a atualizar.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Tarefa: geração de linguagem causal
    # Rank: dimensão das matrizes menores (A e B)
    r=64,
    # Alpha: fator de escala dos novos pesos (escala = alpha/r)
    lora_alpha=16,
    lora_dropout=0.1,               # Dropout para regularização e evitar overfitting
    bias="none",                    # Não treina os bias das camadas
    target_modules=[                # Camadas onde o LoRA será aplicado
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


def format_instruction(sample: dict) -> str:
    """Formata cada amostra no template Alpaca (instruction → response)."""
    return (
        "### Instrução:\n"
        f"{sample['instruction']}\n\n"
        "### Resposta:\n"
        f"{sample['response']}"
    )


def main():
    # ── Carrega datasets ──────────────────────────────────────────────────────
    print("📂 Carregando datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "test": TEST_FILE},
    )
    print(f"   Treino: {len(dataset['train'])} exemplos")
    print(f"   Teste : {len(dataset['test'])} exemplos")

    # ── Carrega tokenizer ─────────────────────────────────────────────────────
    print("\n🔤 Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # Llama não tem pad_token por padrão
    # Necessário para evitar warnings com fp16
    tokenizer.padding_side = "right"

    # ── Carrega modelo base com quantização 4-bit ─────────────────────────────
    print("\n🤖 Carregando modelo base com QLoRA (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",          # Distribui automaticamente entre GPUs disponíveis
        trust_remote_code=True,
    )
    # Desabilita KV cache durante o treino
    model.config.use_cache = False
    model.config.pretraining_tp = 1            # Evita bug de tensor parallelism

    # ─────────────────────────────────────────────
    # PASSO 4: PIPELINE DE TREINAMENTO E OTIMIZAÇÃO
    # ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     # Número de épocas
        # Batch por GPU (reduza para 1 se necessário)
        per_device_train_batch_size=4,
        # Acumula gradientes para simular batch maior
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,            # Recomputa ativações para economizar VRAM

        # ── Engenharia do otimizador (conforme requisito) ───────────────────
        # AdamW paginado: transfere picos de memória GPU→CPU
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",             # Scheduler cosine: LR decai em curva suave
        warmup_ratio=0.03,                      # Primeiros 3% do treino com LR crescente

        # ── Hiperparâmetros de aprendizado ──────────────────────────────────
        learning_rate=2e-4,
        weight_decay=0.001,
        max_grad_norm=0.3,                      # Gradient clipping

        # ── Configurações de precisão ────────────────────────────────────────
        fp16=True,                              # Mixed precision com float16

        # ── Logging e avaliação ──────────────────────────────────────────────
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=25,
        save_steps=100,
        save_total_limit=2,

        # ── Reprodutibilidade ────────────────────────────────────────────────
        seed=42,
        # Desabilita W&B/TensorBoard (use "tensorboard" se quiser)
        report_to="none",
    )

    # Instancia o SFTTrainer (Supervised Fine-Tuning Trainer da biblioteca trl)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        formatting_func=format_instruction,     # Aplica template Alpaca a cada amostra
        max_seq_length=512,                     # Comprimento máximo da sequência tokenizada
        tokenizer=tokenizer,
        args=training_args,
    )

    # ── Inicia o treinamento ──────────────────────────────────────────────────
    print("\n🚀 Iniciando fine-tuning...")
    trainer.train()

    # ── Salva o adaptador LoRA (apenas os pesos delta, não o modelo completo) ─
    print(f"\n💾 Salvando adaptador LoRA em '{OUTPUT_DIR}/adapter'...")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")

    print("\n✅ Fine-tuning concluído com sucesso!")
    print(f"   Adaptador salvo em: {OUTPUT_DIR}/adapter")


if __name__ == "__main__":
    main()
