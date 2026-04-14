"""
Passo 1: Geração de Dataset Sintético usando a API da OpenAI
Domínio: Suporte Técnico de TI
Gera 50+ pares de instrução/resposta e salva em .jsonl (90% treino, 10% teste)
"""

import json
import os
import random
from openai import OpenAI

# Inicializa o cliente OpenAI (defina sua chave em variável de ambiente)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DOMAIN_TOPICS = [
    "problemas de conexão Wi-Fi",
    "lentidão no computador",
    "erros no Windows 10/11",
    "configuração de e-mail corporativo",
    "problemas com impressora",
    "recuperação de senha",
    "instalação de software",
    "backup e recuperação de dados",
    "segurança e antivírus",
    "configuração de VPN",
    "problemas com drivers",
    "erros no navegador",
    "configuração de roteador",
    "problemas com memória RAM",
    "erros de disco rígido",
]

SYSTEM_PROMPT = """Você é um especialista em suporte técnico de TI. 
Gere pares de pergunta e resposta no formato JSON para um dataset de fine-tuning.
Cada par deve conter:
- "instruction": uma pergunta ou problema técnico realista de um usuário
- "response": uma resposta detalhada, clara e técnica de um especialista de suporte

Responda APENAS com um array JSON válido contendo os pares, sem texto adicional."""


def generate_batch(topic: str, n: int = 4) -> list[dict]:
    """Gera n pares de instrução/resposta para um tópico."""
    user_prompt = f"""Gere {n} pares distintos de instrução/resposta sobre: {topic}
    
Formato esperado:
[
  {{"instruction": "...", "response": "..."}},
  ...
]

As perguntas devem ser variadas e realistas. As respostas devem ter pelo menos 3 passos práticos."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    content = response.choices[0].message.content.strip()
    # Remove possíveis marcadores de código
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def main():
    all_pairs = []

    print("Gerando dataset sintético de Suporte Técnico de TI...")
    for i, topic in enumerate(DOMAIN_TOPICS):
        print(f"  [{i+1}/{len(DOMAIN_TOPICS)}] Gerando pares para: {topic}")
        try:
            pairs = generate_batch(topic, n=4)
            all_pairs.extend(pairs)
            print(f"    → {len(pairs)} pares gerados. Total: {len(all_pairs)}")
        except Exception as e:
            print(f"    ✗ Erro no tópico '{topic}': {e}")

    print(f"\nTotal de pares gerados: {len(all_pairs)}")

    # Embaralha e divide 90/10
    random.seed(42)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * 0.9)
    train_data = all_pairs[:split_idx]
    test_data = all_pairs[split_idx:]

    print(
        f"Treino: {len(train_data)} exemplos | Teste: {len(test_data)} exemplos")

    # Salva no formato .jsonl
    with open("train_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("test_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n✅ Datasets salvos:")
    print("   → train_dataset.jsonl")
    print("   → test_dataset.jsonl")


if __name__ == "__main__":
    main()
