# Laboratório 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

> Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Yuri Estrela.

---

## 1. Visão Geral

Pipeline de **Retrieval-Augmented Generation (RAG)** de nível de produção para busca em manuais médicos. O sistema supera a falha da similaridade de cosseno pura ao combinar três técnicas complementares:

| Etapa | Técnica | Papel |
|---|---|---|
| 1 | **HNSW** (Hierarchical Navigable Small World) | Indexação e busca rápida em espaço vetorial |
| 2 | **HyDE** (Hypothetical Document Embeddings) | Transformação da query coloquial em âncora técnica |
| 3 | **Bi-Encoder** | Recuperação rápida dos Top-10 candidatos |
| 4 | **Cross-Encoder** | Re-ranking preciso dos Top-3 finais |

---

## 2. Estrutura do Projeto

```
lab09-rag/
├── rag_pipeline.py   # Pipeline completo (4 passos)
├── requirements.txt  # Dependências Python
└── README.md         # Este arquivo
```

---

## 3. Como Executar

### 3.1 Instalar dependências

```bash
pip install -r requirements.txt
```

### 3.2 Configurar a chave da API Anthropic

```bash
export ANTHROPIC_API_KEY="sua-chave-aqui"
```

### 3.3 Rodar o pipeline

```bash
python rag_pipeline.py
```

A query padrão é `"dor de cabeça latejante e luz incomodando"`.  
Para alterar, edite a variável `USER_QUERY` no bloco `if __name__ == "__main__"`.

---

## 4. Tarefa Analítica — HNSW vs KNN: Impacto na Memória RAM

### 4.1 KNN Exato (baseline)

A busca K-Nearest Neighbors exata armazena **todos os N vetores de dimensão D** em uma matriz plana:

```
RAM_KNN = N × D × 4 bytes   (float32)
```

Para 1 milhão de documentos com embeddings de 384 dimensões:
```
RAM_KNN = 1.000.000 × 384 × 4 = ~1,5 GB apenas para os vetores
```

Além disso, **cada busca percorre toda a base** — custo O(N·D) —, tornando-se inviável acima de alguns milhões de vetores.

---

### 4.2 HNSW — Estrutura em Grafo Hierárquico

O HNSW organiza os vetores em **camadas de grafos navegáveis de mundo pequeno**. A camada superior é esparsa (poucos nós de alta conectividade) e a inferior densa (todos os nós). A busca parte do topo e desce até a camada 0 de forma gulosa, atingindo complexidade **O(log N)**.

**Custo de memória adicional** do grafo:

```
RAM_grafo ≈ N × M × 2 × 8 bytes   (ponteiros de 64 bits, ida e volta)
```

Com N = 1.000.000 e M = 32:
```
RAM_grafo ≈ 1.000.000 × 32 × 2 × 8 = ~512 MB
```

Somando ao custo base dos vetores (≈1,5 GB), o total fica em **~2 GB** — mais caro que o KNN em RAM pura, mas com velocidade de busca incomparavelmente superior.

---

### 4.3 O Papel de M e ef_construction

| Hiperparâmetro | O que controla | Efeito na RAM | Efeito na Qualidade |
|---|---|---|---|
| **M** | Número de conexões bidirecionais por nó | Cresce linearmente com M (RAM ∝ N × M) | Recall maior com M maior |
| **ef_construction** | Tamanho da fila de candidatos durante a construção do grafo | **Não afeta RAM diretamente** | Qualidade do grafo: ef_construction maior → vizinhos melhores → maior recall |

**Conclusão prática:**
- Aumentar **M** de 16 para 64 dobra o custo do grafo em RAM, mas melhora o recall de ~95% para ~99%.
- Aumentar **ef_construction** de 100 para 400 torna a **construção** do índice mais lenta (e usa mais RAM temporária durante o build), mas **não aumenta a RAM do índice em produção**.
- Para servidores com memória limitada, o trade-off recomendado é **M=16–32** com **ef_construction=100–200**.

---

### 4.4 Resumo Comparativo

| Critério | KNN Exato | HNSW |
|---|---|---|
| RAM base (vetores) | N × D × 4 B | N × D × 4 B |
| RAM adicional (índice) | Nenhuma | N × M × 2 × 8 B |
| Velocidade de busca | O(N·D) — lento | O(log N) — rápido |
| Recall | 100% (exato) | ~95–99% (aproximado) |
| Escalabilidade | Inviável em >10M docs | Produção em bilhões de docs |

---

## 5. Fluxo do Pipeline

```
Query coloquial do usuário
        │
        ▼
┌─────────────────────────┐
│  PASSO 2: HyDE          │  LLM "alucina" documento técnico
│  (Transformação da Query)│  com jargão médico equivalente
└──────────┬──────────────┘
           │  vetor do doc hipotético
           ▼
┌─────────────────────────┐
│  PASSO 3: Bi-Encoder    │  Similaridade de cosseno no
│  + Índice HNSW          │  grafo → Top-10 candidatos
└──────────┬──────────────┘
           │  10 documentos candidatos
           ▼
┌─────────────────────────┐
│  PASSO 4: Cross-Encoder │  Atenção profunda (query, doc)
│  Re-ranking             │  → Top-3 documentos finais
└──────────┬──────────────┘
           │
           ▼
   Contexto injetado no LLM
   para geração da resposta final
```

---

## 6. Decisões de Implementação

- **Modelo de Embedding:** `sentence-transformers/all-MiniLM-L6-v2` — leve (384 dims), sem necessidade de chave de API externa, qualidade excelente para triagem semântica.
- **Índice HNSW:** `faiss.IndexHNSWFlat` com `METRIC_INNER_PRODUCT` (equivalente a cosseno com vetores L2-normalizados).
- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — treinado em pares (query, passagem) para ranking de relevância.
- **LLM para HyDE:** Claude Sonnet via API Anthropic, que gera documentos hipotéticos em português com terminologia clínica.

---

## 7. Dependências

```
anthropic>=0.40.0          # Cliente oficial da API Anthropic
faiss-cpu>=1.8.0           # Índice HNSW eficiente
sentence-transformers>=3.0.0  # Bi-Encoder + Cross-Encoder
numpy>=1.26.0              # Operações vetoriais
```
