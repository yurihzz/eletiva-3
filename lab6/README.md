# Laboratório 6 - P2: Construindo um Tokenizador BPE e Explorando o WordPiece

**Versão:** v1.0

---

## Descrição

Este laboratório implementa o motor básico do algoritmo **Byte Pair Encoding (BPE)** do zero e explora o funcionamento do **WordPiece** na prática por meio da biblioteca Hugging Face Transformers.

---

## Estrutura do Projeto

```
lab6_bpe.py   # Código-fonte principal com as 3 tarefas
README.md     # Este arquivo
```

---

## Como Executar

### Pré-requisitos

```bash
pip install transformers
```

### Execução

```bash
python lab6_bpe.py
```

---

## Tarefas Implementadas

### Tarefa 1 — O Motor de Frequências

A função `get_stats(vocab)` recebe o dicionário de vocabulário onde cada chave é uma palavra já segmentada em caracteres separados por espaço (ex: `'n e w e s t </w>'`) e o valor é a frequência dessa palavra no corpus.

A função percorre cada palavra, divide seus símbolos e acumula a frequência de cada par adjacente. Ao ser aplicada ao corpus inicial, o par `('e', 's')` retorna obrigatoriamente a contagem máxima de **9** — resultado da soma das frequências em *newest* (6) e *widest* (3).

**Saída de validação:**
```
Frequência do par ('e', 's'): 9
✓ Validação OK — par ('e', 's') retornou contagem máxima de 9
```

---

### Tarefa 2 — O Loop de Fusão

A função `merge_vocab(pair, v_in)` recebe o par mais frequente e o vocabulário atual, substituindo todas as ocorrências isoladas desse par pela versão unificada (sem espaço) em todas as entradas do dicionário.

O loop principal executa **5 iterações (K=5)**, chamando `get_stats` e `merge_vocab` sucessivamente. A cada rodada, imprime o par fundido e o estado do vocabulário.

**Saída das 5 iterações:**
```
Iteração 1:
  Par mais frequente fundido : ('e', 's')  (freq=9)
  Vocabulário após a fusão   :
    'l o w </w>': 5
    'l o w e r </w>': 2
    'n e w es t </w>': 6
    'w i d es t </w>': 3

Iteração 2:
  Par mais frequente fundido : ('es', 't')  (freq=9)
  Vocabulário após a fusão   :
    'l o w </w>': 5
    'l o w e r </w>': 2
    'n e w est </w>': 6
    'w i d est </w>': 3

Iteração 3:
  Par mais frequente fundido : ('est', '</w>')  (freq=9)
  Vocabulário após a fusão   :
    'l o w </w>': 5
    'l o w e r </w>': 2
    'n e w est</w>': 6
    'w i d est</w>': 3

Iteração 4:
  Par mais frequente fundido : ('l', 'o')  (freq=7)
  Vocabulário após a fusão   :
    'lo w </w>': 5
    'lo w e r </w>': 2
    'n e w est</w>': 6
    'w i d est</w>': 3

Iteração 5:
  Par mais frequente fundido : ('lo', 'w')  (freq=7)
  Vocabulário após a fusão   :
    'low </w>': 5
    'low e r </w>': 2
    'n e w est</w>': 6
    'w i d est</w>': 3
```

**Validação:** Após as 5 iterações é possível observar claramente a formação do token morfológico `est</w>`, confirmando que o BPE identifica e consolida sufixos frequentes de forma automática.

---

### Tarefa 3 — Integração Industrial e WordPiece

Utiliza o tokenizador `bert-base-multilingual-cased` do Hugging Face, que implementa o algoritmo **WordPiece**, para segmentar a frase:

> *"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."*

**Saída esperada:**
```
Tokens gerados pelo WordPiece (bert-base-multilingual-cased):
  ['Os', 'hi', '##per', '-', 'par', '##â', '##metro', '##s', 'do',
   'transform', '##er', 'são', 'inc', '##ons', '##titu', '##cion',
   '##al', '##mente', 'dif', '##íce', '##is', 'de', 'ajust', '##ar', '.']

Total de tokens: 25
```

---

## Sobre o Sinal `##` no WordPiece

Os tokens precedidos por `##` (cerquilha dupla) indicam **sub-palavras que são continuação de uma palavra maior** — ou seja, não iniciam uma nova palavra no texto original. Por exemplo, ao tokenizar *"inconstitucionalmente"*, o WordPiece a decompõe em `inc`, `##ons`, `##titu`, `##cion`, `##al`, `##mente`, onde apenas `inc` representa o início da palavra e todos os fragmentos seguintes carregam o prefixo `##` para sinalizar que são sufixos contínuos.

Esse mecanismo é fundamental porque **impede o travamento do modelo diante de vocabulário desconhecido (palavras OOV — *out-of-vocabulary*)**: em vez de substituir uma palavra rara por um único token genérico `[UNK]`, o tokenizador decompõe a palavra em sub-palavras que já existem no vocabulário. Assim, mesmo que o modelo nunca tenha visto *"inconstitucionalmente"* no treinamento, ele consegue processar seus componentes morfológicos (`inc`, `##ons`, `##titu`, `##cion`, `##al`, `##mente`) e construir uma representação vetorial útil, mantendo a capacidade de generalização para termos novos, compostos e palavras com variações morfológicas.

---

## Declaração de Uso de IA Generativa

Conforme exigido pelas instruções de entrega, declaro que **trechos do código foram elaborados com auxílio de IA generativa (Claude, da Anthropic)** e revisados pelo autor:

| Trecho | Arquivo | Descrição |
|--------|---------|-----------|
| Expressão regular na função `merge_vocab` | `lab6_bpe.py` (linhas ~80–85) | A lógica dos lookarounds `(?<!\S)` e `(?!\S)` para garantir que apenas pares de tokens isolados sejam substituídos foi sugerida pela IA e verificada manualmente contra os casos de teste. |

Todos os demais trechos — incluindo `get_stats`, o loop principal e a análise da Tarefa 3 — foram escritos e compreendidos pelo autor.

---

## Versionamento

A versão final avaliada está marcada com a tag **`v1.0`**:

```bash
git tag v1.0
git push origin v1.0
```
