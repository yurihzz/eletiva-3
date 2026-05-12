"""
Laboratório 09: Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)
Pipeline de Retrieval-Augmented Generation para busca em manuais médicos.

Partes deste laboratório foram geradas/complementadas com IA,
revisadas e validadas por Yuri Estrela.
"""

import os
import json
import numpy as np
import faiss
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─────────────────────────────────────────────
# BASE DE DADOS: 20 fragmentos de manuais médicos
# ─────────────────────────────────────────────
MEDICAL_DOCUMENTS = [
    # Neurologia
    "Cefaleia pulsátil (migrânea) caracteriza-se por dor unilateral, latejante, de intensidade moderada a grave, "
    "frequentemente acompanhada de náuseas, vômitos, fotofobia e fonofobia. O tratamento agudo inclui triptanos "
    "e AINEs; a profilaxia utiliza betabloqueadores e antidepressivos tricíclicos.",

    "A fotofobia é a hipersensibilidade anormal à luz, comum em quadros de migrânea, meningite e uveíte. "
    "Neuronalmente, ocorre por ativação do nervo trigêmeo e projeções melanopsínicas da retina ao tálamo.",

    "Acidente Vascular Cerebral Isquêmico (AVCI): bloqueio súbito do fluxo sanguíneo cerebral por trombo ou êmbolo. "
    "Apresenta-se com hemiparesia, afasia, desvio do olhar conjugado e alteração do nível de consciência. "
    "Janela terapêutica para trombólise: até 4,5 horas do início dos sintomas.",

    "Epilepsia focal com generalização secundária: crise que inicia em um foco cortical e se propaga bilateralmente. "
    "EEG demonstra descarga epileptiforme focal evoluindo para padrão generalizado. "
    "Tratamento: carbamazepina, oxcarbazepina ou lacosamida.",

    "Esclerose Múltipla (EM): doença desmielinizante autoimune do SNC. Cursa com neurite óptica, parestesias, "
    "fraqueza muscular e sintomas cognitivos. Diagnóstico por critérios de McDonald (RM + LCR). "
    "Tratamento: interferons, acetato de glatirâmer, natalizumabe.",

    # Cardiologia
    "Infarto Agudo do Miocárdio com Supradesnivelamento do Segmento ST (IAMCSST): oclusão total de artéria coronária. "
    "ECG: supradesnivelamento ≥1 mm em ≥2 derivações contíguas. Conduta: reperfusão por angioplastia primária "
    "em até 90 minutos ou trombólise em até 12 horas.",

    "Insuficiência Cardíaca com Fração de Ejeção Reduzida (ICFEr): FE <40%. Apresenta dispneia, ortopneia, "
    "edema de membros inferiores e terceira bulha. Tratamento padrão: IECA/BRA, betabloqueadores, "
    "antagonistas da aldosterona e SGLT2i.",

    "Fibrilação Atrial: arritmia supraventricular caracterizada por ativação atrial caótica, sem onda P definida. "
    "RVM irregularmente irregular. Risco embólico avaliado pelo escore CHA2DS2-VASc. "
    "Anticoagulação com NOACs (rivaroxabana, apixabana) ou warfarina.",

    # Pneumologia
    "Pneumonia Adquirida na Comunidade (PAC): consolidação do parênquima pulmonar por agentes infecciosos. "
    "Agentes mais comuns: S. pneumoniae, H. influenzae, M. pneumoniae. Gravidade pelo escore CURB-65. "
    "Antibioticoterapia: amoxicilina ou macrolídeo em casos leves; beta-lactâmico + macrolídeo em graves.",

    "Doença Pulmonar Obstrutiva Crônica (DPOC): obstrução crônica ao fluxo aéreo, pouco reversível. "
    "VEF1/CVF <0,70 pós broncodilatador. Exacerbações tratadas com broncodilatadores de curta ação, "
    "corticoides sistêmicos e antibióticos. Estadiamento GOLD I–IV.",

    "Embolia Pulmonar (EP): obstrução das artérias pulmonares por trombos, geralmente originados nas veias profundas. "
    "Apresenta dispneia súbita, taquicardia, dor pleurítica e hemoptise. Diagnóstico por angiotomografia. "
    "Anticoagulação plena imediata; trombólise em EP maciça.",

    # Gastroenterologia
    "Doença de Crohn: inflamação transmural e granulomatosa do trato gastrointestinal, com predileção pelo íleo terminal. "
    "Manifesta-se com dor abdominal, diarreia, perda de peso e fístulas. Tratamento: corticoides, azatioprina, "
    "metotrexato e agentes biológicos anti-TNF.",

    "Cirrose Hepática: fibrose avançada do fígado com remodelamento nodular. Causas: álcool, hepatite B/C, DHGNA. "
    "Complicações: hipertensão portal, ascite, encefalopatia hepática, peritonite bacteriana espontânea. "
    "Avaliação: escore Child-Pugh e MELD.",

    "Úlcera Péptica: perda de substância da mucosa gástrica ou duodenal. Associada ao H. pylori (70–90%) e AINEs. "
    "Sintoma cardinal: dor epigástrica em queimação, aliviada por alimentos (duodenal) ou piorada por eles (gástrica). "
    "Tratamento: IBP + esquema de erradicação do H. pylori (amoxicilina + claritromicina).",

    # Endocrinologia
    "Diabetes Mellitus Tipo 2: resistência insulínica e disfunção progressiva das células beta. "
    "Diagnóstico: glicemia de jejum ≥126 mg/dL, HbA1c ≥6,5% ou TOTG ≥200 mg/dL. "
    "Metas: HbA1c <7%. Tratamento: metformina de primeira linha, acrescida de SGLT2i, GLP-1 RA ou insulina.",

    "Hipotireoidismo Primário: deficiência de hormônios tireoidianos por falência da glândula. "
    "Causa mais comum: tireoidite de Hashimoto (autoimune). TSH elevado, T4 livre baixo. "
    "Sintomas: fadiga, intolerância ao frio, ganho de peso, constipação, bradicardia. "
    "Tratamento: levotiroxina sódica.",

    "Síndrome de Cushing: hipercortisolismo endógeno, mais frequentemente por adenoma hipofisário (doença de Cushing). "
    "Apresenta obesidade centrípeta, estrias violáceas, hipertensão, osteoporose e diabetes. "
    "Diagnóstico: cortisol livre urinário de 24h, teste de supressão com dexametasona.",

    # Nefrologia
    "Insuficiência Renal Aguda (IRA): redução abrupta da função renal, com elevação de creatinina ≥0,3 mg/dL em 48h "
    "ou ≥50% em 7 dias. Classificada em pré-renal, renal (intrínseca) e pós-renal. "
    "Tratamento: correção da causa base, balanço hídrico e suporte dialítico se necessário.",

    "Síndrome Nefrótica: proteinúria maciça >3,5 g/24h, hipoalbuminemia, edema e hiperlipidemia. "
    "Causas primárias: doença por lesão mínima, glomeruloesclerose focal e segmentar. "
    "Tratamento: corticoides, imunossupressores e IECA para redução da proteinúria.",

    # Infectologia
    "Sepse: disfunção orgânica com risco de vida causada por resposta desregulada do hospedeiro à infecção. "
    "Critérios: aumento de ≥2 pontos no SOFA. Manejo: bundle da hora de ouro — hemoculturas, antibioticoterapia "
    "ampla em até 1 hora, ressuscitação volêmica com 30 mL/kg de cristaloide e vasopressores se PAM <65 mmHg.",
]

# ─────────────────────────────────────────────
# PASSO 1: CONSTRUÇÃO E INDEXAÇÃO DO GRAFO HNSW
# ─────────────────────────────────────────────

def build_hnsw_index(documents: list[str], model: SentenceTransformer) -> tuple[faiss.IndexHNSWFlat, np.ndarray]:
    """
    Constrói o índice HNSW com FAISS.

    Hiperparâmetros HNSW:
      - M (32): número de conexões bidirecionais por nó no grafo.
        Valores maiores → maior precisão de recall e maior consumo de RAM.
      - ef_construction (200): tamanho da lista dinâmica durante a construção.
        Valores maiores → índice mais preciso, mas construção mais lenta.

    Comparação de RAM com KNN exato:
      KNN exato armazena todos os vetores e calcula distância com todos em O(N·D).
      HNSW usa O(N · M · log(N)) memória adicional para o grafo, mas reduz a
      busca a O(log N) — essencial para grandes bases de dados.
    """
    print("=" * 60)
    print("PASSO 1: Construindo embeddings e índice HNSW...")
    print("=" * 60)

    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]

    # Parâmetros HNSW: M=32, ef_construction=200
    M = 32
    index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64   # qualidade da busca em tempo de execução

    index.add(embeddings)

    print(f"\n✅ Índice HNSW criado com sucesso!")
    print(f"   Dimensão dos vetores : {dimension}")
    print(f"   Documentos indexados : {index.ntotal}")
    print(f"   M (conexões)         : {M}")
    print(f"   ef_construction      : 200")
    print(f"   ef_search            : 64\n")

    return index, embeddings


# ─────────────────────────────────────────────
# PASSO 2: QUERY TRANSFORMATION (HyDE)
# ─────────────────────────────────────────────

def generate_hypothetical_document(query: str, client: Anthropic) -> str:
    """
    HyDE — Hypothetical Document Embeddings.
    Pede ao LLM que 'alucine' um trecho técnico de manual médico
    como se ele já contivesse a resposta ideal para a query do usuário.
    O embedding desse documento hipotético serve de âncora no espaço vetorial,
    aproximando-se do jargão técnico dos documentos reais indexados.
    """
    print("=" * 60)
    print("PASSO 2: Gerando Documento Hipotético (HyDE)...")
    print("=" * 60)
    print(f"   Query do usuário: \"{query}\"\n")

    system_prompt = (
        "Você é um redator especializado em manuais médicos técnicos. "
        "Sua tarefa é escrever um parágrafo de 4 a 6 linhas no estilo de um manual clínico, "
        "usando terminologia médica precisa (jargões técnicos, epônimos, siglas clínicas), "
        "como se esse parágrafo fosse o trecho exato de um manual que responde à pergunta do paciente. "
        "Não faça introduções; escreva diretamente o conteúdo técnico."
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )

    hypothetical_doc = response.content[0].text.strip()
    print(f"   Documento hipotético gerado:\n   \"{hypothetical_doc[:200]}...\"\n")
    return hypothetical_doc


# ─────────────────────────────────────────────
# PASSO 3: BUSCA RÁPIDA VIA BI-ENCODER (HNSW)
# ─────────────────────────────────────────────

def retrieve_top_k(
    hypothetical_doc: str,
    index: faiss.IndexHNSWFlat,
    documents: list[str],
    model: SentenceTransformer,
    k: int = 10,
) -> list[tuple[int, float, str]]:
    """
    Vetoriza o documento hipotético e busca os Top-K mais similares
    no índice HNSW usando similaridade de cosseno (produto interno com vetores normalizados).
    """
    print("=" * 60)
    print(f"PASSO 3: Buscando Top-{k} via Bi-Encoder + HNSW...")
    print("=" * 60)

    query_vector = model.encode([hypothetical_doc], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vector, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        results.append((idx, float(score), documents[idx]))
        print(f"   [{rank:02d}] Score={score:.4f} | {documents[idx][:80]}...")

    print()
    return results


# ─────────────────────────────────────────────
# PASSO 4: RE-RANKING COM CROSS-ENCODER
# ─────────────────────────────────────────────

def rerank_with_cross_encoder(
    original_query: str,
    candidates: list[tuple[int, float, str]],
    cross_encoder: CrossEncoder,
    top_n: int = 3,
) -> list[dict]:
    """
    Aplica o Cross-Encoder (atenção profunda entre query e documento).
    Diferente do Bi-Encoder, o Cross-Encoder processa o par (query, doc)
    de forma conjunta, capturando interações token a token — muito mais preciso,
    porém computacionalmente caro (usado apenas para re-ranquear candidatos pré-selecionados).
    """
    print("=" * 60)
    print(f"PASSO 4: Re-ranking com Cross-Encoder (Top-{top_n})...")
    print("=" * 60)

    pairs = [(original_query, doc_text) for (_, _, doc_text) in candidates]
    cross_scores = cross_encoder.predict(pairs)

    reranked = sorted(
        [
            {"original_idx": idx, "bi_score": bi_score, "cross_score": float(cs), "text": text}
            for (idx, bi_score, text), cs in zip(candidates, cross_scores)
        ],
        key=lambda x: x["cross_score"],
        reverse=True,
    )

    top_results = reranked[:top_n]

    print(f"\n   ✅ Top-{top_n} documentos após Re-ranking:\n")
    for rank, doc in enumerate(top_results, start=1):
        print(f"   ┌─ Posição #{rank}")
        print(f"   │  Cross-Score : {doc['cross_score']:.4f}")
        print(f"   │  Bi-Score    : {doc['bi_score']:.4f}")
        print(f"   │  Texto       : {doc['text'][:120]}...")
        print(f"   └{'─' * 55}")
        print()

    return top_results


# ─────────────────────────────────────────────
# PIPELINE COMPLETO
# ─────────────────────────────────────────────

def run_rag_pipeline(query: str):
    """Executa o pipeline RAG completo: HNSW → HyDE → Bi-Encoder → Cross-Encoder."""

    print("\n" + "█" * 60)
    print("  PIPELINE RAG AVANÇADO — LABORATÓRIO 09")
    print("█" * 60 + "\n")

    # Carrega modelos
    print("🔄 Carregando modelos...\n")
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Inicializa cliente Anthropic
    client = Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    # Passo 1: Construir índice HNSW
    index, _ = build_hnsw_index(MEDICAL_DOCUMENTS, bi_encoder)

    # Passo 2: HyDE — gerar documento hipotético
    hypothetical_doc = generate_hypothetical_document(query, client)

    # Passo 3: Recuperação rápida (Top-10)
    candidates = retrieve_top_k(hypothetical_doc, index, MEDICAL_DOCUMENTS, bi_encoder, k=10)

    # Passo 4: Re-ranking preciso com Cross-Encoder (Top-3)
    final_docs = rerank_with_cross_encoder(query, candidates, cross_encoder, top_n=3)

    # Resumo final
    print("=" * 60)
    print("RESULTADO FINAL — Documentos para injetar no LLM:")
    print("=" * 60)
    context_for_llm = "\n\n---\n\n".join(
        [f"[Documento {i+1}]\n{doc['text']}" for i, doc in enumerate(final_docs)]
    )
    print(context_for_llm)
    print("\n" + "█" * 60)

    return final_docs, context_for_llm


# ─────────────────────────────────────────────
# PONTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Query coloquial do paciente (exemplo do enunciado)
    USER_QUERY = "dor de cabeça latejante e luz incomodando"

    final_docs, context = run_rag_pipeline(USER_QUERY)
