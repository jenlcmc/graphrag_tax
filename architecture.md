# Architecture

Hybrid GraphRAG system for federal income tax Q&A.  The system pairs a
NetworkX knowledge graph with a FAISS vector index so that retrieval can
follow cross-publication reference chains (graph traversal) and semantic
similarity (vector search) at the same time.

---

## Full Pipeline

```mermaid
flowchart TD
    subgraph Ph1["Phase 1 — Sources (knowledge/)"]
        USC[usc26.xml\nTitle 26 IRC\nSubtitle A]
        IRS[36 IRS XML publications\np17, p596, i1040gi, p501, ...]
    end

    subgraph Ph2["Phase 2 — Processing (src/ingestion + src/preprocessing)"]
        P1[usc26_parser.py\nextract Subtitle A sections\nnormalize ref elements]
        P2[irs_xml_parser.py\nauto-detect tipx / instrx schema\nextract section text]
        NRM[normalizer.py\nextract cross-refs via spaCy regex\nIRS pub mentions + § citations]
        CHK[chunker.py\nsplit at sentence boundaries\nif text > MAX_CHUNK_CHARS]
    end

    subgraph Ph3["Phase 3 — Indexing (src/indexing)"]
        VI[vector_index.py\nFAISS IndexFlatIP\ncontent + section_id embeddings]
        GI[graph_index.py\nNetworkX DiGraph\nthree edge types]
        SL[section_linker.py\ninject coverage edges\ncross-publication edges]
    end

    subgraph Ph4["Phase 4 — Retrieval (src/retrieval)"]
        VR[vector_retriever.py\ncosine similarity]
        GR[graph_retriever.py\nBFS traversal]
        HR[hybrid_retriever.py\nmerge + re-rank\nalpha × vector + 1-alpha × graph]
    end

    subgraph Ph5["Phase 5 — Generation"]
        LLM[Claude / Gemini\nwith retrieved context\nas system prompt]
        ANS[Source-grounded answer\nwith IRC citations]
    end

    USC --> P1
    IRS --> P2
    P1 --> NRM
    P2 --> NRM
    NRM --> CHK
    CHK --> VI
    CHK --> GI
    GI --> SL
    VI --> VR
    SL --> GR
    VR --> HR
    GR --> HR
    HR --> LLM
    LLM --> ANS
```

---

## Knowledge Graph: Three Edge Types

```mermaid
flowchart LR
    subgraph USC["26 USC (statute)"]
        S32["26 USC §32"]
        S32a["26 USC §32(a)"]
        S32c["26 USC §32(c)"]
    end

    subgraph IRS["IRS Publications"]
        P17["Pub 17\n(overview)"]
        P596["Pub 596\n(EIC guide)"]
        EIC["Schedule EIC\nInstructions"]
    end

    S32 -->|hierarchy| S32a
    S32 -->|hierarchy| S32c
    P17 -->|coverage| S32
    P596 -->|coverage| S32
    P17 -->|xref| P596
    P596 -->|xref| EIC
```

**Why three edge types matter:**

A flat vector index might retrieve Pub 17 for a query about the EIC — but it
has no mechanism to follow the Pub 17 → Pub 596 → §32 chain.  Explicit edge
types let BFS traversal recover all three nodes in two hops, so the LLM
receives the statutory text, the plain-language guide, and the form
instructions together.

| Edge type   | Source             | Captures                                      |
| ----------- | ------------------ | --------------------------------------------- |
| `hierarchy` | XML tree structure | Parent section → child subsection             |
| `xref`      | `<ref>` tags + NLP | Explicit §NNN citations and pub name mentions |
| `coverage`  | section_linker.py  | Curated IRS pub ↔ IRC section relationships   |

---

## Retrieval Flow

```mermaid
flowchart LR
    Q[User question] --> VR[Vector retriever\ntop-k by cosine similarity]
    Q --> GR[Graph retriever\nBFS from entry nodes]

    subgraph GR_detail["Graph retrieval detail"]
        direction TB
        E1[Find entry nodes\n§ refs + keyword hints]
        E2[Expand BFS depth=2\nacross hierarchy/xref/coverage]
        E3[Community summaries\nfor broad queries]
        E1 --> E2
        E3 --> E2
    end

    GR --- GR_detail

    VR --> M[hybrid_retriever.py\nmerge by section_id\nscore = 0.6×vector + 0.4×graph]
    GR --> M
    M --> K[Top-k ranked chunks]
    K --> SYS[System prompt\nwith retrieved excerpts]
    SYS --> LLM[LLM]
    LLM --> A[Answer with citations]
```

---

## Evaluation Design

Four retrieval conditions are run for each LLM.  Comparing `none` vs
`hybrid` isolates the contribution of GraphRAG.

```
                 mode=none   mode=vector   mode=graph   mode=hybrid
claude-sonnet       ✓            ✓             ✓             ✓
gemini-2.5-pro      ✓            ✓             ✓             ✓
```

**Metric:** rubric-based LLM-as-judge score (points earned / points possible).
The judge receives the scoring rubric and the model response, and assigns
partial credit for each criterion.

---

## File Interaction Map

```
scripts/build_pipeline.py
  calls  src/ingestion/usc26_parser.py
  calls  src/ingestion/irs_xml_parser.py
  calls  src/preprocessing/normalizer.py
  calls  src/preprocessing/chunker.py
  calls  src/indexing/vector_index.py   → data/processed/vector_*.faiss + vector_meta.json
  calls  src/indexing/graph_index.py    → data/processed/graph.graphml + communities.json
    calls  src/indexing/section_linker.py  (called by graph_index.py internally)
  calls  src/indexing/graph_audit.py    → data/processed/graph_audit.json

chatbot.py
  calls  src/retrieval/hybrid_retriever.py
    calls  src/retrieval/vector_retriever.py   (loads FAISS indexes)
    calls  src/retrieval/graph_retriever.py    (loads graph.graphml)
  calls  Claude API / Gemini API

evaluation/run_eval.py
  calls  src/retrieval/hybrid_retriever.py
  calls  Claude API / Gemini API  (generation + judge)
  calls  evaluation/datasets/<name>.py
```
