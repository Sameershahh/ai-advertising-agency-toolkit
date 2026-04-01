# Task 3 — Section 3: Speed & Practical Tasks

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange)

## Overview

This module contains all 5 Section 3 speed tasks, implemented as complete, production-ready functions and documented analyses in a single importable Python module.

---

## Task Summary

| Task | Topic | Implementation |
|------|-------|----------------|
| Q1 | Anthropic API with retry on rate limit | `call_anthropic_with_retry()` function |
| Q2 | Debug broken LangChain RAG pipeline | 3 bugs identified, documented, and fixed |
| Q3 | Brand tone enforcer system prompt | Full system prompt + `enforce_brand_tone()` function |
| Q4 | Brand safety evaluation of 3 images | Written analysis with scored rubric |
| Q5 | AI ad personalization engine architecture | 8-layer system design with throughput specs |

---

## Q1 — Anthropic API with Retry Logic

**Goal:** Python function calling Anthropic API, retrying up to 3 times on rate limit errors.

```python
from section3_tasks import call_anthropic_with_retry

result = call_anthropic_with_retry(
    prompt="Write a tagline for a luxury watch brand.",
    system="You are a senior advertising copywriter.",
    max_retries=3
)
print(result)
```

**Implementation highlights:**
- Retries on `RateLimitError` (429), `APIStatusError` (5xx), and `APIConnectionError`
- Exponential back-off: delays of 2s, 4s, 8s between attempts
- Non-retryable errors (4xx other than 429) raise immediately
- Clean `RuntimeError` after all retries exhausted

---

## Q2 — Debugged LangChain RAG Pipeline

Three bugs identified and fixed in a provided broken pipeline:

### Bug 1 — Wrong input type for Chroma
```python
# BROKEN: Passes raw strings to from_documents()
vectorstore = Chroma.from_documents(docs, embeddings)

# FIXED: Wrap in LangChain Document objects first
from langchain.schema import Document
lc_docs = [Document(page_content=d) for d in docs]
vectorstore = Chroma.from_documents(lc_docs, embeddings)
```

### Bug 2 — Missing required chain_type argument
```python
# BROKEN: Missing chain_type → TypeError
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, ...)

# FIXED: Add required chain_type
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # ← required argument
    retriever=retriever,
    return_source_documents=True
)
```

### Bug 3 — Deprecated direct callable interface
```python
# BROKEN: chain(question) removed in LangChain ≥0.1.x
result = chain(question)

# FIXED: Use invoke() with proper key
result = chain.invoke({"query": question})
```

---

## Q3 — Brand Tone Enforcer System Prompt

System prompt designed for Verde Beauty — premium organic skincare brand.

**Run live test:**
```bash
python -c "
from section3_tasks import enforce_brand_tone
result = enforce_brand_tone(\"This is literally the BEST skincare — it TRANSFORMS your skin overnight!!\")
print(result)
"
```

**Key elements of the system prompt:**
- Explicit forbidden language list with examples
- Tonal calibration with wrong/right example pairs
- Structured output format: violations → rewrite → changes made
- Evidence-based, authoritative brand voice guidelines

---

## Q4 — Brand Safety Evaluation

Three AI-generated advertising images evaluated:

| Image | Score | Key Issue |
|-------|-------|-----------|
| Luxury perfume bottle with smoke | 9/10 | Minor: smoke may be tobacco-associated in conservative markets |
| Rooftop party with cocktails | 6/10 | Alcohol visible — age-verified placements only |
| Before/after skin transformation | 3/10 | FDA violation — cannot run without legal review |

Full reasoning and placement restrictions documented in `section3_tasks.py`.

---

## Q5 — AI Ad Personalization Engine Architecture

8-layer system architecture for a major e-commerce client:

1. **Data Ingestion** — Kafka + CRM sync + real-time context API
2. **Feature Store** — Redis (online, <10ms) + BigQuery (offline) + Flink
3. **AI/ML Layer** — Two-tower embeddings, LightGBM CTR, Claude copy, CLIP visuals
4. **Personalization Orchestrator** — <250ms P95 end-to-end latency
5. **Ad Server Integration** — OpenRTB 2.6, DV360, TTD, Meta Advantage+
6. **Brand Safety Layer** — Claude Vision + LLM compliance checker
7. **Feedback Loop** — Thompson Sampling, weekly retraining on Vertex AI
8. **Infrastructure** — GKE Kubernetes, Cloud Armor, 99.95% SLA

**Key throughput specs:**
- 50,000 bid requests/second at peak
- 3M unique users/day
- 15M personalized impressions/day

---

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-api-key"
python section3_tasks.py
```

For Q2 LangChain functions (optional):
```bash
pip install langchain langchain-openai langchain-community chromadb
```

---

## Repository Structure

```
task3-speed-practical-tasks/
├── section3_tasks.py    # All 5 tasks implemented
└── README.md
```

---

## Assumptions

- Q1: `ANTHROPIC_API_KEY` environment variable is set
- Q2: LangChain dependencies are optional — Q2 can be reviewed as a documented analysis without installing LangChain
- Q3: Verde Beauty brand guidelines are defined within the system prompt itself
- Q4: Image evaluations are based on provided image descriptions (no actual images required for scoring)
- Q5: Architecture is described as a design specification — no running infrastructure required
