# Task 2 — AI System Design & API Development

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

## Overview

This repository contains three production-ready AI systems:

| Sub-task | Component | Description |
|----------|-----------|-------------|
| 2.1 | Campaign Brief Analyzer | FastAPI REST API — analyzes campaign briefs |
| 2.2 | Image Auto-Tagger | Batch vision analysis with brand safety scoring |
| 2.3 | RAG Knowledge Bot | Document-grounded Q&A chatbot |

---

## Task 2.1 — AI Campaign Brief Analyzer

### Features

- `POST /analyze-brief` — accepts JSON body, returns structured analysis
- `POST /analyze-pdf` — (bonus) accepts PDF upload, extracts and analyzes text
- SSE streaming support via `"stream": true` flag
- Full OpenAPI docs at `/docs`
- Docker-ready with Dockerfile included

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze-brief` | Analyze brief from JSON text |
| POST | `/analyze-pdf` | Analyze brief from PDF upload |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

### Setup & Run

**Local:**
```bash
cd campaign_analyzer
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"
uvicorn main:app --reload --port 8000
```

**Docker:**
```bash
docker build -t campaign-analyzer .
docker run -e ANTHROPIC_API_KEY="your-key" -p 8000:8000 campaign-analyzer
```

### Example Request

```bash
curl -X POST http://localhost:8000/analyze-brief \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "brief_text": "We are launching Verde Beauty — a premium organic skincare line targeting women aged 28-44, household income $80k+. Campaign goal: 25% uplift in repeat purchase rate. Tone: confident, transparent. Channels: Instagram, Pinterest, Google Display.",
  "stream": false
}
EOF
```

### Example Response

```json
{
  "status": "success",
  "model": "claude-opus-4-5",
  "processing_time_ms": 1840,
  "analysis": {
    "campaign_objective": "Achieve a 25% uplift in repeat purchase rate within 6 months of launch",
    "target_audience": {
      "primary": "Women aged 28-44, HHI $80k+, eco-conscious, skeptical of greenwashing",
      "secondary": null
    },
    "key_messages": [
      "USDA Certified Organic — not just 'clean' or 'natural'",
      "Ingredients traceable to farm of origin",
      "Premium performance without compromise"
    ],
    "tone": "confident, transparent, quietly authoritative",
    "channels": ["Instagram", "Pinterest", "Google Display"],
    "risks": [
      "Budget not specified — may constrain channel mix",
      "No influencer vs UGC strategy confirmed"
    ],
    "one_line_brief": "Organic authority brand driving repeat purchase growth"
  }
}
```

### Architecture

- **FastAPI** with Pydantic v2 for schema validation and automatic OpenAPI generation
- **Anthropic Claude** with temperature 0.3 (low = deterministic, consistent analytical outputs)
- **SSE streaming** via FastAPI `StreamingResponse` for real-time progressive delivery
- **PDF support** via PyMuPDF with 10MB upload limit and graceful error handling
- **Global exception handler** ensures all errors return structured JSON

---

## Task 2.2 — AI Image Description & Auto-Tagging

### Features

- Batch processes all images in a folder in a single run
- No manual input required per image
- Generates: `alt_text`, `tags[]`, `brand_safety_score` (1-10), `use_cases[]`
- Handles base64 encoding for all supported vision formats
- Graceful error handling for unsupported formats and API failures

### Setup & Run

```bash
cd image_tagger
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"

# Create an images folder and add your test images
mkdir images
cp /path/to/test/images/* images/

python image_tagger.py --folder ./images --output ./tags_output.json
```

### Output Format

```json
{
  "images": [
    {
      "filename": "perfume_hero.jpg",
      "alt_text": "Dark glass perfume bottle on black marble surface with atmospheric smoke wisps, dramatic side lighting",
      "tags": ["luxury", "product photography", "dark aesthetic", "minimalist", "premium"],
      "brand_safety_score": 9,
      "use_cases": ["Instagram feed", "Display network premium", "Magazine digital", "OOH digital"]
    }
  ],
  "total": 5
}
```

### Brand Safety Scoring Guide

| Score | Meaning |
|-------|---------|
| 10 | Fully safe — all audiences and placements |
| 7-9 | Minor concerns — restrict in specific markets |
| 4-6 | Moderate concerns — age-verified placements only |
| 1-3 | Unsafe — requires legal review before any placement |

### Supported Formats

`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

---

## Task 2.3 — RAG Campaign Knowledge Bot

### Features

- Answers questions strictly from provided documents — refuses out-of-scope questions
- Returns: answer + source document name + relevant quote
- Supports `.txt` and `.pdf` documents
- Index cached to disk for fast subsequent runs
- No external vector database required (custom cosine similarity over NumPy matrix)

### Stack

- **Anthropic Claude** for grounded generation (temperature 0.1 — near-factual)
- **Voyage-3 embeddings** via Anthropic API (falls back to TF-IDF offline)
- **NumPy cosine similarity** as the retrieval engine (zero external DB dependencies)
- **Pickle cache** for index persistence between sessions

### Setup & Run

```bash
cd rag_bot
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"

# Place your documents in the docs folder
mkdir docs
cp your_docs/*.txt docs/
cp your_docs/*.pdf docs/

python rag_bot.py --docs ./docs
```

### Sample Conversation

```
You: What is the campaign objective for Verde?
Bot:
Answer: The campaign objective is to reestablish Verde as the authentic authority
in organic skincare and drive a 25% uplift in repeat purchase rate among
existing customers within 6 months.
Source: verde_campaign_brief.txt
Quote: "drive a 25% uplift in repeat purchase rate among existing customers
within 6 months of campaign launch"

You: What is the population of France?
Bot:
Answer: I cannot answer this question from the provided documents.
Source: N/A
Quote: N/A
```

### Architecture

```
Documents → Chunker (800 chars, 150 overlap)
         → Voyage-3 Embeddings
         → NumPy Matrix (cached to .rag_index.pkl)
                    ↓
Query → Embed → Cosine Similarity → Top-5 Chunks
                                  → Claude (temp 0.1)
                                  → Answer + Source + Quote
```

---

## Repository Structure

```
task2-ai-system-design/
├── campaign_analyzer/
│   ├── main.py                # FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── sample_brief.txt       # 500-word demo brief
├── image_tagger/
│   ├── image_tagger.py        # Batch image analysis script
│   ├── requirements.txt
│   └── images/                # Place test images here
├── rag_bot/
│   ├── rag_bot.py             # CLI RAG chatbot
│   ├── requirements.txt
│   └── docs/                  # Place documents here
└── README.md
```

---

## Assumptions

- `ANTHROPIC_API_KEY` environment variable is set before running any component
- For Task 2.2: test images must be placed in the `./images` folder before running
- For Task 2.3: source documents must be placed in the `./docs` folder before running
- Python 3.11+ recommended for all components
- PDF support in Tasks 2.1 and 2.3 requires `pymupdf` (`pip install pymupdf`)
