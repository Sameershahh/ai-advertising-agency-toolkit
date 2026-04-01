# AI Advertising Agency Toolkit

A complete, production-ready AI engineering toolkit built for an advertising agency — covering LLM integration, API development, vision AI, RAG systems, and prompt engineering.

Built with **Python** · **Claude (Anthropic)** · **FastAPI** · **Docker**

---

## Repository Structure

```
ai-advertising-agency-toolkit/
├── task1-llm-integration/          # Section 1 — LLM Integration & Prompt Engineering
│   ├── copy_generator/
│   │   ├── copy_generator.py       # AI copywriting script (3 variations per brief)
│   │   ├── example_output.json     # Pre-generated Noir perfume brief output
│   │   └── requirements.txt
│   ├── prompt_engineering/
│   │   └── prompt_engineering.md   # 3 rewritten prompts with technique analysis
│   └── README.md
│
├── task2-ai-system-design/         # Section 2 — AI System Design & API Development
│   ├── campaign_analyzer/
│   │   ├── main.py                 # FastAPI app — POST /analyze-brief
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── sample_brief.txt        # 500-word demo campaign brief
│   │   └── requirements.txt
│   ├── image_tagger/
│   │   ├── image_tagger.py         # Batch image auto-tagging with brand safety scores
│   │   └── requirements.txt
│   ├── rag_bot/
│   │   ├── rag_bot.py              # RAG CLI chatbot — answers only from documents
│   │   ├── docs/                   # Sample agency documents included
│   │   └── requirements.txt
│   └── README.md
│
├── task3-speed-practical-tasks/    # Section 3 — Speed & Practical Tasks
│   ├── section3_tasks.py           # All 5 tasks: Q1 retry, Q2 debug, Q3 tone enforcer, Q4 safety eval, Q5 architecture
│   └── README.md
│
└── README.md                       ← You are here
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- An Anthropic API key → [console.anthropic.com](https://console.anthropic.com)

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Task 1 — Generate Ad Copy
```bash
cd task1-llm-integration/copy_generator
pip install -r requirements.txt
python copy_generator.py --brief "New luxury perfume for men, brand: Noir, target: 30-45 year old professionals"
```

### Task 2 — Run the Campaign Analyzer API
```bash
cd task2-ai-system-design/campaign_analyzer
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

Or with Docker:
```bash
docker-compose up
```

### Task 2 — Run the RAG Bot
```bash
cd task2-ai-system-design/rag_bot
pip install -r requirements.txt
python rag_bot.py --docs ./docs
```

### Task 3 — All Speed Tasks
```bash
cd task3-speed-practical-tasks
pip install anthropic
python section3_tasks.py
```

---

## Live Demo

**Interactive demo:** https://sameershahh.github.io/ai-advertising-agency-toolkit/

Includes live Claude API integration for:
- Task 1.1 — Ad copy generation (3 variations)
- Task 2.1 — Campaign brief analyzer
- Section 3 Q3 — Brand tone enforcer

---

## Tasks Covered

| Section | Task | Description |
|---------|------|-------------|
| 1 | 1.1 | AI Copywriting API — 3 structured copy variations per brief |
| 1 | 1.2 | Advanced Prompt Engineering — 3 rewrites with technique analysis |
| 2 | 2.1 | Campaign Brief Analyzer — FastAPI + PDF upload + SSE streaming |
| 2 | 2.2 | Image Auto-Tagger — Batch vision analysis + brand safety scoring |
| 2 | 2.3 | RAG Knowledge Bot — Document-grounded Q&A with source citations |
| 3 | Q1 | Anthropic API retry function with exponential back-off |
| 3 | Q2 | LangChain RAG pipeline — 3 bugs debugged and fixed |
| 3 | Q3 | Brand tone enforcer system prompt + live function |
| 3 | Q4 | Brand safety evaluation of 3 AI image generation outputs |
| 3 | Q5 | AI ad personalization engine — full 8-layer system architecture |

---

## Tech Stack

- **LLM:** Anthropic Claude (claude-opus-4-5)
- **API Framework:** FastAPI + Pydantic v2
- **Vision:** Claude Vision API (base64 image encoding)
- **RAG:** Custom cosine similarity retrieval (NumPy) + Voyage-3 embeddings
- **PDF:** PyMuPDF
- **Containerisation:** Docker + docker-compose
- **Streaming:** Server-Sent Events (SSE)

---

## Author

**Sameer Shah** · [@sameershahh](https://github.com/sameershahh)
