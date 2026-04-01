"""
main.py — AI Campaign Brief Analyzer API
=========================================
FastAPI service that accepts a campaign brief (text or PDF) and returns
a structured strategic analysis using Claude.

Endpoints:
  POST /analyze-brief   — JSON body with brief_text field
  POST /analyze-pdf     — Multipart form upload of a PDF file (bonus)
  GET  /health          — Health check
  GET  /                — API info / documentation redirect

Run:
  uvicorn main:app --reload --port 8000
"""

import os
import io
import json
import time
import logging
import anthropic

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Campaign Brief Analyzer",
    description="Analyzes advertising campaign briefs using Claude to extract audience, messages, tone, channels, and risks.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anthropic Client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic()
MODEL = "claude-opus-4-5"
MAX_TOKENS = 2048

# ── Schemas ───────────────────────────────────────────────────────────────────
class BriefRequest(BaseModel):
    brief_text: str = Field(
        ...,
        min_length=50,
        max_length=20000,
        description="Raw campaign brief text (50–20,000 characters)",
        example="We are launching a new organic skincare line targeting women aged 28-45 who are eco-conscious..."
    )
    stream: bool = Field(
        default=False,
        description="If true, response will be streamed via SSE"
    )


class TargetAudience(BaseModel):
    primary: Optional[str]
    secondary: Optional[str]


class BriefAnalysis(BaseModel):
    campaign_objective: Optional[str]
    target_audience: TargetAudience
    key_messages: list[str]
    tone: str
    channels: list[str]
    risks: list[str]
    one_line_brief: Optional[str]


class AnalysisResponse(BaseModel):
    status: str
    analysis: BriefAnalysis
    model: str
    processing_time_ms: int


# ── System Prompt ──────────────────────────────────────────────────────────────
ANALYSIS_SYSTEM_PROMPT = """You are a senior strategic account planner at a world-class advertising agency.
Your role is to analyse campaign briefs and produce precise, actionable strategic summaries.

TASK
Analyse the provided campaign brief and return a structured JSON analysis.

OUTPUT FORMAT — respond with ONLY a valid JSON object, no markdown, no commentary:

{
  "campaign_objective": "<single sentence — the measurable campaign goal>",
  "target_audience": {
    "primary": "<demographic + psychographic descriptor>",
    "secondary": "<secondary audience if mentioned, else null>"
  },
  "key_messages": ["<core message 1>", "<core message 2>", "<core message 3>"],
  "tone": "<2-4 descriptive adjectives defining the communication tone>",
  "channels": ["<recommended or mentioned channel 1>", "<channel 2>", "<channel 3>"],
  "risks": ["<risk or ambiguity 1>", "<risk or ambiguity 2>"],
  "one_line_brief": "<the campaign essence in 10 words or fewer>"
}

STRICT RULES:
- Derive information ONLY from the provided brief — no external assumptions
- If a field cannot be determined, use null for strings or [] for arrays
- risks must be specific and constructive (not generic)
- Output must be valid parseable JSON only — absolutely no markdown fences"""


# ── Helper: Call Claude ────────────────────────────────────────────────────────
def _call_claude_analysis(brief_text: str) -> dict:
    """Call Claude synchronously and return parsed analysis dict."""
    user_prompt = f"CAMPAIGN BRIEF:\n\n{brief_text}\n\nProvide your structured analysis as JSON."

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.3,   # Low temperature for analytical tasks = more deterministic/consistent
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    log.debug("Raw Claude response: %s", raw[:500])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Failed to parse Claude response as JSON: %s", raw[:300])
        raise ValueError(f"Claude returned non-JSON output: {exc}") from exc


def _structure_analysis(raw: dict) -> BriefAnalysis:
    """Convert raw dict to typed BriefAnalysis model."""
    audience_raw = raw.get("target_audience", {}) or {}
    return BriefAnalysis(
        campaign_objective=raw.get("campaign_objective"),
        target_audience=TargetAudience(
            primary=audience_raw.get("primary"),
            secondary=audience_raw.get("secondary"),
        ),
        key_messages=raw.get("key_messages") or [],
        tone=raw.get("tone") or "Not specified",
        channels=raw.get("channels") or [],
        risks=raw.get("risks") or [],
        one_line_brief=raw.get("one_line_brief"),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "AI Campaign Brief Analyzer",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze-brief": "Analyze brief from JSON text",
            "POST /analyze-pdf":   "Analyze brief from PDF upload",
            "GET  /health":        "Health check",
            "GET  /docs":          "Interactive API docs (Swagger UI)",
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/analyze-brief", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_brief(request: BriefRequest):
    """
    Analyze a campaign brief provided as plain text.

    Accepts a JSON body with a `brief_text` field (50–20,000 chars).
    Returns structured analysis: objective, audience, messages, tone, channels, risks.

    Set `stream: true` to receive a Server-Sent Events stream instead.
    """
    if request.stream:
        return await _stream_analysis(request.brief_text)

    start = time.time()
    log.info("Analyzing brief (%d chars)…", len(request.brief_text))

    try:
        raw_analysis = _call_claude_analysis(request.brief_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Upstream rate limit reached. Please retry shortly.")
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc.status_code}")

    analysis = _structure_analysis(raw_analysis)
    elapsed_ms = int((time.time() - start) * 1000)

    log.info("Analysis complete in %dms", elapsed_ms)
    return AnalysisResponse(
        status="success",
        analysis=analysis,
        model=MODEL,
        processing_time_ms=elapsed_ms,
    )


@app.post("/analyze-pdf", tags=["Analysis"])
async def analyze_pdf(file: UploadFile = File(..., description="PDF file of the campaign brief")):
    """
    (Bonus) Upload a PDF campaign brief for analysis.
    Extracts text via PyMuPDF and runs the same analysis pipeline.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF support requires PyMuPDF. Install with: pip install pymupdf"
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="PDF must be under 10 MB.")

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=422, detail="PDF appears to contain insufficient text content.")

    start = time.time()
    log.info("Analyzing PDF brief '%s' (%d chars extracted)…", file.filename, len(text))

    try:
        raw_analysis = _call_claude_analysis(text[:20000])  # cap at 20k chars
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Upstream rate limit reached.")
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc.status_code}")

    analysis = _structure_analysis(raw_analysis)
    elapsed_ms = int((time.time() - start) * 1000)

    return AnalysisResponse(
        status="success",
        analysis=analysis,
        model=MODEL,
        processing_time_ms=elapsed_ms,
    )


async def _stream_analysis(brief_text: str) -> StreamingResponse:
    """Stream Claude's analysis response as Server-Sent Events."""

    async def event_generator():
        user_prompt = f"CAMPAIGN BRIEF:\n\n{brief_text}\n\nProvide your structured analysis as JSON."
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0.3,
                system=ANALYSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )
