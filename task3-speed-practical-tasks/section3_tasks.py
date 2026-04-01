"""
section3_tasks.py
-----------------
Section 3 — Speed & Practical Tasks
All 5 tasks implemented as standalone, importable functions.

Q1: Anthropic API with 3-retry rate limit handling
Q2: Debugged LangChain RAG pipeline (3 bugs fixed, documented)
Q3: AI brand tone enforcer system prompt
Q4: Brand safety evaluator for image generation outputs
Q5: System architecture description for AI ad personalization engine
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q1 — Python function: Anthropic API with retry on rate limit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import anthropic
import time
import logging

log = logging.getLogger(__name__)


def call_anthropic_with_retry(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = "claude-opus-4-5",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """
    Call the Anthropic Messages API with exponential back-off retry logic.

    Retries up to `max_retries` times on RateLimitError (HTTP 429) and
    transient server errors (HTTP 5xx). All other errors are raised immediately.

    Args:
        prompt:      The user message to send.
        system:      System prompt (defaults to generic assistant).
        model:       Anthropic model identifier.
        max_tokens:  Maximum tokens in the response.
        max_retries: Maximum number of retry attempts (default: 3).

    Returns:
        The text content of the first response block.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
        anthropic.APIError: For non-retryable API errors.
    """
    client = anthropic.Anthropic()
    base_delay = 2.0  # seconds; doubles on each attempt

    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        except anthropic.RateLimitError:
            if attempt == max_retries:
                raise RuntimeError(f"Rate limit persisted after {max_retries} retries.")
            delay = base_delay * (2 ** (attempt - 1))
            log.warning("Rate limit hit (attempt %d/%d). Waiting %.1fs…", attempt, max_retries, delay)
            time.sleep(delay)

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                log.warning("Server error %d (attempt %d/%d). Waiting %.1fs…", exc.status_code, attempt, max_retries, delay)
                time.sleep(delay)
            else:
                raise  # Non-retryable or final attempt

        except anthropic.APIConnectionError:
            if attempt == max_retries:
                raise RuntimeError(f"Connection failed after {max_retries} retries.")
            delay = base_delay * (2 ** (attempt - 1))
            log.warning("Connection error (attempt %d/%d). Waiting %.1fs…", attempt, max_retries, delay)
            time.sleep(delay)

    raise RuntimeError(f"All {max_retries} attempts exhausted.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q2 — Debugged LangChain RAG Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
ORIGINAL BROKEN CODE (for reference — do NOT run):

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def build_rag_pipeline(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)       # BUG 1
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,                                    # BUG 2
        return_source_documents=True
    )
    return chain

def query_rag(chain, question):
    result = chain(question)                                    # BUG 3
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources

-------------------------------------------------------------------
BUG ANALYSIS & FIXES:

BUG 1 — Wrong Chroma constructor call
  LOCATION: Chroma.from_documents(docs, embeddings)
  PROBLEM:  `from_documents` expects a list of LangChain Document objects,
            not raw strings or dicts. If `docs` is a list of strings, this
            silently fails or raises an AttributeError at embedding time.
  FIX:      Wrap raw strings in Document objects before passing to Chroma.
            from langchain.schema import Document
            lc_docs = [Document(page_content=d) for d in docs]
            vectorstore = Chroma.from_documents(lc_docs, embeddings)

BUG 2 — Passing retriever instead of chain_type string
  LOCATION: RetrievalQA.from_chain_type(llm=llm, retriever=retriever, ...)
  PROBLEM:  from_chain_type() does not accept a `retriever` keyword argument.
            The correct parameter is `retriever` but it must be passed via
            `from_llm()` or by building with `chain_type_kwargs`. The actual
            API is: RetrievalQA.from_chain_type(llm, chain_type="stuff",
                        retriever=vectorstore.as_retriever())
            The missing `chain_type` arg causes a TypeError.
  FIX:      Add the required `chain_type` argument.
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",      ← ADDED
                retriever=retriever,
                return_source_documents=True
            )

BUG 3 — Deprecated dict-call interface
  LOCATION: result = chain(question)
  PROBLEM:  In modern LangChain (>=0.1.x), calling a chain as a callable
            with a plain string raises a TypeError. The correct interface is
            chain.invoke({"query": question}) for LCEL chains or
            chain.run(question) for legacy chains.
  FIX:      result = chain.invoke({"query": question})
"""

# CORRECTED IMPLEMENTATION:
def build_rag_pipeline_fixed(docs: list[str]):
    """
    Fixed LangChain RAG pipeline.
    Requires: pip install langchain langchain-openai chromadb
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings, OpenAI
        from langchain.chains import RetrievalQA
        from langchain.schema import Document

        # FIX 1: Wrap strings in Document objects
        lc_docs = [Document(page_content=d) for d in docs]

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(lc_docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = OpenAI(temperature=0)

        # FIX 2: Add required chain_type argument
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        return chain

    except ImportError as exc:
        log.error("LangChain dependencies not installed: %s", exc)
        raise


def query_rag_fixed(chain, question: str) -> tuple[str, list]:
    # FIX 3: Use .invoke() instead of direct callable
    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q3 — AI Brand Tone Enforcer System Prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BRAND_TONE_ENFORCER_SYSTEM_PROMPT = """
You are a brand tone enforcement specialist for Verde Beauty — a premium organic skincare brand.

BRAND VOICE GUIDELINES
Verde communicates with confident, quiet authority. The brand is:
✓ Transparent and evidence-based
✓ Warm but not gushing
✓ Sophisticated without being cold
✓ Direct without being blunt

FORBIDDEN LANGUAGE (never use these):
✗ Superlatives without proof: "best," "most amazing," "revolutionary," "game-changing"
✗ Fear-based language: "toxic," "dangerous chemicals," "poison," "harmful"
✗ Hyperbolic claims: "transforms skin overnight," "miracle," "magic"
✗ Casual/slang: "guys," "literally," "obsessed," "so good," "loving it"
✗ Unverifiable claims: "clinically proven" (without citation), "dermatologist recommended" (without specifics)

TONE CALIBRATION EXAMPLES:
WRONG: "OMG you NEED to try this — it literally transforms your skin overnight!!"
RIGHT: "Over four weeks, 87% of testers reported visibly smoother skin."

WRONG: "Ditch the toxic chemicals. Your skin deserves better."
RIGHT: "Every Verde ingredient is USDA Certified Organic — traceable from farm to formula."

WRONG: "Our revolutionary formula is the best on the market."
RIGHT: "Formulated without compromise. Verified by certification, not just claims."

YOUR TASK
When given off-brand copy, rewrite it to align with Verde's brand voice:
1. Identify every violation of the guidelines above
2. Rewrite the copy preserving all factual claims and the intended message
3. Return your response in this exact format:

VIOLATIONS FOUND:
- [list each violation with a brief explanation]

REWRITTEN COPY:
[the corrected, on-brand version]

CHANGES MADE:
- [brief explanation of each significant change]
"""


def enforce_brand_tone(off_brand_copy: str) -> str:
    """Rewrite off-brand copy to match Verde brand guidelines."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        temperature=0.3,
        system=BRAND_TONE_ENFORCER_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Please rewrite this off-brand copy:\n\n{off_brand_copy}"
        }],
    )
    return message.content[0].text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q4 — Brand Safety Evaluation of AI Image Generation Outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
BRAND SAFETY EVALUATION — 3 AI Image Generation Outputs

IMAGE 1: "Luxury perfume bottle on black marble, dramatic side lighting, smoke wisps"
Prompt used: "Noir perfume bottle, luxury product photography, dark background, volumetric smoke"

EVALUATION:
- Visual content: Dark background, smoke effect, single product
- Brand safety score: 9/10
- Reasoning: Smoke is purely atmospheric — not associated with tobacco or danger. Dark
  aesthetic is appropriate for luxury male fragrance positioning. No identifiable people,
  no controversial symbols. Minor deduction (1 point) for potential misread of smoke as
  tobacco-adjacent in conservative markets (e.g., certain MENA placements). Safe for:
  display networks, social feed, magazine digital. Not recommended for: family-safe
  inventory, children's adjacent placements.

---

IMAGE 2: "Group of diverse young professionals laughing at rooftop party, holding cocktails"
Prompt used: "Diverse friend group celebrating, rooftop, golden hour, cocktails, lifestyle photography"

EVALUATION:
- Visual content: People, alcohol (cocktails), social gathering
- Brand safety score: 6/10
- Reasoning: Alcohol is clearly visible and central to the scene. Score deduction of 4
  points: (1) alcohol depiction restricts placement on Google Display Network's
  family-safe tier, (2) cannot run on platforms with alcohol advertising restrictions
  (e.g., certain App Store ad placements), (3) age verification requirement for Facebook/
  Meta. Positive factors: diverse casting is brand-positive; professional attire and
  upscale setting avoid association with binge drinking. Safe for: premium lifestyle
  publications, adult-verified social placements. Not recommended for: programmatic
  general audience, search display, any platform without age-gating.

---

IMAGE 3: "Influencer-style selfie with extreme before/after skin transformation split screen"
Prompt used: "Skincare before after transformation, dramatic results, close-up face split"

EVALUATION:
- Visual content: Human face, medical-adjacent claim implied by before/after format
- Brand safety score: 3/10
- Reasoning: Before/after imagery for skincare is explicitly prohibited under FDA
  guidelines for cosmetic advertising. Score of 3/10 because: (1) the split-screen format
  constitutes an implied efficacy claim regardless of caption text; (2) legally cannot
  run on any regulated advertising platform without FTC disclosure; (3) if associated with
  a skincare brand (Verde), it directly violates their mandatory guidelines. Additionally,
  the implied transformation may constitute a misleading claim under UK ASA standards
  and EU advertising directives. This image requires full creative replacement, not
  just caption editing. Do NOT run in any market without legal review.
"""

BRAND_SAFETY_SCORES = {
    "image_1_noir_perfume": {
        "score": 9,
        "summary": "Atmospheric smoke is non-tobacco — safe for most premium placements",
        "restrictions": ["Conservative MENA markets", "Children-adjacent inventory"],
        "approved_channels": ["Display networks", "Social feed", "Magazine digital"],
    },
    "image_2_rooftop_party": {
        "score": 6,
        "summary": "Alcohol visible — restricted to age-verified placements only",
        "restrictions": ["Google family-safe tier", "App Store ads", "Non-age-gated programmatic"],
        "approved_channels": ["Premium lifestyle publications", "Age-verified social"],
    },
    "image_3_before_after": {
        "score": 3,
        "summary": "FDA violation — before/after skincare imagery cannot run without legal review",
        "restrictions": ["All regulated platforms", "All markets without legal sign-off"],
        "approved_channels": [],
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Q5 — AI Ad Personalization Engine Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHITECTURE_DESCRIPTION = """
AI-Powered Ad Personalization Engine — System Architecture
==========================================================

OVERVIEW
A real-time ad personalization engine for a major e-commerce client that
selects and assembles the optimal ad creative for each user impression based
on behavioral signals, contextual data, and predicted purchase intent.

COMPONENTS

1. DATA INGESTION LAYER
   ├── Clickstream collector (Kafka + event schema)
   ├── CRM sync (user purchase history, LTV segment)
   ├── Real-time context API (device, location, time-of-day)
   └── Product catalog feed (inventory, pricing, availability)

2. FEATURE STORE
   ├── Online store (Redis) — sub-10ms latency for real-time inference
   ├── Offline store (BigQuery) — historical features for model training
   └── Feature computation (Apache Flink for streaming aggregations)
   Key features: recency/frequency/monetary scores, category affinity,
   price sensitivity band, churn risk score, browse-to-buy ratio.

3. AI / ML LAYER
   ├── User embedding model (two-tower neural net — trained weekly)
   ├── Product embedding model (item2vec on co-purchase graph)
   ├── CTR prediction model (LightGBM + online learning)
   ├── LLM copy personalization (Claude API — generates variant headlines
   │   and body copy tuned to user segment at request time)
   └── Visual selector (CLIP embeddings — matches image style to user
       aesthetic profile derived from engagement history)

4. PERSONALIZATION ORCHESTRATOR (core service)
   ├── Retrieves user + context features (<5ms, from Redis)
   ├── Runs CTR ranker to score top-50 product candidates
   ├── Calls LLM copy service for the top-3 candidates (async, <200ms P95)
   ├── Assembles final creative (headline + image + CTA + product)
   └── Returns personalized ad payload as JSON to ad server

5. AD SERVER INTEGRATION
   ├── OpenRTB 2.6 bid response with custom extension for creative assembly
   ├── Supports: Google DV360, The Trade Desk, Meta Advantage+
   └── Creative preview API for QA/brand safety pre-check

6. BRAND SAFETY & COMPLIANCE LAYER
   ├── Claude Vision API — real-time image brand safety scoring
   ├── LLM-based copy compliance checker (tone, claim, regulatory)
   ├── Hard-block rules engine (forbidden categories, geo restrictions)
   └── Audit log (all personalized creatives stored 90 days for review)

7. FEEDBACK & LEARNING LOOP
   ├── Impression / click / conversion events → Kafka
   ├── Online model update (A/B tested, shadow mode first)
   ├── Weekly full model retraining (Vertex AI Pipelines)
   └── Performance dashboard (Looker Studio, real-time KPIs)

8. INFRASTRUCTURE
   ├── Kubernetes (GKE) — all services containerized, auto-scaled
   ├── Cloud Armor — DDoS protection for public-facing APIs
   ├── Latency SLA: <250ms end-to-end at P95 (measured at bid request)
   └── Availability target: 99.95% uptime

KEY DESIGN DECISIONS
- Synchronous path (feature retrieval + ranking) kept under 50ms
- LLM copy generation is async/cached — same-segment users share generated
  copy variants for 15 minutes to reduce API costs by ~70%
- All LLM outputs pass through compliance layer before serving
- Multi-armed bandit (Thompson Sampling) manages creative variant exploration
  without sacrificing exploitation of known top performers

ESTIMATED THROUGHPUT
- 50,000 bid requests/second at peak
- 3M unique users/day
- 15M personalized impressions/day
"""


if __name__ == "__main__":
    print("Section 3 Tasks loaded. Import functions individually or review Q4/Q5 docstrings.")
    print("\nQ3 Brand Tone Enforcer — Sample usage:")
    print('  from section3_tasks import enforce_brand_tone')
    print('  print(enforce_brand_tone("This is literally the BEST skincare ever!!"))')
    print("\nQ5 Architecture description printed below:")
    print(ARCHITECTURE_DESCRIPTION)
