# Task 1 — LLM Integration & Prompt Engineering

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange)

## Overview

This repository contains two components:

1. **Task 1.1 — AI Copywriting Generator** (`copy_generator/`): A production-grade Python script that generates 3 structured advertising copy variations from a product brief using the Anthropic Claude API.
2. **Task 1.2 — Advanced Prompt Engineering** (`prompt_engineering/`): A documented analysis of 3 rewritten prompts with technique explanations and before/after output comparisons.

---

## Task 1.1 — AI Copywriting Generator

### Features

- Generates 3 distinct copy variations: aspirational, benefit-driven, and provocative
- Each variation includes: `headline`, `tagline`, `body`, and `cta`
- Returns valid, parseable JSON — no cleanup required
- Implements exponential back-off retry logic for rate limits and server errors
- Temperature tuned to 0.85 for optimal creative variety (documented in code)
- Structured system + user prompt design for consistent output format

### Setup

```bash
cd copy_generator
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"
```

### Usage

```bash
# Run with default demo brief (Noir perfume)
python copy_generator.py

# Run with custom brief
python copy_generator.py --brief "New eco-friendly water bottle, brand: AquaFlow, target: 25-40 year old fitness enthusiasts"

# Save output to file
python copy_generator.py --output output.json
```

### Example Output

Run with the brief: `"New luxury perfume for men, brand name: Noir, target: 30-45 year old professionals"`

```json
{
  "variation_1": {
    "headline": "Some Men Don't Follow Paths. They Leave Trails.",
    "tagline": "Noir. The scent of ambition.",
    "body": "Crafted for the man who defines the room before he speaks, Noir is a deep fusion of oud, black pepper, and vetiver — a fragrance that lingers long after you've moved on. This is not just a scent. It's a signature.",
    "cta": "Discover Your Signature"
  },
  "variation_2": {
    "headline": "12 Hours of Lasting Presence. Zero Compromise.",
    "tagline": "Noir. Precision-engineered luxury.",
    "body": "Formulated with rare ingredients sourced from four continents, Noir delivers a consistent, professional-grade scent profile from morning briefing to evening event. Tested for longevity. Designed for professionals who demand performance in everything they wear.",
    "cta": "Shop Noir Now"
  },
  "variation_3": {
    "headline": "Ordinary Fragrances Are for Ordinary Men.",
    "tagline": "Noir. Not for everyone.",
    "body": "You've outgrown mass-market. You've earned exclusivity. Noir is the fragrance that doesn't try to please — it commands. Bold, complex, unapologetic. Wear it if you can handle the attention.",
    "cta": "Claim Your Bottle"
  }
}
```

### Architecture & Key Decisions

**Temperature (0.85):** Selected to maximize creative variation between the 3 outputs while maintaining brand coherence. Values above 1.0 introduce incoherence; values below 0.7 produce repetitive outputs.

**System + User Prompt Structure:** The system prompt defines the AI's persona, output schema, and per-variation tone directives. The user prompt contains only the brief. This separation ensures schema compliance across all runs.

**Retry Logic:** Implements exponential back-off (`2^(attempt-1) * base_delay`) for RateLimitError (429), APIStatusError (5xx), and APIConnectionError. Non-retryable errors (4xx other than 429) surface immediately.

**Schema Validation:** After JSON parsing, a `_validate_schema()` function checks for all required keys and non-empty string values before returning — preventing silent data quality issues.

---

## Task 1.2 — Advanced Prompt Engineering

See `prompt_engineering/prompt_engineering.md` for the full analysis.

### Summary of Techniques Applied

| Prompt | Techniques Used |
|--------|----------------|
| Social media post | Role assignment, output constraints, few-shot tonal reference, context injection, negative constraints |
| Creative rewrite | Senior creative role, constraint-based creativity, context injection, word count anchor, output purity |
| Brief summary | Chain-of-thought via JSON schema, grounding constraint, risk flags field, machine-parseable output |

---

## Repository Structure

```
task1-llm-integration/
├── copy_generator/
│   ├── copy_generator.py      # Main script
│   ├── requirements.txt
│   └── example_output.json    # Pre-generated Noir brief output
├── prompt_engineering/
│   └── prompt_engineering.md  # Full analysis with before/after comparisons
└── README.md
```

---

## Assumptions

- `ANTHROPIC_API_KEY` environment variable is set before running
- Python 3.11+ is used (uses built-in type hints syntax)
- The Anthropic SDK version ≥ 0.25.0 is installed
