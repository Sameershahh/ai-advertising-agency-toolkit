"""
copy_generator.py
-----------------
AI-powered advertising copy generator using the Anthropic API.

Generates 3 structured variations of ad copy (headline, tagline, body, CTA)
from a product brief using Claude.

Temperature: 0.85
- High enough to encourage creative variation between the 3 outputs
- Low enough to keep outputs coherent, brand-relevant, and grammatically polished
- This range is optimal for creative marketing copy: avoids repetition while
  preventing incoherent/off-brief outputs that occur at temperature > 1.0

Usage:
    python copy_generator.py --brief "Your product brief here"
    python copy_generator.py  # Uses default demo brief
"""

import anthropic
import json
import time
import argparse
import logging
import sys
from typing import Optional

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-5"
MAX_TOKENS = 2048
TEMPERATURE = 0.85          # See module docstring for rationale
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0      # seconds; exponential back-off applied per attempt

# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an award-winning advertising copywriter with 20+ years of experience
crafting campaigns for luxury, lifestyle, and consumer brands. Your copy is
precise, evocative, and conversion-focused.

TASK
Given a product brief, generate EXACTLY 3 distinct advertising copy variations.

OUTPUT FORMAT
Respond with ONLY a valid JSON object — no markdown fences, no commentary, no
preamble. The JSON must match this schema exactly:

{
  "variation_1": {
    "headline": "<punchy, 6-10 word headline>",
    "tagline": "<memorable brand tagline, under 8 words>",
    "body": "<2-3 sentence body copy, benefit-driven and emotionally resonant>",
    "cta": "<action-oriented CTA, under 6 words>"
  },
  "variation_2": { ... },
  "variation_3": { ... }
}

RULES
- Each variation must be meaningfully different in angle, tone, or hook.
- Variation 1: Aspirational / emotional tone
- Variation 2: Benefit-driven / rational tone
- Variation 3: Bold / provocative tone
- Never repeat headlines or taglines across variations.
- All strings must be properly escaped for JSON.
- Output ONLY the JSON object — nothing else."""


# ── Core Generation ──────────────────────────────────────────────────────────
def generate_ad_copy(brief: str, client: Optional[anthropic.Anthropic] = None) -> dict:
    """
    Call the Anthropic API to generate 3 ad copy variations from a brief.

    Implements exponential back-off retry logic for rate-limit (429) and
    transient server (5xx) errors.

    Args:
        brief:  The product/campaign brief string.
        client: Optional pre-constructed Anthropic client (useful for testing).

    Returns:
        Parsed dict with variation_1 / variation_2 / variation_3 keys.

    Raises:
        ValueError: If the API returns non-JSON or schema-invalid content.
        RuntimeError: If all retry attempts are exhausted.
    """
    if client is None:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    user_prompt = f"""PRODUCT BRIEF:
{brief}

Generate 3 distinct advertising copy variations following the system instructions exactly."""

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("API call attempt %d / %d …", attempt, MAX_RETRIES)

            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_text: str = message.content[0].text.strip()
            log.debug("Raw API response:\n%s", raw_text)

            # ── Parse & validate JSON ────────────────────────────────────────
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"API returned non-JSON content: {exc}\nRaw: {raw_text[:300]}") from exc

            _validate_schema(data)
            log.info("Copy generated successfully.")
            return data

        except anthropic.RateLimitError as exc:
            last_error = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("Rate limit hit (attempt %d). Retrying in %.1fs …", attempt, delay)
            time.sleep(delay)

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning("Server error %d (attempt %d). Retrying in %.1fs …", exc.status_code, attempt, delay)
                time.sleep(delay)
            else:
                log.error("Non-retryable API error: %s", exc)
                raise

        except anthropic.APIConnectionError as exc:
            last_error = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("Connection error (attempt %d). Retrying in %.1fs …", attempt, delay)
            time.sleep(delay)

        except ValueError:
            # Schema / parsing errors — don't retry, surface immediately
            raise

    raise RuntimeError(
        f"All {MAX_RETRIES} API attempts exhausted. Last error: {last_error}"
    )


def _validate_schema(data: dict) -> None:
    """Ensure the returned dict matches the expected copy schema."""
    required_top = {"variation_1", "variation_2", "variation_3"}
    required_fields = {"headline", "tagline", "body", "cta"}

    missing_top = required_top - data.keys()
    if missing_top:
        raise ValueError(f"JSON missing top-level keys: {missing_top}")

    for var_key in required_top:
        var = data[var_key]
        if not isinstance(var, dict):
            raise ValueError(f"{var_key} is not a dict")
        missing_fields = required_fields - var.keys()
        if missing_fields:
            raise ValueError(f"{var_key} missing fields: {missing_fields}")
        for field in required_fields:
            if not isinstance(var[field], str) or not var[field].strip():
                raise ValueError(f"{var_key}.{field} must be a non-empty string")


# ── CLI Entry Point ──────────────────────────────────────────────────────────
DEFAULT_BRIEF = (
    "New luxury perfume for men, brand name: Noir, "
    "target: 30-45 year old professionals"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3 ad copy variations from a product brief using Claude."
    )
    parser.add_argument(
        "--brief",
        type=str,
        default=DEFAULT_BRIEF,
        help="Product brief string (default: Noir perfume demo brief)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON output (e.g. output.json)",
    )
    args = parser.parse_args()

    log.info("Brief: %s", args.brief)
    log.info("Model: %s | Temperature: %s", MODEL, TEMPERATURE)

    result = generate_ad_copy(args.brief)

    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print("AD COPY VARIATIONS")
    print("=" * 60)
    print(json_output)
    print("=" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        log.info("Output saved to %s", args.output)


if __name__ == "__main__":
    main()
