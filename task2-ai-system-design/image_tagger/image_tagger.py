"""
image_tagger.py
---------------
Batch advertising image analyzer using Claude's vision capabilities.

Processes all supported images in a specified folder and produces a single
tags_output.json file with:
  - alt_text: SEO-optimised image description
  - tags: content/semantic tags
  - brand_safety_score: 1–10 (10 = fully brand safe)
  - use_cases: suggested campaign placements

Supported formats: .jpg, .jpeg, .png, .gif, .webp

Usage:
    python image_tagger.py --folder ./images
    python image_tagger.py --folder ./images --output ./results/tags_output.json
"""

import anthropic
import base64
import json
import time
import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MEDIA_TYPE_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert advertising image analyst with deep knowledge of
brand safety, digital advertising standards, and content classification.

TASK
Analyse the provided advertising image and return a structured JSON object.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no commentary:

{
  "alt_text": "<concise, descriptive alt text for accessibility and SEO, 10-20 words>",
  "tags": ["<tag1>", "<tag2>", "<tag3>", "<tag4>", "<tag5>"],
  "brand_safety_score": <integer 1-10>,
  "use_cases": ["<use case 1>", "<use case 2>", "<use case 3>"]
}

FIELD DEFINITIONS:
- alt_text: Describe who/what is in the image, the setting, and mood. No opinions.
- tags: 5-8 descriptive content tags (subject, mood, colour palette, setting, style)
- brand_safety_score: Rate from 1 (unsafe — graphic/controversial) to 10 (fully safe for all audiences)
  Scoring guide:
    10: Neutral, universally appropriate, no sensitive content
    7-9: Minor concerns (mild suggestiveness, extreme sports, alcohol in context)
    4-6: Moderate concerns (strong emotion, political imagery, age-restricted products)
    1-3: Unsafe (violence, adult content, hate speech, drugs)
- use_cases: 2-4 specific advertising placements (e.g., "Instagram feed", "Family-friendly display network")

Output ONLY valid JSON. Nothing else."""


# ── Core Functions ─────────────────────────────────────────────────────────────
def encode_image(image_path: Path) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    ext = image_path.suffix.lower()
    media_type = MEDIA_TYPE_MAP.get(ext)
    if not media_type:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def analyze_image(
    image_path: Path,
    client: anthropic.Anthropic,
) -> dict:
    """
    Send a single image to Claude Vision and return the structured analysis dict.
    Implements exponential back-off on rate limit / server errors.
    """
    try:
        image_data, media_type = encode_image(image_path)
    except (ValueError, IOError) as exc:
        log.error("Cannot encode %s: %s", image_path.name, exc)
        return _error_entry(image_path.name, str(exc))

    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("  Analyzing %s (attempt %d/%d)…", image_path.name, attempt, MAX_RETRIES)

            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0.2,     # Low temp for consistent, factual image descriptions
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyse this advertising image and return the JSON analysis.",
                            },
                        ],
                    }
                ],
            )

            raw = message.content[0].text.strip()

            try:
                result = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("Non-JSON response for %s: %s", image_path.name, raw[:200])
                return _error_entry(image_path.name, f"JSON parse error: {exc}")

            # Validate required fields
            required = {"alt_text", "tags", "brand_safety_score", "use_cases"}
            missing = required - result.keys()
            if missing:
                return _error_entry(image_path.name, f"Missing fields: {missing}")

            result["filename"] = image_path.name
            log.info("  ✓ %s — safety score: %s", image_path.name, result.get("brand_safety_score"))
            return result

        except anthropic.RateLimitError as exc:
            last_error = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("  Rate limit hit. Retrying in %.1fs…", delay)
            time.sleep(delay)

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_error = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning("  Server error %d. Retrying in %.1fs…", exc.status_code, delay)
                time.sleep(delay)
            else:
                log.error("  Non-retryable error for %s: %s", image_path.name, exc)
                return _error_entry(image_path.name, f"API error {exc.status_code}")

        except anthropic.APIConnectionError as exc:
            last_error = exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("  Connection error. Retrying in %.1fs…", delay)
            time.sleep(delay)

    return _error_entry(image_path.name, f"All retries exhausted: {last_error}")


def _error_entry(filename: str, reason: str) -> dict:
    """Return a structured error placeholder for a failed image."""
    return {
        "filename": filename,
        "error": reason,
        "alt_text": None,
        "tags": [],
        "brand_safety_score": None,
        "use_cases": [],
    }


def process_folder(folder: Path, output_path: Path) -> None:
    """
    Batch process all supported images in a folder and write tags_output.json.
    """
    if not folder.exists() or not folder.is_dir():
        log.error("Folder not found: %s", folder)
        sys.exit(1)

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        log.warning("No supported images found in %s", folder)
        log.warning("Supported formats: %s", ", ".join(SUPPORTED_EXTENSIONS))
        sys.exit(0)

    log.info("Found %d image(s) in %s", len(images), folder)
    client = anthropic.Anthropic()

    results = []
    for i, img_path in enumerate(images, 1):
        log.info("[%d/%d] Processing: %s", i, len(images), img_path.name)
        entry = analyze_image(img_path, client)
        results.append(entry)

        # Polite delay between requests to avoid hammering rate limits
        if i < len(images):
            time.sleep(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"images": results, "total": len(results)}, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if "error" not in r or r["error"] is None)
    log.info("Complete. %d/%d images processed successfully.", success, len(results))
    log.info("Output saved to: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch analyze advertising images with Claude Vision."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("./images"),
        help="Path to folder containing images (default: ./images)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./tags_output.json"),
        help="Output JSON file path (default: ./tags_output.json)",
    )
    args = parser.parse_args()

    log.info("Image Tagger | Model: %s", MODEL)
    log.info("Input folder : %s", args.folder.resolve())
    log.info("Output file  : %s", args.output.resolve())

    process_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
