# Task 1.2 — Advanced Prompt Engineering Challenge

## Overview

Three weak prompts are rewritten using advanced prompt engineering techniques.
Each rewrite includes the technique applied and a documented explanation.

---

## Prompt 1 — Social Media Post for Shoe Brand

### ❌ Weak Version
```
Write a social media post for our new shoe brand.
```

### ✅ Rewritten Version
```
You are a sharp, culturally-attuned social media strategist specialising in
Gen Z streetwear and sneaker culture. Your writing is confident, punchy, and
platform-native — it feels like it belongs in the feed, not like an ad.

TASK
Write ONE Instagram caption for the launch of "STRYDE" — a new direct-to-consumer
sneaker brand targeting 18-28 year-olds in urban markets.

BRAND VOICE: Bold. Minimal. Anti-hype. Real.

PRODUCT: The STRYDE ONE — a clean low-top silhouette in triple-white, built for
all-day wear. Retail: $110.

CAPTION REQUIREMENTS:
- Length: 3-5 lines max
- Open with a hook (do NOT start with "Introducing" or the brand name)
- Include exactly 1 line of white space between hook and body
- End with a soft CTA (not "shop now" or "buy today")
- Include 3-5 relevant hashtags on a new line at the end
- Tone: conversational but aspirational — like a cool friend, not a brand account

OUTPUT FORMAT:
Return only the caption text, ready to paste directly into Instagram.
No explanations, no alternatives, no commentary.

EXAMPLE OF BRAND VOICE (do NOT copy, use as tonal reference only):
"Your fit doesn't need more noise.
Just the right pair.
STRYDE ONE — now live."
```

### Explanation of Techniques Used
1. **Role assignment**: Opens by defining the AI as a culturally-specific expert (Gen Z streetwear strategist), which anchors tone and cultural fluency throughout the output.
2. **Output constraints**: Explicit structural rules (line length, no "Introducing," spacing, CTA style, hashtag count) eliminate the most common failure modes of weak prompts and ensure copy is immediately usable.
3. **Few-shot / tonal reference**: The example demonstrates voice without prescribing content, giving the model a calibration anchor while preserving creative freedom.
4. **Context injection**: Brand name, target audience, price point, silhouette, and brand voice are all specified — removing ambiguity that leads to generic, off-brief outputs.
5. **Negative constraints**: "Do NOT start with…", "not 'shop now'" — explicitly ruling out clichés forces the model toward less predictable, more authentic outputs.

---

### Before/After Comparison (Actual AI Outputs)

**BEFORE (output from weak prompt):**
> "Introducing our new shoe brand! 👟 We're excited to launch our latest collection
> of stylish and comfortable footwear. Perfect for any occasion, our shoes combine
> fashion with function. Shop now and step up your style game! #shoes #fashion
> #newcollection"

**AFTER (output from rewritten prompt):**
> Clean lines. No logos screaming for attention.
>
> The STRYDE ONE was built for the days when you want your fit to speak quietly —
> and still be the loudest thing in the room.
>
> On site now. Link in bio.
>
> #STRYDE #StreetStyle #SneakerCulture #STRYDE ONE #CleanKicks

---

## Prompt 2 — Make Ad Copy More Creative

### ❌ Weak Version
```
Make our ad copy more creative.
```

### ✅ Rewritten Version
```
You are a senior creative director at a top-tier advertising agency. You are
known for transforming flat, functional ad copy into culturally resonant,
emotionally charged creative work — without sacrificing clarity or conversion.

TASK
Rewrite the ad copy below to be significantly more creative, memorable, and
emotionally engaging. Preserve the core message and CTA intent.

ORIGINAL COPY:
[INSERT COPY HERE]

BRAND CONTEXT:
- Brand: [Brand name]
- Industry: [Industry]
- Target audience: [Age range, psychographic descriptor]
- Tone of voice: [e.g., "witty and irreverent" / "warm and trustworthy"]
- Channels this copy will run on: [e.g., Instagram feed, Google Display]

REWRITE INSTRUCTIONS:
1. Lead with an unexpected angle or emotional hook — avoid stating the obvious
2. Use vivid, sensory, or specific language — eliminate all generic adjectives
   (great, amazing, best, innovative, etc.)
3. Preserve the original CTA but make it feel inevitable, not forced
4. Keep within 10% of the original word count
5. Do not use rhetorical questions as the opening line

DELIVERABLE:
Return the rewritten copy only. No commentary, no "here's what I changed."
```

### Explanation of Techniques Used
1. **Role assignment with specialisation**: Framing the AI as a "senior creative director known for X" sets a high performance bar and primes it for quality-over-speed creative output.
2. **Structured context injection**: Brand, audience, tone, and channel are required fields — ensuring the rewrite is strategically grounded, not just stylistically different.
3. **Constraint-based creativity**: Rules like "no generic adjectives" and "avoid rhetorical questions" are evidence-based constraints that eliminate the most common creative clichés.
4. **Word count anchoring**: The "within 10% of original word count" constraint prevents scope creep and ensures the output is drop-in replaceable.
5. **Output purity instruction**: "No commentary, no 'here's what I changed'" ensures the output is immediately usable without manual cleanup.

---

## Prompt 3 — Summarise Campaign Brief

### ❌ Weak Version
```
Summarize this campaign brief.
```

### ✅ Rewritten Version
```
You are a strategic account planner at a leading advertising agency. Your role
is to distil complex campaign briefs into sharp, actionable summaries that
creative teams can use as their north star.

TASK
Analyse the campaign brief below and produce a structured strategic summary.

CAMPAIGN BRIEF:
[INSERT BRIEF TEXT HERE]

OUTPUT FORMAT
Return a JSON object with the following fields. All values must be derived
strictly from the provided brief — do not infer or add information not present.

{
  "campaign_objective": "<single sentence — the ONE measurable goal>",
  "target_audience": {
    "primary": "<demographic + psychographic descriptor>",
    "secondary": "<if mentioned, else null>"
  },
  "key_messages": ["<message 1>", "<message 2>", "<message 3 if present>"],
  "tone_of_voice": "<2-4 adjectives from the brief or clearly implied by it>",
  "suggested_channels": ["<channel 1>", "<channel 2>"],
  "mandatory_inclusions": ["<any mandatories, disclaimers, or brand assets required>"],
  "red_flags": ["<ambiguities, missing info, or strategic risks>"],
  "one_line_brief": "<10 words or fewer — the essence of the campaign>"
}

RULES:
- If a field cannot be determined from the brief, set its value to null
- Do not add information not present in the brief
- Red flags must be constructive and specific (not generic observations)
- Output must be valid, parseable JSON — no markdown fences, no commentary
```

### Explanation of Techniques Used
1. **Chain-of-thought structure via JSON schema**: Forcing output into a structured JSON schema implicitly requires the model to work through each dimension of the brief systematically, producing more thorough analysis than a free-form summary.
2. **Role assignment**: The "strategic account planner" role primes the model for analytical, business-oriented thinking rather than a generic summarisation task.
3. **Grounding constraint**: "Derived strictly from the provided brief — do not infer" prevents hallucinated insights, which are a critical failure mode for brief summarisation in real agency workflows.
4. **Red flags field**: Explicitly asking for risks and ambiguities surfaces problems that generic summarisation would omit — adding strategic value beyond information compression.
5. **Output purity**: JSON-only output with no markdown fences ensures the result is machine-parseable and directly usable in downstream systems (e.g., a campaign analysis API like Task 2.1).
