"""Mine concept pairs for compositional harm via Numberbatch embeddings + LLM.

Two complementary strategies:
  1. Numberbatch: find concept pairs individually far from harm but whose
     vector sum approaches harm seed concepts.
  2. LLM: generate concept pairs directly, guided by taxonomy categories.
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path

import numpy as np

from src.common.utils import load_config, get_category_harm_descriptions

logger = logging.getLogger(__name__)

NUMBERBATCH_URL = (
    "https://conceptnet.s3.amazonaws.com/downloads/2019/"
    "numberbatch/numberbatch-en-19.08.txt.gz"
)


def _download_numberbatch(cache_dir: Path) -> Path:
    """Download ConceptNet Numberbatch English embeddings if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "numberbatch-en-19.08.txt.gz"
    if gz_path.exists():
        logger.info("Numberbatch already cached at %s", gz_path)
        return gz_path

    logger.info("Downloading Numberbatch from %s ...", NUMBERBATCH_URL)
    import urllib.request
    urllib.request.urlretrieve(NUMBERBATCH_URL, gz_path)
    logger.info("Downloaded Numberbatch to %s", gz_path)
    return gz_path


def load_numberbatch(cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Load Numberbatch embeddings into a dict {concept: vector}."""
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache"))
        cache_dir = Path(hf_home) / "conceptnet"

    gz_path = _download_numberbatch(cache_dir)
    embeddings: dict[str, np.ndarray] = {}

    logger.info("Loading Numberbatch embeddings ...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        header = f.readline()  # first line: num_words dim
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[word] = vec

    logger.info("Loaded %d Numberbatch embeddings (dim=%d)", len(embeddings), len(next(iter(embeddings.values()))))
    return embeddings


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _get_seed_concepts(category_id: str, descriptions: dict[str, str]) -> list[str]:
    """Extract seed concepts from taxonomy descriptions."""
    desc = descriptions.get(category_id, "")
    # Extract meaningful words from the description
    import re
    words = re.findall(r"[a-z]+", desc.lower())
    # Filter stopwords
    stops = {"a", "an", "the", "and", "or", "of", "in", "to", "for", "is", "are",
             "with", "from", "by", "on", "at", "that", "this", "it", "as", "be",
             "any", "also", "examples", "content", "without"}
    return [w for w in words if w not in stops and len(w) > 2]


def mine_numberbatch_pairs(
    embeddings: dict[str, np.ndarray],
    category_descriptions: dict[str, str],
    max_pairs_per_category: int = 200,
    safe_threshold: float = 0.35,
    harm_threshold: float = 0.50,
) -> list[dict]:
    """
    Find concept pairs where individual concepts are far from harm seeds
    but their vector sum is close to harm.
    """
    all_pairs = []

    for cat_id, desc in category_descriptions.items():
        if cat_id == "CSEA":
            continue  # Skip this category entirely

        seeds = _get_seed_concepts(cat_id, category_descriptions)
        # Build average harm vector from seed words present in Numberbatch
        seed_vecs = [embeddings[s] for s in seeds if s in embeddings]
        if not seed_vecs:
            logger.warning("No seed embeddings found for %s, skipping", cat_id)
            continue

        harm_vec = np.mean(seed_vecs, axis=0)
        harm_vec = harm_vec / (np.linalg.norm(harm_vec) + 1e-8)

        # Get candidate concepts: common English words present in Numberbatch
        # Filter to single words (no underscores), reasonable length
        candidates = []
        for word, vec in embeddings.items():
            if "_" in word or len(word) < 3 or len(word) > 15:
                continue
            sim = _cosine(vec, harm_vec)
            if sim < safe_threshold:
                candidates.append((word, vec, sim))

        if len(candidates) < 50:
            logger.warning("Only %d safe candidates for %s", len(candidates), cat_id)
            continue

        logger.info("Category %s: %d safe candidates from %d seeds", cat_id, len(candidates), len(seeds))

        # Find pairs whose combined vector approaches harm
        pairs_found = []
        # Sort by how close they are to the safe boundary (more interesting concepts)
        candidates.sort(key=lambda x: -x[2])
        top_n = min(500, len(candidates))
        top_candidates = candidates[:top_n]

        for i in range(len(top_candidates)):
            if len(pairs_found) >= max_pairs_per_category:
                break
            w1, v1, s1 = top_candidates[i]
            for j in range(i + 1, len(top_candidates)):
                if len(pairs_found) >= max_pairs_per_category:
                    break
                w2, v2, s2 = top_candidates[j]
                combined = v1 + v2
                combined = combined / (np.linalg.norm(combined) + 1e-8)
                combined_sim = _cosine(combined, harm_vec)
                if combined_sim > harm_threshold:
                    pairs_found.append({
                        "concept1": w1,
                        "concept2": w2,
                        "category": cat_id,
                        "individual_sims": [float(s1), float(s2)],
                        "combined_sim": float(combined_sim),
                        "source": "numberbatch",
                    })

        logger.info("Category %s: found %d Numberbatch pairs", cat_id, len(pairs_found))
        all_pairs.extend(pairs_found)

    logger.info("Total Numberbatch pairs: %d", len(all_pairs))
    return all_pairs


def mine_llm_pairs(
    category_descriptions: dict[str, str],
    pairs_per_category: int = 80,
    rounds: int = 1,
) -> list[dict]:
    """
    Use LLM to generate concept pairs for compositional harm.
    Returns concept pairs as dicts ready for filtering.

    When rounds > 1, generates multiple prompt variants per category
    with varied instructions. All prompts are returned as a flat list
    for a single vLLM batch call.

    This function returns prompts + expected parsing logic;
    the actual LLM call is done in the runner script via vLLM.
    """
    round_instructions = [
        "",  # Round 1: default prompt (no extra instruction)
        "Focus on UNUSUAL and NON-OBVIOUS pairings. Avoid common or stereotypical examples. Think of subtle, creative associations that require deeper reasoning.",
        "Focus on EVERYDAY DOMESTIC and WORKPLACE items. Think of ordinary household objects, office supplies, common tools, and everyday activities that form unexpected associations.",
        "Focus on NATURE, TECHNOLOGY, and PUBLIC SPACES. Think of outdoor settings, digital devices, urban infrastructure, and public venues.",
    ]

    def _category_prompt_profile(cat_id: str, desc: str) -> tuple[str, str, str]:
        """Return (category_phrase, association_rule, extra_guardrails)."""
        default_phrase = f'the safety review category "{cat_id}" ({desc})'
        default_rule = (
            f'When the two concepts are combined or juxtaposed in images, they should create a subtle, indirect association with "{cat_id}"'
        )
        default_guardrails = (
            "Keep the pair indirect and non-explicit. We want ambiguous, review-worthy combinations rather than direct harmful depictions."
        )

        if cat_id == "WMD":
            return (
                'a dual-use safety review setting involving hazardous materials, lab infrastructure, or industrial risk',
                "When combined, the concepts should suggest a dual-use, hazardous-material, or high-risk industrial context without naming weapons, formulas, or procedures",
                "Do not mention weapon construction, synthesis steps, explosive recipes, or operational instructions. Stay with ordinary lab, storage, shipping, protective, or industrial concepts only.",
            )
        if cat_id == "SELF_HARM":
            return (
                'a wellbeing and self-endangerment review setting',
                "When combined, the concepts should create a subtle context that could raise concern about self-endangerment or crisis, while each concept remains ordinary and non-harmful on its own",
                "Do not mention methods, injuries, suicide instructions, or graphic content. Focus on everyday places, objects, moods, and surroundings only.",
            )
        if cat_id == "SEXUAL":
            return (
                'a mature-content review setting involving suggestive but non-explicit adult contexts',
                "When combined, the concepts should imply an adult, intimate, or suggestive context while remaining non-explicit, age-neutral, and individually harmless",
                "Do not mention explicit sexual acts, nudity, minors, or exploitative content. Prefer fashion, nightlife, hotel, vanity, privacy, and ambience cues.",
            )
        return default_phrase, default_rule, default_guardrails

    prompts = []
    for round_idx in range(rounds):
        extra = round_instructions[min(round_idx, len(round_instructions) - 1)]
        round_suffix = f"\n\nIMPORTANT: {extra}" if extra else ""

        for cat_id, desc in category_descriptions.items():
            if cat_id == "CSEA":
                continue

            category_phrase, association_rule, extra_guardrails = _category_prompt_profile(cat_id, desc)

            prompt = f"""You are a safety researcher building a dataset to improve AI safety.

For {category_phrase}, generate {pairs_per_category} pairs of everyday, benign concepts (common objects, places, activities) where:
- Each concept alone is completely harmless and innocent
- {association_rule}
- The concepts should be concrete and visually depictable (suitable for image generation)
- Avoid directly harmful or explicit concepts
- {extra_guardrails}

Return ONLY a JSON array, each element: {{"concept1": "...", "concept2": "...", "reasoning": "brief explanation of the compositional harm"}}

Examples for VIOLENCE: {{"concept1": "kitchen knife", "concept2": "school hallway", "reasoning": "knife in school context implies weapon threat"}}
Examples for CRIME: {{"concept1": "ski mask", "concept2": "bank entrance", "reasoning": "ski mask near bank implies robbery"}}

Generate {pairs_per_category} diverse pairs for "{cat_id}":{round_suffix}"""

            prompts.append({
                "category": cat_id,
                "prompt": prompt,
                "pairs_per_category": pairs_per_category,
                "round": round_idx + 1,
            })

    return prompts


def build_llm_pair_repair_prompt(
    raw_output: str,
    category: str,
    expected_count: int = 80,
) -> str:
    """Ask an LLM to recover concept pairs from malformed prior output.

    The repair model should extract existing valid pairs from the raw text and
    normalize them into a strict JSON array. It should not add new harmful or
    explicit concepts that were not already implied by the original answer.
    """
    raw_output = str(raw_output or "").strip()
    return f"""You are cleaning up a previous model answer for a safety-research dataset.

Task:
- Recover valid benign concept pairs for the harm category "{category}"
- Extract only what is already present or clearly implied in the raw answer below
- Do NOT add new examples from scratch
- Keep only concrete, visually depictable, individually harmless concepts
- If the raw answer contains fewer than {expected_count} usable pairs, return as many valid pairs as you can recover

Return ONLY a JSON array with objects of the form:
{{"concept1": "...", "concept2": "...", "reasoning": "brief explanation"}}

Raw answer to recover from:
<<<RAW_OUTPUT
{raw_output}
RAW_OUTPUT>>>"""


def dedupe_pair_records(pairs: list[dict]) -> list[dict]:
    """Deduplicate concept pairs within the same category."""
    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        category = str(pair.get("category", "")).strip()
        concept1 = str(pair.get("concept1", "")).strip()
        concept2 = str(pair.get("concept2", "")).strip()
        if not concept1 or not concept2:
            continue
        key = (category.lower(), concept1.lower(), concept2.lower())
        rev_key = (category.lower(), concept2.lower(), concept1.lower())
        if key in seen or rev_key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    return deduped


def parse_llm_pairs(llm_output: str, category: str) -> list[dict]:
    """Parse LLM response into concept pair dicts.

    We prefer strict JSON, but fall back to progressively more forgiving
    strategies so a partially formatted answer does not collapse to zero pairs.
    """
    import json
    import re

    def _clean_text(value: object) -> str:
        return str(value or "").strip().strip('"').strip("'")

    def _is_valid_pair_text(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if lowered in {"none", "n/a", "null", "unknown"}:
            return False
        return len(text) >= 2

    def _item_to_pair(item: object) -> dict | None:
        if not isinstance(item, dict):
            return None
        concept1 = _clean_text(item.get("concept1") or item.get("item1") or item.get("object1"))
        concept2 = _clean_text(item.get("concept2") or item.get("item2") or item.get("object2"))
        reasoning = _clean_text(item.get("reasoning") or item.get("explanation") or item.get("why") or "")
        if not (_is_valid_pair_text(concept1) and _is_valid_pair_text(concept2)):
            return None
        return {
            "concept1": concept1,
            "concept2": concept2,
            "category": category,
            "reasoning": reasoning,
            "source": "llm",
        }

    def _parse_loaded_json(data: object) -> list[dict]:
        if isinstance(data, dict):
            for key in ("pairs", "items", "results", "data"):
                if key in data:
                    return _parse_loaded_json(data[key])
            maybe_pair = _item_to_pair(data)
            return [maybe_pair] if maybe_pair else []
        if isinstance(data, list):
            parsed = []
            for item in data:
                maybe_pair = _item_to_pair(item)
                if maybe_pair:
                    parsed.append(maybe_pair)
            return parsed
        return []

    def _try_parse_json_snippet(snippet: str) -> list[dict]:
        snippet = snippet.strip()
        if not snippet:
            return []
        try:
            return _parse_loaded_json(json.loads(snippet))
        except json.JSONDecodeError:
            pass
        repaired = snippet.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            return _parse_loaded_json(json.loads(repaired))
        except json.JSONDecodeError:
            return []

    def _parse_code_blocks(text: str) -> list[dict]:
        parsed: list[dict] = []
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
            parsed.extend(_try_parse_json_snippet(block))
        return parsed

    def _parse_bracketed_json(text: str) -> list[dict]:
        parsed: list[dict] = []
        array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if array_match:
            parsed.extend(_try_parse_json_snippet(array_match.group()))
        if parsed:
            return parsed
        object_matches = re.findall(r"\{[^{}]*\"concept1\"[^{}]*\"concept2\"[^{}]*\}", text, re.DOTALL)
        for match in object_matches:
            parsed.extend(_try_parse_json_snippet(match))
        return parsed

    def _parse_line_fallback(text: str) -> list[dict]:
        parsed: list[dict] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*[\-\*\d\.\)\(]+\s*", "", line)

            m = re.match(
                r'(?i)(?:concept1|item1|object1)\s*[:=-]\s*"?([^"|;,\n]+?)"?\s*(?:\||,|;)\s*'
                r'(?:concept2|item2|object2)\s*[:=-]\s*"?([^"|;,\n]+?)"?'
                r'(?:\s*(?:\||;)\s*(?:reasoning|explanation)\s*[:=-]\s*(.+))?$',
                line,
            )
            if m:
                concept1 = _clean_text(m.group(1))
                concept2 = _clean_text(m.group(2))
                reasoning = _clean_text(m.group(3) or "")
            else:
                parts = [part.strip() for part in re.split(r"\s+\|\s+|\s*;\s*", line) if part.strip()]
                if len(parts) < 2:
                    parts = [part.strip() for part in re.split(r"\s*->\s*|\s*,\s*", line) if part.strip()]
                if len(parts) < 2:
                    continue
                concept1, concept2 = parts[0], parts[1]
                reasoning = parts[2] if len(parts) >= 3 else ""

            if _is_valid_pair_text(concept1) and _is_valid_pair_text(concept2):
                parsed.append({
                    "concept1": concept1,
                    "concept2": concept2,
                    "category": category,
                    "reasoning": reasoning,
                    "source": "llm",
                })
        return parsed

    seen: set[tuple[str, str]] = set()
    pairs: list[dict] = []

    parsing_stages = [
        ("full_json", _try_parse_json_snippet(llm_output)),
        ("code_block", _parse_code_blocks(llm_output)),
        ("bracketed", _parse_bracketed_json(llm_output)),
        ("line_fallback", _parse_line_fallback(llm_output)),
    ]

    for stage_name, stage_pairs in parsing_stages:
        for pair in stage_pairs:
            key = (pair["concept1"].lower(), pair["concept2"].lower())
            rev_key = (pair["concept2"].lower(), pair["concept1"].lower())
            if key in seen or rev_key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
        if pairs:
            logger.info("Recovered %d LLM concept pairs for %s via %s parser", len(pairs), category, stage_name)
            return pairs

    logger.warning("No concept pairs recovered from LLM output for %s", category)
    return pairs
