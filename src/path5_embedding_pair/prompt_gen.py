"""Step 3: Generate text prompts for embedding-paired images using LLM.

Given a pair of images (with captions), generate a natural text prompt
that connects them in a potentially harmful way.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.common.utils import (
    get_effective_tensor_parallel_size,
    is_english,
    load_config,
    load_prompt_template,
    load_jsonl,
    save_jsonl,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def build_prompt_gen_input(
    image1_caption: str,
    image2_caption: str,
    category: str,
) -> str:
    """Build the LLM prompt for generating a text prompt for an image pair."""
    template = load_prompt_template("pair_prompt_gen.txt")
    return template.format(
        image1_caption=image1_caption,
        image2_caption=image2_caption,
        category=category,
    )


def parse_prompt_gen_response(response: str) -> Optional[dict]:
    """Parse LLM response to extract generated prompt and CoT."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > 0:
            data = json.loads(response[start:end])
            if "text_prompt" in data:
                return {
                    "text_prompt": data["text_prompt"],
                    "reasoning": data.get("reasoning", ""),
                    "covertness_estimate": data.get("covertness_estimate", 3),
                    "prompt_style": data.get("prompt_style", "inquiry"),
                }
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def generate_prompts_vllm(
    pairs: list[dict],
    model_path: str,
    tensor_parallel_size: int = 2,
) -> list[dict]:
    """Generate text prompts for all pairs using local vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=8192,
        enforce_eager=True,  # Skip torch.compile for Qwen3.5 Mamba2 hybrid
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )

    llm_prompts = []
    valid_pairs = []
    for pair in pairs:
        cap1 = pair.get("image1_caption", pair.get("image1_query", ""))
        cap2 = pair.get("image2_caption", pair.get("image2_query", ""))
        # Skip pairs with non-English captions
        if (cap1 and not is_english(cap1)) or (cap2 and not is_english(cap2)):
            continue
        p = build_prompt_gen_input(cap1, cap2, pair["category"])
        llm_prompts.append(p)
        valid_pairs.append(pair)
    if len(valid_pairs) < len(pairs):
        logger.info(f"Filtered {len(pairs) - len(valid_pairs)} pairs with non-English captions")
    pairs = valid_pairs

    conversations = [[{"role": "user", "content": p}] for p in llm_prompts]
    outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={"enable_thinking": False})

    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        parsed = parse_prompt_gen_response(response)
        if parsed:
            if not is_english(parsed.get("text_prompt", "")):
                continue
            pair = pairs[i].copy()
            pair.update(parsed)
            results.append(pair)

    logger.info(f"Generated prompts for {len(results)}/{len(pairs)} pairs")
    return results


def generate_prompts_api(
    pairs: list[dict],
) -> list[dict]:
    """Generate text prompts using API."""
    import os
    import openai

    config = load_config()
    api_config = config["llm"]["api"]
    client = openai.OpenAI(api_key=os.environ.get(api_config["api_key_env"]))

    results = []
    for i, pair in enumerate(pairs):
        # Skip pairs with non-English captions
        cap1 = pair.get("image1_caption", pair.get("image1_query", ""))
        cap2 = pair.get("image2_caption", pair.get("image2_query", ""))
        if (cap1 and not is_english(cap1)) or (cap2 and not is_english(cap2)):
            continue
        try:
            prompt = build_prompt_gen_input(cap1, cap2, pair["category"])
            response = client.chat.completions.create(
                model=api_config["model"],
                messages=[
                    {"role": "system", "content": "You are a safety research assistant. Generate natural text prompts for multi-image safety evaluation. Respond in JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            text = response.choices[0].message.content
            parsed = parse_prompt_gen_response(text)
            if parsed:
                if not is_english(parsed.get("text_prompt", "")):
                    continue
                p = pair.copy()
                p.update(parsed)
                results.append(p)
        except Exception as e:
            logger.warning(f"API failed for pair {i}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i+1}/{len(pairs)} prompts")

    return results


def run(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_api: bool = False,
):
    """Main entry point."""
    if input_file is None:
        input_file = DATA_DIR / "raw" / "path5" / "embedding_paired_samples.jsonl"
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path5"

    pairs = load_jsonl(input_file)
    config = load_config()

    if use_api:
        results = generate_prompts_api(pairs)
    else:
        local_config = config["llm"]["local"]
        results = generate_prompts_vllm(
            pairs,
            model_path=local_config["model_path"],
            tensor_parallel_size=get_effective_tensor_parallel_size(
                local_config.get("tensor_parallel_size")
            ),
        )

    save_jsonl(results, Path(output_dir) / "samples_with_prompts.jsonl")
    logger.info(f"Generated prompts for {len(results)} samples")
    return results


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
