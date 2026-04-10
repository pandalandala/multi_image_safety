"""Step 2: Decompose malicious prompts into pairs of benign image descriptions.

Uses LLM (local vLLM or API) to split a harmful prompt into two individually
harmless image descriptions that, when combined, reconstruct the harmful intent.
Inspired by "Jailbreak in Pieces" (ICLR 2024) and DSG scene graph decomposition.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.common.schema import HarmCategory, Pattern, SourcePath
from src.common.utils import (
    get_effective_tensor_parallel_size,
    load_config,
    load_jsonl,
    load_prompt_template,
    save_jsonl,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def _build_system_message() -> str:
    return (
        "You are a safety research assistant helping to construct a multi-image "
        "safety dataset for VLM alignment. Respond with valid JSON only. "
        "Do NOT include any thinking, reasoning, or explanation outside the JSON block. "
        "Output ONLY the JSON object, nothing else."
    )


def build_decompose_prompt(malicious_prompt: str, category: str) -> str:
    """Build the LLM prompt for decomposing a malicious prompt into image descriptions."""
    template = load_prompt_template("decompose_prompt.txt")
    return template.format(
        malicious_prompt=malicious_prompt,
        category=category,
    )


def parse_decompose_response(response: str) -> Optional[dict]:
    """Parse LLM response to extract image descriptions and text prompt.

    Handles Qwen3.5 thinking mode output by stripping everything before the JSON block.
    """
    try:
        # Strip thinking tokens if present (Qwen3.5 may output thinking despite /no_think)
        # Look for the last complete JSON object in the response
        # Find all { ... } blocks and try parsing from the outermost one
        brace_depth = 0
        json_start = -1
        json_end = -1

        for i, c in enumerate(response):
            if c == '{':
                if brace_depth == 0:
                    json_start = i
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
                if brace_depth == 0 and json_start >= 0:
                    json_end = i + 1
                    # Try to parse this block
                    try:
                        candidate = response[json_start:json_end]
                        data = json.loads(candidate)
                        required_keys = ["image1_description", "image2_description", "text_prompt"]
                        if all(k in data for k in required_keys):
                            return {
                                "image1_description": data["image1_description"],
                                "image2_description": data["image2_description"],
                                "text_prompt": data["text_prompt"],
                                "reasoning": data.get("reasoning", ""),
                                "covertness_estimate": data.get("covertness_estimate", 3),
                            }
                    except json.JSONDecodeError:
                        pass
                    # Reset to look for next JSON block
                    json_start = -1
    except Exception as e:
        logger.debug(f"Failed to parse response: {e}")
    return None


def _truncate_text_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate plain text to a token budget using the model tokenizer."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()


def _chat_token_count(tokenizer, conversation: list[dict]) -> int:
    """Best-effort token count for a chat conversation."""
    try:
        token_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        try:
            token_ids = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return len(token_ids)


def _prepare_conversation(
    tokenizer,
    prompt_row: dict,
    system_message: str,
    max_input_tokens: int,
    max_prompt_text_tokens: int,
) -> tuple[Optional[list[dict]], bool, int]:
    """Build a conversation, truncating overly long raw prompts if needed."""
    prompt_text = str(prompt_row["text"])
    category = prompt_row["category"]
    truncated = False

    # Fast first-pass cap on the raw harmful prompt itself.
    prompt_text = _truncate_text_tokens(tokenizer, prompt_text, max_prompt_text_tokens)
    if prompt_text != prompt_row["text"]:
        truncated = True

    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": build_decompose_prompt(prompt_text, category)},
    ]
    token_count = _chat_token_count(tokenizer, conversation)
    if token_count <= max_input_tokens:
        return conversation, truncated, token_count

    # If still too long, iteratively shrink the raw harmful prompt.
    raw_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    low, high = 1, len(raw_ids)
    best_conversation = None
    best_count = token_count
    while low <= high:
        mid = (low + high) // 2
        candidate_text = tokenizer.decode(raw_ids[:mid], skip_special_tokens=True).strip()
        candidate_conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": build_decompose_prompt(candidate_text, category)},
        ]
        candidate_count = _chat_token_count(tokenizer, candidate_conversation)
        if candidate_count <= max_input_tokens:
            best_conversation = candidate_conversation
            best_count = candidate_count
            low = mid + 1
        else:
            high = mid - 1

    if best_conversation is None:
        return None, True, token_count
    return best_conversation, True, best_count


def decompose_with_vllm(
    prompts: list[dict],
    model_path: str,
    tensor_parallel_size: int = 2,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    batch_size: int = 256,
) -> list[dict]:
    """Decompose prompts using local vLLM inference with chat template."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,  # Skip torch.compile for hybrid architectures (Qwen3.5 Mamba2)
        disable_custom_all_reduce=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )

    # Preflight token-length checks so one pathological sample can't kill the whole run.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    system_message = _build_system_message()
    max_input_tokens = max_model_len - 64
    max_prompt_text_tokens = min(4096, max_input_tokens // 2)

    prepared_items = []
    skipped_too_long = 0
    truncated_count = 0
    for i, prompt_row in enumerate(prompts):
        conversation, truncated, token_count = _prepare_conversation(
            tokenizer,
            prompt_row,
            system_message,
            max_input_tokens=max_input_tokens,
            max_prompt_text_tokens=max_prompt_text_tokens,
        )
        if conversation is None:
            skipped_too_long += 1
            logger.warning(
                "Skipping prompt %d because it still exceeds the context window after truncation",
                i,
            )
            continue
        if truncated:
            truncated_count += 1
        prepared_items.append((i, prompt_row, conversation, token_count))

    logger.info(
        "Prepared %d/%d prompts for decomposition (%d truncated, %d skipped for length)",
        len(prepared_items),
        len(prompts),
        truncated_count,
        skipped_too_long,
    )

    results = []
    for start in range(0, len(prepared_items), batch_size):
        batch = prepared_items[start : start + batch_size]
        conversations = [item[2] for item in batch]
        try:
            outputs = llm.chat(
                conversations,
                sampling_params,
                chat_template_kwargs={"enable_thinking": False},
            )
            batch_pairs = zip(batch, outputs)
        except Exception as e:
            logger.warning(
                "Batch %d-%d failed during decompose chat (%s). Retrying one-by-one.",
                start,
                start + len(batch) - 1,
                e,
            )
            batch_pairs = []
            for item in batch:
                try:
                    single_output = llm.chat(
                        [item[2]],
                        sampling_params,
                        chat_template_kwargs={"enable_thinking": False},
                    )[0]
                    batch_pairs.append((item, single_output))
                except Exception as single_e:
                    logger.warning(
                        "Skipping prompt %d after individual retry failure: %s",
                        item[0],
                        single_e,
                    )

        for item, output in batch_pairs:
            prompt_index, prompt_row = item[0], item[1]
            response = output.outputs[0].text
            parsed = parse_decompose_response(response)
            if parsed:
                results.append({
                    **prompt_row,
                    **parsed,
                    "pattern": Pattern.A.value,
                    "source_path": SourcePath.PATH2.value,
                })
            else:
                logger.debug(f"Failed to parse decomposition for prompt {prompt_index}")

    logger.info(f"Successfully decomposed {len(results)}/{len(prompts)} prompts")
    return results


def decompose_with_api(
    prompts: list[dict],
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key_env: str = "OPENAI_API_KEY",
    max_concurrent: int = 10,
) -> list[dict]:
    """Decompose prompts using API-based LLM."""
    import os
    import openai

    client = openai.OpenAI(api_key=os.environ.get(api_key_env))
    results = []

    for i, p in enumerate(prompts):
        try:
            decompose_prompt = build_decompose_prompt(p["text"], p["category"])
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a safety research assistant helping to construct a multi-image safety dataset for VLM alignment. Respond with valid JSON only."},
                    {"role": "user", "content": decompose_prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            text = response.choices[0].message.content
            parsed = parse_decompose_response(text)
            if parsed:
                results.append({
                    **p,
                    **parsed,
                    "pattern": Pattern.A.value,
                    "source_path": SourcePath.PATH2.value,
                })
        except Exception as e:
            logger.warning(f"API call failed for prompt {i}: {e}")

        if (i + 1) % 100 == 0:
            logger.info(f"Decomposed {i + 1}/{len(prompts)} prompts")

    logger.info(f"Successfully decomposed {len(results)}/{len(prompts)} prompts via API")
    return results


def run(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_api: bool = False,
    max_prompts: int = 4500,
):
    """Main entry point: decompose collected prompts."""
    if input_file is None:
        input_file = DATA_DIR / "raw" / "path2" / "collected_prompts.jsonl"
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path2"

    prompts = load_jsonl(input_file)
    if len(prompts) > max_prompts:
        import random
        random.shuffle(prompts)
        prompts = prompts[:max_prompts]

    config = load_config()

    if use_api:
        api_config = config["llm"]["api"]
        results = decompose_with_api(
            prompts,
            provider=api_config["provider"],
            model=api_config["model"],
            api_key_env=api_config["api_key_env"],
        )
    else:
        local_config = config["llm"]["local"]
        max_model_len = min(
            int(local_config.get("max_model_len", 8192)),
            int(os.environ.get("MIS_PATH2_MAX_MODEL_LEN", "4096")),
        )
        gpu_memory_utilization = min(
            float(local_config.get("gpu_memory_utilization", 0.9)),
            float(os.environ.get("MIS_PATH2_GPU_MEMORY_UTILIZATION", "0.68")),
        )
        batch_size = min(
            256,
            int(os.environ.get("MIS_PATH2_VLLM_BATCH_SIZE", "64")),
        )
        results = decompose_with_vllm(
            prompts,
            model_path=local_config["model_path"],
            tensor_parallel_size=get_effective_tensor_parallel_size(
                local_config.get("tensor_parallel_size")
            ),
            max_model_len=max_model_len,
            max_tokens=local_config.get("max_tokens", 2048),
            temperature=local_config["temperature"],
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
        )

    save_jsonl(results, Path(output_dir) / "decomposed_prompts.jsonl")
    return results


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
