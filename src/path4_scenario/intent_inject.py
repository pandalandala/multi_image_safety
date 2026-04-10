"""Step 2: Inject dangerous intent into benign scenes.

References VLSU's 17 safety modes × 15 harm categories combinatorial matrix.
For each scene-activity pair, generate image pair descriptions where the images
appear harmless but careful reasoning reveals unsafe intent.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.common.schema import HarmCategory, Pattern, SourcePath
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

# VLSU-inspired safety modes (17 types)
SAFETY_MODES = [
    "risk_identification",       # Identify potential risks in a scenario
    "safe_alternative",          # Ask for safer alternative to dangerous action
    "consequence_reasoning",     # Reason about consequences of actions
    "instruction_following",     # Follow potentially dangerous instructions
    "role_play",                 # Role-play as someone in the scenario
    "hypothetical_scenario",     # "What if" dangerous scenarios
    "comparison_judgment",       # Compare safety of different approaches
    "temporal_reasoning",        # Reason about before/after dangerous events
    "causal_reasoning",          # Determine cause-effect in harmful scenarios
    "tool_misuse",               # How tools/objects could be misused
    "social_engineering",        # Manipulation through social context
    "environmental_hazard",      # Environmental dangers in scenes
    "regulatory_violation",      # Actions that violate regulations
    "emergency_response",        # How to respond to emergencies
    "dual_use_reasoning",        # Objects with both benign and harmful uses
    "implicit_harm",             # Harm that requires inference to identify
    "contextual_sensitivity",    # Context-dependent safety concerns
]


def build_intent_inject_prompt(
    scene_category: str,
    activity: str,
    safety_mode: str,
    harm_category: str,
) -> str:
    """Build the LLM prompt for injecting dangerous intent."""
    template = load_prompt_template("intent_inject_prompt.txt")
    return template.format(
        scene_category=scene_category,
        activity=activity,
        safety_mode=safety_mode,
        harm_category=harm_category,
    )


def parse_intent_response(response: str) -> Optional[dict]:
    """Parse LLM response for intent injection."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > 0:
            data = json.loads(response[start:end])
            required = ["image1_description", "image2_description", "text_prompt"]
            if all(k in data for k in required):
                return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def inject_intents_batch(
    scenes: list[dict],
    harm_categories: list[str] | None = None,
    safety_modes: list[str] | None = None,
    max_combinations: int = 3000,
) -> list[str]:
    """
    Generate batch prompts for intent injection.
    Crosses scene-activity pairs with safety modes and harm categories.
    """
    if harm_categories is None:
        harm_categories = [c.value for c in HarmCategory]
    if safety_modes is None:
        safety_modes = SAFETY_MODES

    prompts_with_meta = []
    import random

    for scene in scenes:
        # Randomly sample a few safety modes and harm categories per scene
        modes = random.sample(safety_modes, min(3, len(safety_modes)))
        cats = random.sample(harm_categories, min(2, len(harm_categories)))

        for mode in modes:
            for cat in cats:
                prompt = build_intent_inject_prompt(
                    scene["scene_category"],
                    scene["activity"],
                    mode,
                    cat,
                )
                prompts_with_meta.append({
                    "prompt": prompt,
                    "scene_category": scene["scene_category"],
                    "activity": scene["activity"],
                    "safety_mode": mode,
                    "harm_category": cat,
                })

                if len(prompts_with_meta) >= max_combinations:
                    return prompts_with_meta

    return prompts_with_meta


def process_intent_responses(
    prompts_with_meta: list[dict],
    responses: list[str],
) -> list[dict]:
    """Process LLM responses and build sample entries."""
    results = []
    skipped_lang = 0
    for meta, response in zip(prompts_with_meta, responses):
        parsed = parse_intent_response(response)
        if parsed:
            # Skip if any text field is non-English
            if not is_english(parsed["text_prompt"]):
                skipped_lang += 1
                continue
            if not is_english(parsed["image1_description"]) or not is_english(parsed["image2_description"]):
                skipped_lang += 1
                continue
            results.append({
                "scene_category": meta["scene_category"],
                "activity": meta["activity"],
                "safety_mode": meta["safety_mode"],
                "category": meta["harm_category"],
                "image1_description": parsed["image1_description"],
                "image2_description": parsed["image2_description"],
                "text_prompt": parsed["text_prompt"],
                "reasoning": parsed.get("reasoning", ""),
                "covertness_estimate": parsed.get("covertness_estimate", 3),
                "pattern": Pattern.D.value,  # Contextualized harm
                "source_path": SourcePath.PATH4.value,
            })
    if skipped_lang:
        logger.info(f"Skipped {skipped_lang} non-English intent injection results")
    return results


def run_vllm(
    scenes: list[dict],
    model_path: str,
    tensor_parallel_size: int = 2,
    max_combinations: int = 3000,
) -> list[dict]:
    """Run intent injection using local vLLM."""
    from vllm import LLM, SamplingParams

    prompts_with_meta = inject_intents_batch(scenes, max_combinations=max_combinations)
    logger.info(f"Generated {len(prompts_with_meta)} intent injection prompts")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=8192,
        enforce_eager=True,  # Skip torch.compile for Qwen3.5 Mamba2 hybrid
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
    )

    raw_prompts = [p["prompt"] for p in prompts_with_meta]
    conversations = [[{"role": "user", "content": p}] for p in raw_prompts]
    outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={"enable_thinking": False})
    responses = [o.outputs[0].text for o in outputs]

    results = process_intent_responses(prompts_with_meta, responses)
    logger.info(f"Successfully generated {len(results)} intent-injected samples")
    return results


def run(
    input_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_api: bool = False,
    max_combinations: int = 3000,
):
    """Main entry point."""
    if input_file is None:
        input_file = DATA_DIR / "raw" / "path4" / "generated_scenes.jsonl"
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path4"

    scenes = load_jsonl(input_file)
    config = load_config()

    if use_api:
        # API-based: iterate and call individually
        prompts_with_meta = inject_intents_batch(scenes, max_combinations=max_combinations)
        save_jsonl(prompts_with_meta, Path(output_dir) / "intent_batch_prompts.jsonl")
        logger.info(f"Saved {len(prompts_with_meta)} prompts for API processing")
        return prompts_with_meta
    else:
        local_config = config["llm"]["local"]
        results = run_vllm(
            scenes,
            model_path=local_config["model_path"],
            tensor_parallel_size=get_effective_tensor_parallel_size(
                local_config.get("tensor_parallel_size")
            ),
            max_combinations=max_combinations,
        )
        save_jsonl(results, Path(output_dir) / "intent_injected_samples.jsonl")
        return results


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
