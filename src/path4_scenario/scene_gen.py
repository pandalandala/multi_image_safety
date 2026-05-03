"""Step 1: Generate everyday life scenes using LLM.

Inspired by MSSBench and VLSU approaches - generate realistic daily scenarios
that appear benign but can be contextualized with dangerous intent.
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
    save_jsonl,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


# Predefined scene categories for systematic coverage (80 categories)
SCENE_CATEGORIES = [
    # Original 40
    "kitchen", "laboratory", "factory", "construction_site", "hospital",
    "school", "office", "street", "park", "subway", "airport", "shopping_mall",
    "parking_lot", "gym", "swimming_pool", "restaurant", "bar", "pharmacy",
    "gas_station", "warehouse", "farm", "forest", "beach", "bridge",
    "rooftop", "basement", "garage", "playground", "stadium", "train_station",
    "library", "museum", "church", "bank", "post_office", "pet_store",
    "hardware_store", "electronics_store", "grocery_store", "laundromat",
    # Additional 40
    "coffee_shop", "hotel_lobby", "convention_center", "art_studio", "music_studio",
    "climbing_wall", "bowling_alley", "billiard_room", "video_arcade", "escape_room",
    "barbershop", "nail_salon", "spa", "dry_cleaner", "tattoo_parlor",
    "car_wash", "tire_shop", "bike_shop", "marina", "fishing_pier",
    "greenhouse", "vineyard", "dairy_farm", "butcher_shop", "fish_market",
    "food_truck", "catering_kitchen", "brewery", "distillery", "bakery",
    "solar_farm", "power_plant", "water_treatment", "recycling_center", "data_center",
    "radio_station", "print_shop", "photo_studio", "film_set", "recording_booth",
]


def build_scene_gen_prompt(scene_category: str, num_activities: int = 20) -> str:
    """Build the LLM prompt for generating scene-activity pairs."""
    template = load_prompt_template("scene_gen_prompt.txt")
    return template.format(
        scene_category=scene_category,
        num_activities=num_activities,
    )


def parse_scene_response(response: str) -> list[dict]:
    """Parse LLM response to extract scene-activity pairs."""
    scenes = []
    try:
        # Try JSON array
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > 0:
            data = json.loads(response[start:end])
            for item in data:
                if isinstance(item, dict) and "activity" in item:
                    scenes.append(item)
                elif isinstance(item, str):
                    scenes.append({"activity": item, "objects": []})
    except json.JSONDecodeError:
        # Fall back to line-by-line parsing
        for line in response.split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line and len(line) > 5:
                scenes.append({"activity": line, "objects": []})

    return scenes


def generate_scenes_vllm(
    model_path: str,
    tensor_parallel_size: int = 2,
    num_activities_per_scene: int = 25,
) -> list[dict]:
    """Generate scenes using local vLLM."""
    from vllm import LLM, SamplingParams

    from src.common.utils import get_safe_vllm_kwargs
    llm = LLM(**get_safe_vllm_kwargs(
        "path4",
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
    ))
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=4096,
        top_p=0.95,
    )

    prompts = [
        build_scene_gen_prompt(cat, num_activities_per_scene)
        for cat in SCENE_CATEGORIES
    ]
    conversations = [
        [{"role": "user", "content": p}]
        for p in prompts
    ]
    outputs = llm.chat(conversations, sampling_params, chat_template_kwargs={"enable_thinking": False})

    all_scenes = []
    skipped_lang = 0
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        activities = parse_scene_response(response)
        for act in activities:
            activity_text = act.get("activity", "")
            if activity_text and not is_english(activity_text):
                skipped_lang += 1
                continue
            all_scenes.append({
                "scene_category": SCENE_CATEGORIES[i],
                "activity": activity_text,
                "objects": act.get("objects", []),
                "setting_description": act.get("setting", ""),
            })
    if skipped_lang:
        logger.info(f"Skipped {skipped_lang} non-English activities")

    logger.info(
        f"Generated {len(all_scenes)} scene-activity pairs "
        f"across {len(SCENE_CATEGORIES)} scene categories"
    )
    return all_scenes


def generate_scenes_api(
    num_activities_per_scene: int = 25,
) -> list[dict]:
    """Generate scenes using API."""
    import os
    import openai

    config = load_config()
    api_config = config["llm"]["api"]
    client = openai.OpenAI(api_key=os.environ.get(api_config["api_key_env"]))

    all_scenes = []
    for cat in SCENE_CATEGORIES:
        try:
            prompt = build_scene_gen_prompt(cat, num_activities_per_scene)
            response = client.chat.completions.create(
                model=api_config["model"],
                messages=[
                    {"role": "system", "content": "You are helping to generate realistic everyday scenarios for safety research. Respond in JSON format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=4096,
            )
            text = response.choices[0].message.content
            activities = parse_scene_response(text)
            for act in activities:
                activity_text = act.get("activity", "")
                if activity_text and not is_english(activity_text):
                    continue
                all_scenes.append({
                    "scene_category": cat,
                    "activity": activity_text,
                    "objects": act.get("objects", []),
                    "setting_description": act.get("setting", ""),
                })
        except Exception as e:
            logger.warning(f"API failed for scene {cat}: {e}")

    logger.info(f"Generated {len(all_scenes)} scene-activity pairs via API")
    return all_scenes


def run(
    output_dir: str | Path | None = None,
    use_api: bool = False,
):
    """Main entry point: generate scene-activity pairs."""
    if output_dir is None:
        output_dir = DATA_DIR / "raw" / "path4"
    output_dir = Path(output_dir)

    config = load_config()

    if use_api:
        scenes = generate_scenes_api()
    else:
        local_config = config["llm"]["local"]
        scenes = generate_scenes_vllm(
            model_path=local_config["model_path"],
            tensor_parallel_size=get_effective_tensor_parallel_size(
                local_config.get("tensor_parallel_size")
            ),
        )

    save_jsonl(scenes, output_dir / "generated_scenes.jsonl")
    return scenes


if __name__ == "__main__":
    from src.common.utils import setup_logging
    setup_logging()
    run()
