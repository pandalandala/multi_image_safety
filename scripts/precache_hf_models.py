#!/usr/bin/env python3
"""Pre-download Hugging Face models used by this project into the shared cache."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.utils import configure_hf_download_debug, load_config, setup_logging


def collect_model_ids(cfg: dict) -> list[str]:
    models: list[str] = []

    llm_local = cfg.get("llm", {}).get("local", {})
    if llm_local.get("model_path"):
        models.append(llm_local["model_path"])
    elif llm_local.get("model_name"):
        models.append(llm_local["model_name"])

    image_cfg = cfg.get("image_generation", {})
    for backend_cfg in image_cfg.get("backends", {}).values():
        model_id = backend_cfg.get("model")
        if model_id:
            models.append(model_id)

    safety_cfg = cfg.get("quality", {}).get("safety_check", {})
    for key in ("nsfw_model", "llama_guard_model"):
        model_id = safety_cfg.get(key)
        if model_id:
            models.append(model_id)

    seen: set[str] = set()
    unique: list[str] = []
    for model_id in models:
        if model_id not in seen:
            seen.add(model_id)
            unique.append(model_id)
    return unique


def main() -> int:
    setup_logging()
    cfg = load_config()
    hf_home = Path(cfg.get("hf_home", "/mnt2/xuran_hdd/cache"))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("MIS_HF_DOWNLOAD_VERBOSE", os.environ.get("MIS_HF_DOWNLOAD_VERBOSE", "0"))
    configure_hf_download_debug()

    token_path = hf_home / "token"
    if token_path.exists():
        os.environ.setdefault("HF_TOKEN", token_path.read_text(encoding="utf-8").strip())

    from huggingface_hub import snapshot_download

    models = collect_model_ids(cfg)
    results: list[dict] = []

    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"Preparing to cache {len(models)} Hugging Face models")

    for model_id in models:
        print(f"\n=== Caching {model_id} ===", flush=True)
        try:
            snapshot_path = snapshot_download(
                repo_id=model_id,
                token=os.environ.get("HF_TOKEN"),
                resume_download=True,
            )
            print(f"OK {model_id} -> {snapshot_path}", flush=True)
            results.append({"model": model_id, "status": "ok", "path": snapshot_path})
        except Exception as exc:  # pragma: no cover - network/gated failures are runtime concerns
            print(f"FAIL {model_id}: {exc}", flush=True)
            results.append({"model": model_id, "status": "failed", "error": str(exc)})

    out_path = PROJECT_ROOT / "logs" / "hf_precache_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote manifest to {out_path}")

    failed = [r for r in results if r["status"] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
