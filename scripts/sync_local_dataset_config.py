#!/usr/bin/env python3
"""Enable/disable local dataset flags based on actual downloaded files."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path("/mnt/hdd/xuran/multi_image_safety")
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


def has_any_image_files(root: Path) -> bool:
    if not root.exists():
        return False
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPEG"):
        if next(root.rglob(pattern), None) is not None:
            return True
    return False


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    local_cfg = config.setdefault("local_datasets", {})
    dataset_root = Path(local_cfg.get("root", "/mnt2/xuran_hdd/cache/datasets"))

    coco_cfg = local_cfg.setdefault("mscoco", {})
    open_images_cfg = local_cfg.setdefault("open_images", {})
    imagenet_cfg = local_cfg.setdefault("imagenet", {})

    coco_images_dir = dataset_root / coco_cfg.get("images_dir", "coco/train2017")
    coco_ann = dataset_root / coco_cfg.get("annotations", "coco/annotations/captions_train2017.json")
    open_images_dir = dataset_root / open_images_cfg.get("images_dir", "open_images/train")
    open_images_labels = dataset_root / open_images_cfg.get("labels_csv", "open_images/oidv7-class-descriptions.csv")
    open_images_ann = dataset_root / open_images_cfg.get("annotations_csv", "open_images/oidv7-train-annotations-human-imagelabels.csv")
    imagenet_images_dir = dataset_root / imagenet_cfg.get("images_dir", "imagenet/ILSVRC/Data/CLS-LOC/train")
    imagenet_synsets = dataset_root / imagenet_cfg.get("synsets_file", "imagenet/LOC_synset_mapping.txt")

    coco_ready = coco_images_dir.is_dir() and coco_ann.is_file() and has_any_image_files(coco_images_dir)
    open_images_ready = (
        open_images_dir.is_dir()
        and open_images_labels.is_file()
        and open_images_ann.is_file()
        and has_any_image_files(open_images_dir)
    )
    imagenet_ready = imagenet_images_dir.is_dir() and imagenet_synsets.is_file() and has_any_image_files(imagenet_images_dir)

    coco_cfg["enabled"] = bool(coco_ready)
    open_images_cfg["enabled"] = bool(open_images_ready)
    imagenet_cfg["enabled"] = bool(imagenet_ready)
    local_cfg["enabled"] = bool(coco_ready or open_images_ready or imagenet_ready)

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"Config updated: {CONFIG_PATH}")
    print(f"  mscoco.enabled={coco_cfg['enabled']} ({coco_images_dir})")
    print(f"  open_images.enabled={open_images_cfg['enabled']} ({open_images_dir})")
    print(f"  imagenet.enabled={imagenet_cfg['enabled']} ({imagenet_images_dir})")
    print(f"  local_datasets.enabled={local_cfg['enabled']}")


if __name__ == "__main__":
    main()
