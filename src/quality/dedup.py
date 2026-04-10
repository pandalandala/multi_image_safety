"""Deduplication: SHA-256 exact, pHash perceptual, SemDeDup semantic."""

import hashlib
import logging
from pathlib import Path

import numpy as np

from src.common.utils import sha256_file

logger = logging.getLogger(__name__)


def exact_dedup(samples: list[dict]) -> list[dict]:
    """Remove samples with identical image files (SHA-256 hash)."""
    seen_hashes = set()
    unique = []

    for sample in samples:
        img1 = sample.get("image1_path", "")
        img2 = sample.get("image2_path", "")

        if not img1 or not img2:
            continue

        try:
            h1 = sha256_file(img1)
            h2 = sha256_file(img2)
            pair_key = tuple(sorted([h1, h2]))

            if pair_key not in seen_hashes:
                seen_hashes.add(pair_key)
                unique.append(sample)
        except FileNotFoundError:
            continue

    removed = len(samples) - len(unique)
    logger.info(f"Exact dedup: {removed} removed, {len(unique)} remaining")
    return unique


def perceptual_dedup(
    samples: list[dict],
    hash_size: int = 16,
    threshold: int = 8,
) -> list[dict]:
    """Remove near-duplicate image pairs using perceptual hashing (pHash)."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        logger.warning("imagehash not available, skipping perceptual dedup")
        return samples

    def compute_phash(image_path: str) -> str:
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img, hash_size=hash_size))
        except Exception:
            return ""

    # Compute hashes for all image pairs
    pair_hashes = []
    for sample in samples:
        h1 = compute_phash(sample.get("image1_path", ""))
        h2 = compute_phash(sample.get("image2_path", ""))
        pair_hashes.append((h1, h2))

    # Remove pairs that are too similar to existing ones
    unique = []
    seen_pairs = []

    for i, (h1, h2) in enumerate(pair_hashes):
        if not h1 or not h2:
            unique.append(samples[i])
            continue

        is_dup = False
        for sh1, sh2 in seen_pairs:
            try:
                dist1 = imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(sh1)
                dist2 = imagehash.hex_to_hash(h2) - imagehash.hex_to_hash(sh2)
                if dist1 < threshold and dist2 < threshold:
                    is_dup = True
                    break
                # Also check swapped order
                dist1s = imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(sh2)
                dist2s = imagehash.hex_to_hash(h2) - imagehash.hex_to_hash(sh1)
                if dist1s < threshold and dist2s < threshold:
                    is_dup = True
                    break
            except Exception:
                continue

        if not is_dup:
            seen_pairs.append((h1, h2))
            unique.append(samples[i])

    removed = len(samples) - len(unique)
    logger.info(f"Perceptual dedup: {removed} removed, {len(unique)} remaining")
    return unique


def semantic_dedup(
    samples: list[dict],
    similarity_threshold: float = 0.92,
    num_clusters: int = 500,
) -> list[dict]:
    """
    SemDeDup: Remove semantically similar samples using CLIP embeddings + k-means.
    Based on Abbas et al., NeurIPS 2023.
    """
    from src.common.clip_utils import encode_texts_batch

    # Build text representation of each sample for semantic comparison
    texts = []
    for s in samples:
        desc1 = s.get("image1_description", s.get("image1_caption", ""))
        desc2 = s.get("image2_description", s.get("image2_caption", ""))
        prompt = s.get("text_prompt", "")
        texts.append(f"{desc1} | {desc2} | {prompt}")

    if len(texts) < num_clusters:
        num_clusters = max(2, len(texts) // 10)

    logger.info(f"Computing semantic embeddings for {len(texts)} samples...")
    embeddings = encode_texts_batch(texts)

    # K-means clustering
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=256)
    labels = kmeans.fit_predict(embeddings)

    # Within each cluster, remove samples too similar to the centroid-closest sample
    unique_indices = set()
    for cluster_id in range(num_clusters):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue
        if len(cluster_indices) == 1:
            unique_indices.add(cluster_indices[0])
            continue

        cluster_embeddings = embeddings[cluster_indices]
        # Compute pairwise similarities within cluster
        sims = cluster_embeddings @ cluster_embeddings.T

        # Greedy selection: keep the first, then only add if dissimilar enough
        selected = [0]
        for i in range(1, len(cluster_indices)):
            max_sim = max(sims[i, j] for j in selected)
            if max_sim < similarity_threshold:
                selected.append(i)

        for idx in selected:
            unique_indices.add(cluster_indices[idx])

    unique = [samples[i] for i in sorted(unique_indices)]
    removed = len(samples) - len(unique)
    logger.info(f"Semantic dedup: {removed} removed, {len(unique)} remaining")
    return unique


def run_full_dedup(
    samples: list[dict],
    use_exact: bool = True,
    use_perceptual: bool = True,
    use_semantic: bool = True,
    semantic_threshold: float = 0.92,
) -> list[dict]:
    """Run the full deduplication pipeline."""
    original_count = len(samples)

    if use_exact:
        samples = exact_dedup(samples)

    if use_perceptual:
        samples = perceptual_dedup(samples)

    if use_semantic and len(samples) > 100:
        samples = semantic_dedup(samples, similarity_threshold=semantic_threshold)

    total_removed = original_count - len(samples)
    logger.info(
        f"Full dedup: {total_removed}/{original_count} removed "
        f"({total_removed/original_count*100:.1f}%), {len(samples)} remaining"
    )
    return samples
