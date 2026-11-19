"""Representative batch selection for labeling.

Requirements this implements:

* a batch is drawn from the **current pool**, not from consecutive frames;
* sources and visual clusters are represented proportionally but with a
  square-root dampening so one huge source cannot dominate;
* at most ``per_group_cap`` images per leakage group (session/clip/video/person);
* a minimum frame gap inside the same group;
* near-duplicate rejection by embedding cosine;
* a random audit arm for unbiased quality estimates;
* an optional hard-example arm driven by model uncertainty;
* deterministic output for a given seed and candidate set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import numpy as np

from ..config import SamplingConfig
from ..types import Sample, Selection


@dataclass
class _State:
    cfg: SamplingConfig
    selected: list[int]
    selected_set: set[int]
    group_counts: Counter[str]
    group_frames: dict[str, list[int]]
    vectors: np.ndarray


def _can_take(sample: Sample, index: int, state: _State, cfg: SamplingConfig) -> bool:
    if index in state.selected_set:
        return False
    if state.group_counts[sample.group] >= cfg.per_group_cap:
        return False
    if sample.frame_index is not None:
        frames = state.group_frames.get(sample.group, [])
        if frames and min(abs(sample.frame_index - frame) for frame in frames) < cfg.min_frame_gap:
            return False
    if cfg.near_duplicate_cosine > 0 and state.selected:
        similarity = float(np.max(state.vectors[state.selected] @ state.vectors[index]))
        if similarity >= cfg.near_duplicate_cosine:
            return False
    return True


def _take(sample: Sample, index: int, state: _State, selection: list[Selection], *, score: float, reason: str, stratum: str, cluster: int | None = None) -> bool:
    if not _can_take(sample, index, state, state.cfg):
        return False
    state.selected.append(index)
    state.selected_set.add(index)
    state.group_counts[sample.group] += 1
    if sample.frame_index is not None:
        state.group_frames.setdefault(sample.group, []).append(sample.frame_index)
    selection.append(
        Selection(
            sample_key=sample.key,
            score=round(float(score), 6),
            reason=reason,
            stratum=stratum,
            cluster=cluster,
        )
    )
    return True


def _largest_remainder(weights: dict[str, float], budget: int) -> dict[str, int]:
    total = sum(weights.values())
    if total <= 0 or budget <= 0:
        return {key: 0 for key in weights}
    raw = {key: budget * weight / total for key, weight in weights.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = budget - sum(quotas.values())
    order = sorted(weights, key=lambda key: (raw[key] - quotas[key], key), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _allocate(
    by_source: dict[str, list[int]],
    groups_by_source: dict[str, int],
    cfg: SamplingConfig,
    budget: int,
) -> dict[str, int]:
    """Source quotas with capacity-aware reallocation.

    Capacity is ``min(source size, #groups * per_group_cap)``. Without this,
    a source with only a few sessions would receive a quota it can never fill
    and the batch would silently shrink.
    """

    capacities: dict[str, int] = {}
    for source, indexes in by_source.items():
        capacities[source] = min(len(indexes), groups_by_source.get(source, 0) * cfg.per_group_cap)

    weights = {source: float(len(indexes)) ** cfg.source_power for source, indexes in by_source.items()}
    quotas = _largest_remainder(weights, min(budget, sum(capacities.values())))
    # Enforce one per non-empty source when the budget allows it.
    if budget >= len(by_source):
        for source in quotas:
            quotas[source] = max(quotas[source], 1)
    # Reallocate overflow / shortage until stable.
    for _ in range(12):
        overflow = sum(max(0, quotas[s] - capacities[s]) for s in quotas)
        if overflow == 0:
            # Give remaining budget to sources with spare capacity.
            spare = {s: capacities[s] - quotas[s] for s in quotas if capacities[s] > quotas[s]}
            remaining = budget - sum(quotas.values())
            if remaining <= 0 or not spare:
                break
            extra = _largest_remainder({s: float(spare[s]) for s in spare}, remaining)
            for source, value in extra.items():
                quotas[source] = min(capacities[source], quotas[source] + value)
            continue
        for source in quotas:
            quotas[source] = min(quotas[source], capacities[source])
        room = {s: capacities[s] - quotas[s] for s in quotas if capacities[s] > quotas[s]}
        if not room:
            break
        extra = _largest_remainder({s: float(room[s]) for s in room}, overflow)
        for source, value in extra.items():
            quotas[source] = min(capacities[source], quotas[source] + value)
    return quotas


def _kcenter_order(vectors: np.ndarray) -> list[int]:
    """Farthest-point traversal seeded at the medoid, O(N*K*D) but fine per stratum."""

    if len(vectors) == 0:
        return []
    centroid = vectors.mean(axis=0)
    medoid = int(np.argmin(np.linalg.norm(vectors - centroid, axis=1)))
    order = [medoid]
    chosen = np.zeros(len(vectors), dtype=bool)
    chosen[medoid] = True
    min_distance = np.linalg.norm(vectors - vectors[medoid], axis=1)
    while not chosen.all():
        candidate = int(np.argmax(np.where(chosen, -1.0, min_distance)))
        order.append(candidate)
        chosen[candidate] = True
        min_distance = np.minimum(min_distance, np.linalg.norm(vectors - vectors[candidate], axis=1))
    return order


def _cluster_medoids(vectors: np.ndarray, quota: int, *, seed: int) -> list[tuple[int, int]]:
    """Return ``(row_index, cluster_id)`` medoids, one per cluster when possible."""

    if quota <= 0 or len(vectors) == 0:
        return []
    if len(vectors) <= quota:
        return [(index, index) for index in range(len(vectors))]

    try:
        from sklearn.cluster import KMeans

        clusters = min(quota, len(vectors))
        model = KMeans(n_clusters=clusters, n_init=1, random_state=seed)
        labels = model.fit_predict(vectors)
        centers = model.cluster_centers_
        output: list[tuple[int, int]] = []
        for cluster_id in range(clusters):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                continue
            center = centers[cluster_id]
            distance = np.linalg.norm(vectors[members] - center, axis=1)
            for position in np.argsort(distance):
                output.append((int(members[position]), cluster_id))
                break
        return output
    except Exception:
        # Fallback: farthest-point traversal (still representative, slightly more
        # coverage-oriented than medoid-oriented).
        return [(int(index), 0) for index in _kcenter_order(vectors)]


def select_batch(
    samples: list[Sample],
    embeddings: np.ndarray,
    budget: int,
    config: SamplingConfig,
    *,
    seed: int = 0,
    excluded_keys: set[str] | None = None,
    uncertainty: dict[str, float] | None = None,
    strata_key=lambda sample: sample.source,
) -> tuple[list[Selection], dict]:
    """Select a batch. Returns ``(selections, report)``."""

    if embeddings.shape[0] != len(samples):
        raise ValueError("embeddings must be aligned with samples")
    if budget <= 0 or not samples:
        return [], {"budget": budget, "selected": 0, "reason": "empty pool"}

    excluded = set(excluded_keys or ())
    candidates = [i for i, sample in enumerate(samples) if sample.key not in excluded]
    if not candidates:
        return [], {"budget": budget, "selected": 0, "reason": "no candidates"}

    vectors = np.asarray(embeddings, dtype=np.float32)
    # Normalize defensively: adapters may return unnormalized vectors.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(vectors, np.maximum(norms, 1e-12))
    by_source: dict[str, list[int]] = defaultdict(list)
    for index in candidates:
        by_source[strata_key(samples[index])].append(index)
    groups_by_source: dict[str, int] = {
        source: len({samples[index].group for index in indexes})
        for source, indexes in by_source.items()
    }

    quotas = _allocate(by_source, groups_by_source, config, min(budget, len(candidates)))
    state = _State(
        cfg=config,
        selected=[],
        selected_set=set(),
        group_counts=Counter(),
        group_frames={},
        vectors=vectors,
    )
    selections: list[Selection] = []
    rng = np.random.default_rng(seed)
    rejection_notes = Counter()

    def select_from_indices(
        indexes: list[int],
        count: int,
        *,
        reason: str,
        stratum: str,
        order: list[int] | None = None,
        scores: dict[int, float] | None = None,
        cluster_map: dict[int, int] | None = None,
    ) -> None:
        if count <= 0:
            return
        taken = 0
        sequence = order if order is not None else list(indexes)
        for index in sequence:
            if taken >= count:
                break
            if index not in indexes:
                continue
            sample = samples[index]
            if _take(
                sample,
                index,
                state,
                selections,
                score=(scores or {}).get(index, 0.0),
                reason=reason,
                stratum=stratum,
                cluster=(cluster_map or {}).get(index),
            ):
                taken += 1
            else:
                rejection_notes[reason] += 1

    for source in sorted(by_source):
        quota = quotas.get(source, 0)
        if quota <= 0:
            continue
        indexes = by_source[source]
        if len(indexes) <= quota:
            select_from_indices(indexes, quota, reason="all-candidates", stratum=source)
            continue

        audit_count = max(1, round(quota * config.audit_fraction)) if quota >= 5 else 0
        hard_count = round(quota * config.hard_fraction) if uncertainty and config.hard_fraction > 0 else 0
        diverse_count = max(0, quota - audit_count - hard_count)

        # 1. Random audit arm.
        audit_order = rng.permutation(indexes).tolist()
        select_from_indices(audit_order, audit_count, reason="random-audit", stratum=source)

        # 2. Hard-example arm (highest uncertainty first).
        if hard_count:
            hard_order = sorted(
                indexes,
                key=lambda index: (uncertainty or {}).get(samples[index].key, 0.0),
                reverse=True,
            )
            select_from_indices(
                hard_order,
                hard_count,
                reason="hard-example",
                stratum=source,
                scores={index: (uncertainty or {}).get(samples[index].key, 0.0) for index in indexes},
            )

        # 3. Cluster medoids for representative diversity.
        available = [index for index in indexes if index not in state.selected_set]
        if available and diverse_count > 0:
            local_vectors = vectors[available]
            medoids = _cluster_medoids(local_vectors, diverse_count, seed=seed + len(source))
            # Convert local rows back to global indices, grouped by cluster and
            # ordered by distance-to-centroid as returned.
            medoid_order = [available[local] for local, _ in medoids]
            select_from_indices(
                available,
                diverse_count,
                reason="cluster-medoid",
                stratum=source,
                order=medoid_order,
                cluster_map={available[local]: cluster for local, cluster in medoids},
            )

        # 4. Fill anything left by farthest-point traversal.
        remaining = quota - sum(1 for item in selections if item.stratum == source)
        if remaining > 0:
            available = np.asarray([index for index in indexes if index not in state.selected_set], dtype=int)
            if len(available):
                if state.selected:
                    min_distance = np.min(1.0 - vectors[available] @ vectors[state.selected].T, axis=1)
                else:
                    centroid = vectors[available].mean(axis=0)
                    min_distance = 1.0 - vectors[available] @ centroid
                fill_order = available[np.argsort(-min_distance)].tolist()
                select_from_indices(fill_order, remaining, reason="diversity-fill", stratum=source)

    # 5. Class coverage top-up: do not return a batch that never shows a rare class.
    key_to_sample = {sample.key: sample for sample in samples}
    class_counts: Counter[str] = Counter()
    for item in selections:
        class_counts.update(key_to_sample[item.sample_key].class_counts)
    missing = {
        label: config.min_boxes_per_class - class_counts[label]
        for label in {box.label for sample in samples for box in sample.boxes}
        if class_counts[label] < config.min_boxes_per_class
    }
    if missing and len(selections) < budget:
        for label, needed in sorted(missing.items(), key=lambda item: item[1], reverse=True):
            if needed <= 0:
                continue
            candidates_with_label = [
                index
                for index in candidates
                if index not in state.selected_set and label in samples[index].classes
            ]
            candidates_with_label.sort(
                key=lambda index: (-samples[index].class_counts[label], samples[index].group)
            )
            for index in candidates_with_label:
                if needed <= 0 or len(selections) >= budget:
                    break
                if _take(
                    samples[index],
                    index,
                    state,
                    selections,
                    score=float(samples[index].class_counts[label]),
                    reason="class-coverage",
                    stratum=strata_key(samples[index]),
                ):
                    needed -= 1

    # Audit trailing budget: if constraints blocked some quota, fill from any
    # remaining candidate, still respecting caps/gap/dedup.
    if len(selections) < budget:
        remaining_order = rng.permutation([index for index in candidates if index not in state.selected_set])
        for index in remaining_order:
            if len(selections) >= budget:
                break
            _take(
                samples[index],
                index,
                state,
                selections,
                score=0.0,
                reason="budget-fill",
                stratum=strata_key(samples[index]),
            )

    selected_samples = [samples[state.selected[i]] for i in range(len(state.selected))]
    report = {
        "budget": budget,
        "selected": len(selections),
        "config": {
            "audit_fraction": config.audit_fraction,
            "per_group_cap": config.per_group_cap,
            "min_frame_gap": config.min_frame_gap,
            "near_duplicate_cosine": config.near_duplicate_cosine,
            "source_power": config.source_power,
            "hard_fraction": config.hard_fraction,
        },
        "quotas": quotas,
        "per_source": dict(sorted(Counter(item.stratum for item in selections).items())),
        "per_reason": dict(sorted(Counter(item.reason for item in selections).items())),
        "per_class_images": {
            label: sum(1 for sample in selected_samples if label in sample.classes)
            for label in sorted({box.label for sample in samples for box in sample.boxes})
        },
        "per_class_boxes": dict(sorted(sum((s.class_counts for s in selected_samples), Counter()).items())),
        "groups_used": len({sample.group for sample in selected_samples}),
        "max_per_group": max((Counter(s.group for s in selected_samples).values()), default=0),
        "rejections": dict(rejection_notes),
        "seed": seed,
    }
    return selections, report
