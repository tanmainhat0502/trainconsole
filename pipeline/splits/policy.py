"""Group-aware, leak-free, incremental split policy.

Core invariants
---------------
1. **The unit of randomness is the leakage group**, not the image. All frames of
   a clip/session/person/video land in the same role.
2. **Deterministic** ranking comes from ``sha256(policy|seed|salt|group)``.
   File order, import order and new arrivals cannot reshuffle old assignments.
3. **Val/test are frozen.** Once a group is assigned, later snapshots keep it.
   New groups default to train, which is what "label tới đâu train tới đó"
   requires: the yardstick does not move while the training set grows.
4. **Class coverage is explicit.** After the proportional holdout, rare-class
   groups are moved to val until every class has a minimum number of boxes.
5. **Leakage is checked, not assumed.** Zero group/sample overlap is asserted
   before a snapshot is written.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from ..config import SplitPolicyConfig
from ..io.images import sha256_text
from ..types import Role, Sample, SplitResult


@dataclass
class _Group:
    key: str
    samples: list[Sample]
    source: str
    stratum: str
    classes: Counter[str]
    rank: dict[str, str]

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def class_boxes(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for sample in self.samples:
            counts.update(sample.class_counts)
        return counts


def _role(value: str | Role) -> Role:
    try:
        return value if isinstance(value, Role) else Role(str(value))
    except ValueError as exc:
        raise ValueError(f"unknown split role: {value}") from exc


def _stratum(sample: Sample, fields: list[str]) -> str:
    values = []
    for field in fields:
        if field in {"source", "session", "clip", "group"}:
            values.append(str(getattr(sample, field) or ""))
        else:
            values.append(str((sample.meta or {}).get(field, "")))
    return "|".join(values) or "all"


def _build_groups(samples: list[Sample], cfg: SplitPolicyConfig) -> dict[str, _Group]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if not sample.group:
            raise ValueError(f"sample {sample.key} has an empty leakage group")
        grouped[sample.group].append(sample)

    groups: dict[str, _Group] = {}
    for key, members in grouped.items():
        first = members[0]
        classes: Counter[str] = Counter()
        for sample in members:
            classes.update(sample.class_counts)
        groups[key] = _Group(
            key=key,
            samples=members,
            source=first.source,
            stratum=_stratum(first, cfg.stratify_by),
            classes=classes,
            rank={
                salt: sha256_text(f"{cfg.id}|{cfg.seed}|{salt}|{key}")
                for salt in ("val", "test", "order")
            },
        )
    return groups


def _targets_by_stratum(
    groups: dict[str, _Group], cfg: SplitPolicyConfig, fraction: float
) -> dict[str, int]:
    """Return the *image* target for each stratum.

    Targets are proportional to stratum size, with at least one group's worth of
    images when a stratum can spare a group.
    """

    total = sum(group.size for group in groups.values())
    desired = round(total * fraction)
    if desired <= 0:
        return {}
    by_stratum: dict[str, list[_Group]] = defaultdict(list)
    for group in groups.values():
        by_stratum[group.stratum].append(group)

    eligible = {
        stratum: members
        for stratum, members in by_stratum.items()
        if len(members) >= 2 and members
    }
    weights = {stratum: float(sum(item.size for item in members)) for stratum, members in eligible.items()}
    if not weights:
        return {}
    weight_total = sum(weights.values())
    targets = {
        stratum: max(1, round(desired * weight / weight_total))
        for stratum, weight in weights.items()
    }
    # Never consume every group of a stratum.
    for stratum, members in eligible.items():
        largest_leave_train = max(1, sum(item.size for item in members) - min(item.size for item in members))
        targets[stratum] = min(targets[stratum], largest_leave_train)
    return targets


def _pick_groups(
    groups: dict[str, _Group],
    *,
    target_images: int,
    salt: str,
    taken: set[str],
) -> list[_Group]:
    """Pick a deterministic prefix whose image count is closest to the target."""

    if target_images <= 0:
        return []
    candidates = [group for group in groups.values() if group.key not in taken]
    candidates.sort(key=lambda group: group.rank[salt])
    if len(candidates) <= 1:
        return []
    # Leave at least one group in train.
    candidates = candidates[: max(1, len(candidates) - 1)]
    best_length, best_delta = 1, float("inf")
    cumulative = 0
    for length, group in enumerate(candidates, start=1):
        cumulative += group.size
        delta = abs(cumulative - target_images)
        if delta < best_delta:
            best_length, best_delta = length, delta
    return candidates[:best_length]


def _roles_for_groups(
    groups: dict[str, _Group],
    cfg: SplitPolicyConfig,
    locks: dict[str, Role],
) -> dict[str, Role]:
    roles: dict[str, Role] = {}
    for key, role in locks.items():
        if key in groups and role in {Role.VAL, Role.TEST}:
            roles[key] = role

    taken = set(roles)
    first_snapshot = not any(role in {Role.VAL, Role.TEST} for role in locks.values())

    if first_snapshot:
        # Create the frozen yardstick once. Test first so val is disjoint.
        test_targets = (
            _targets_by_stratum({key: g for key, g in groups.items() if key not in taken}, cfg, cfg.test_fraction)
            if cfg.test_fraction
            else {}
        )
        test_groups: list[_Group] = []
        for stratum in sorted(test_targets):
            members = {
                key: group
                for key, group in groups.items()
                if key not in taken and group.stratum == stratum
            }
            test_groups.extend(
                _pick_groups(members, target_images=test_targets[stratum], salt="test", taken=taken)
            )
        for group in test_groups:
            roles[group.key] = Role.TEST
            taken.add(group.key)

        remaining = {key: group for key, group in groups.items() if key not in taken}
        val_targets = _targets_by_stratum(remaining, cfg, cfg.val_fraction)
        val_groups: list[_Group] = []
        for stratum in sorted(val_targets):
            members = {
                key: group
                for key, group in remaining.items()
                if group.stratum == stratum
            }
            val_groups.extend(
                _pick_groups(members, target_images=val_targets[stratum], salt="val", taken=taken)
            )
        for group in val_groups:
            roles[group.key] = Role.VAL
            taken.add(group.key)

    elif cfg.allow_val_growth:
        # Explicit opt-in: extend val with new groups. Metrics are no longer
        # strictly comparable to the previous snapshot; the run records that.
        available = [group for group in groups.values() if group.key not in taken]
        current_val = sum(group.size for key, group in groups.items() if roles.get(key) == Role.VAL)
        total = sum(group.size for group in groups.values())
        target_val = round(total * cfg.val_fraction)
        missing = max(0, target_val - current_val)
        if missing:
            for group in sorted(available, key=lambda item: item.rank["val"]):
                if missing <= 0:
                    break
                if group.size <= missing or missing >= group.size * 0.5:
                    roles[group.key] = Role.VAL
                    missing -= group.size

    for key in groups:
        roles.setdefault(key, Role.TRAIN)
    return roles


def _add_rare_class_coverage(
    groups: dict[str, _Group],
    roles: dict[str, Role],
    cfg: SplitPolicyConfig,
) -> list[str]:
    """Move the smallest train group containing a missing class into val."""

    warnings: list[str] = []
    total = sum(group.size for group in groups.values())
    max_val = max(1, round(total * cfg.max_val_fraction))
    val_classes: Counter[str] = Counter()
    for key, role in roles.items():
        if role is Role.VAL:
            val_classes.update(groups[key].class_boxes)
    all_classes = {label for group in groups.values() for label in group.classes}
    for label in sorted(all_classes):
        if val_classes[label] >= cfg.min_val_boxes_per_class:
            continue
        candidates = [
            group
            for key, group in groups.items()
            if roles[key] is Role.TRAIN and label in group.classes
        ]
        candidates.sort(key=lambda group: (group.size, group.rank["val"]))
        for group in candidates:
            current_val_size = sum(groups[key].size for key, role in roles.items() if role is Role.VAL)
            if current_val_size + group.size > max_val:
                warnings.append(
                    f"class {label!r} is under-represented in val; adding another group "
                    f"would exceed max_val_fraction={cfg.max_val_fraction}"
                )
                break
            roles[group.key] = Role.VAL
            val_classes.update(group.class_boxes)
            if val_classes[label] >= cfg.min_val_boxes_per_class:
                break
    for label in sorted(all_classes):
        if val_classes[label] == 0:
            warnings.append(f"class {label!r} has no boxes in val after coverage pass")
    return warnings


def _leakage_check(samples: list[Sample], roles: dict[str, Role], groups: dict[str, _Group]) -> dict[str, Any]:
    sample_roles: dict[str, Role] = {}
    group_roles: dict[str, set[Role]] = defaultdict(set)
    for sample in samples:
        role = roles.get(sample.group)
        if role is None:
            raise ValueError(f"group {sample.group!r} has no split role")
        previous = sample_roles.get(sample.key)
        if previous is not None and previous is not role:
            raise ValueError(f"sample {sample.key} appears in both {previous} and {role}")
        sample_roles[sample.key] = role
        group_roles[sample.group].add(role)
    mixed = {key: sorted(role.value for role in values) for key, values in group_roles.items() if len(values) > 1}
    if mixed:
        raise ValueError(f"leakage: groups span several roles: {mixed}")

    counts = Counter(role.value for role in sample_roles.values())
    group_counts = Counter()
    for key, role in roles.items():
        group_counts[role.value] += 1
    train_keys = {key for key, role in sample_roles.items() if role is Role.TRAIN}
    val_keys = {key for key, role in sample_roles.items() if role is Role.VAL}
    test_keys = {key for key, role in sample_roles.items() if role is Role.TEST}
    return {
        "group_overlap": 0,
        "sample_overlap": 0,
        "train_val_sample_overlap": len(train_keys & val_keys),
        "train_test_sample_overlap": len(train_keys & test_keys),
        "val_test_sample_overlap": len(val_keys & test_keys),
        "counts": dict(sorted(counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "total_samples": len(sample_roles),
        "total_groups": len(groups),
    }


def _distribution(samples: list[Sample], roles: dict[str, Role]) -> dict[str, Any]:
    per_role: dict[str, Any] = {}
    for role in (Role.TRAIN, Role.VAL, Role.TEST):
        members = [sample for sample in samples if roles[sample.group] is role]
        if not members:
            continue
        sources = Counter(sample.source for sample in members)
        class_images: Counter[str] = Counter()
        class_boxes: Counter[str] = Counter()
        for sample in members:
            counts = sample.class_counts
            class_boxes.update(counts)
            class_images.update(set(counts))
        per_role[role.value] = {
            "images": len(members),
            "boxes": int(sum(boxes for sample in members for boxes in [len(sample.boxes)])),
            "groups": len({sample.group for sample in members}),
            "sources": dict(sorted(sources.items())),
            "class_images": dict(sorted(class_images.items())),
            "class_boxes": dict(sorted(class_boxes.items())),
        }
    return per_role


def assign_splits(
    samples: list[Sample],
    cfg: SplitPolicyConfig,
    *,
    locks: dict[str, Role] | None = None,
) -> SplitResult:
    if not 0.0 <= cfg.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 <= cfg.test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if cfg.val_fraction + cfg.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")
    if not samples:
        raise ValueError("cannot split an empty dataset")

    groups = _build_groups(samples, cfg)
    normalized_locks = {key: _role(value) for key, value in (locks or {}).items()}
    roles = _roles_for_groups(groups, cfg, normalized_locks)
    warnings = (
        _add_rare_class_coverage(groups, roles, cfg)
        if cfg.val_fraction > 0 and cfg.min_val_boxes_per_class > 0
        else []
    )
    checks = _leakage_check(samples, roles, groups)
    distribution = _distribution(samples, roles)
    total = len(samples)
    val_images = checks["counts"].get(Role.VAL.value, 0)
    test_images = checks["counts"].get(Role.TEST.value, 0)

    if val_images < 2:
        warnings.append("validation has fewer than two images; training metrics will be unstable")
    if test_images and val_images == 0:
        warnings.append("a test split exists but validation is empty")

    report = {
        "policy": {
            "id": cfg.id,
            "seed": cfg.seed,
            "val_fraction": cfg.val_fraction,
            "test_fraction": cfg.test_fraction,
            "stratify_by": cfg.stratify_by,
            "min_val_boxes_per_class": cfg.min_val_boxes_per_class,
            "max_val_fraction": cfg.max_val_fraction,
            "grouping_version": cfg.grouping_version,
        },
        "totals": {"images": total, "groups": len(groups), "boxes": sum(len(s.boxes) for s in samples)},
        "actual": {
            "val_fraction": round(val_images / total, 6) if total else 0.0,
            "test_fraction": round(test_images / total, 6) if total else 0.0,
        },
        "by_role": distribution,
        "checks": checks,
        "warnings": warnings,
    }

    sample_roles = {sample.key: roles[sample.group] for sample in samples}
    # Preserve locks for groups that are not present in this snapshot (for
    # example, a frozen val group whose images have not been labeled yet).
    locks_out = dict(normalized_locks)
    if cfg.lock:
        for key, role in roles.items():
            if role in {Role.VAL, Role.TEST}:
                locks_out[key] = role
    else:
        locks_out = {}
    return SplitResult(roles=sample_roles, groups=roles, locks=locks_out, report=report)


def load_locks(path: str | Path, *, policy_id: str | None = None) -> dict[str, Role]:
    file = Path(path)
    if not file.exists():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    stored_policy = payload.get("policy_id")
    if policy_id and stored_policy and stored_policy != policy_id:
        raise ValueError(
            f"split lock file {file} belongs to policy {stored_policy!r}, "
            f"not {policy_id!r}; create a new lock file to start a new baseline"
        )
    raw = payload.get("locks", payload)
    return {str(key): _role(value) for key, value in raw.items()}


def save_locks(path: str | Path, locks: dict[str, Role], *, policy_id: str) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_id": policy_id,
        "locks": {key: role.value for key, role in sorted(locks.items())},
    }
    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_report(report: dict[str, Any]) -> str:
    totals = report["totals"]
    checks = report["checks"]
    lines = [
        "# Dataset split report",
        "",
        f"Policy: `{report['policy']['id']}` · seed `{report['policy']['seed']}`",
        "",
        "## Summary",
        "",
        "| Role | Images | Groups | Boxes | Share |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, values in report["by_role"].items():
        share = values["images"] / max(1, totals["images"]) * 100
        lines.append(f"| {role} | {values['images']} | {values['groups']} | {values['boxes']} | {share:.1f}% |")
    lines += [
        "",
        f"Total: **{totals['images']}** images, **{totals['groups']}** leakage groups, "
        f"**{totals['boxes']}** boxes.",
        "",
        "## Leakage checks",
        "",
        f"- group overlap: **{checks['group_overlap']}**",
        f"- train/val sample overlap: **{checks['train_val_sample_overlap']}**",
        f"- train/test sample overlap: **{checks['train_test_sample_overlap']}**",
        f"- val/test sample overlap: **{checks['val_test_sample_overlap']}**",
        "",
        "## Class distribution (boxes)",
        "",
    ]
    classes = sorted(
        {label for values in report["by_role"].values() for label in values["class_boxes"]}
    )
    header = "| Class | " + " | ".join(report["by_role"]) + " |"
    lines += [header, "|---|" + "---:|" * len(report["by_role"])]
    for label in classes:
        cells = [str(report["by_role"][role]["class_boxes"].get(label, 0)) for role in report["by_role"]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if report["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in report["warnings"]]
    lines += [
        "",
        "> Validation/test groups are frozen by `group_locks.json`. New groups arrive as train, "
        "so incremental training does not move the evaluation yardstick.",
        "",
    ]
    return "\n".join(lines)
