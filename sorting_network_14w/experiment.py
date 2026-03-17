"""Symmetry-aware 14-wire sorting-network search experiment.

This stays separate from Karpathy's original Python files and borrows only the
general "autonomous experiment loop" idea. The search strategy is built around:

1. analyzing the current 16-wire JSON candidate,
2. projecting a correct 14-wire seed from that family,
3. compressing to shorter fixed lengths with a deletion beam,
4. repairing low-failure candidates with hotspot-guided local search.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Comparator = Tuple[int, int]
Network = List[Comparator]

TARGET_WIRES = 14
REFERENCE_WIRES = 16
TARGET_LT_COMPARATORS = 51
TARGET_LEN = TARGET_LT_COMPARATORS - 1
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
REFERENCE_JSON = REPO_ROOT / "minimal_sorting_networks" / "ga_best_candidate.json"
REFERENCE_GIT_PATH = "minimal_sorting_networks/ga_best_candidate.json"
EXPERIMENT_GIT_PATH = "sorting_network_14w/experiment.py"
DEFAULT_ANALYSIS_FILE = EXPERIMENT_DIR / "analysis.json"
DEFAULT_STATE_FILE = EXPERIMENT_DIR / "state.json"
DEFAULT_RESULTS_FILE = EXPERIMENT_DIR / "results.tsv"
RESULTS_HEADER = "commit\ttarget_len\tfail_count\tdepth\tstatus\tdescription\n"
ALL_COMPARATORS: Tuple[Comparator, ...] = tuple(
    (i, j) for i in range(TARGET_WIRES) for j in range(i + 1, TARGET_WIRES)
)


def build_wire_masks(num_wires: int) -> Tuple[int, ...]:
    total = 1 << num_wires
    wires = [0] * num_wires
    for case in range(total):
        for wire in range(num_wires):
            if (case >> (num_wires - 1 - wire)) & 1:
                wires[wire] |= 1 << case
    return tuple(wires)


BASE_WIRES_14 = build_wire_masks(TARGET_WIRES)
BASE_WIRES_16 = build_wire_masks(REFERENCE_WIRES)


def normalize_comparator(raw: Sequence[int], num_wires: int) -> Comparator | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        i = int(raw[0])
        j = int(raw[1])
    except (TypeError, ValueError):
        return None
    if i == j:
        return None
    if not (0 <= i < num_wires and 0 <= j < num_wires):
        return None
    if j < i:
        i, j = j, i
    return (i, j)


def sanitize_network(raw: object, num_wires: int) -> Network:
    if isinstance(raw, dict):
        for key in ("network", "comparators", "pairs"):
            if key in raw:
                raw = raw[key]
                break
    if not isinstance(raw, (list, tuple)):
        return []
    out: Network = []
    for item in raw:
        comp = normalize_comparator(item, num_wires)
        if comp is not None:
            out.append(comp)
    return out


def pack_layers(network: Sequence[Comparator], num_wires: int) -> List[List[Comparator]]:
    layers: List[List[Comparator]] = []
    last_stage = [-1] * num_wires
    for i, j in network:
        stage = max(last_stage[i], last_stage[j]) + 1
        while len(layers) <= stage:
            layers.append([])
        layers[stage].append((i, j))
        last_stage[i] = stage
        last_stage[j] = stage
    return layers


def flatten_layers(layers: Iterable[Iterable[Comparator]]) -> Network:
    out: Network = []
    for layer in layers:
        out.extend(layer)
    return out


def depth_of(network: Sequence[Comparator], num_wires: int) -> int:
    return len(pack_layers(network, num_wires))


def mirror_network(network: Sequence[Comparator], num_wires: int) -> Network:
    return [(num_wires - 1 - j, num_wires - 1 - i) for i, j in network]


def canonicalize_network(network: Sequence[Comparator], num_wires: int) -> Network:
    sanitized = [comp for comp in (normalize_comparator(c, num_wires) for c in network) if comp]
    packed = flatten_layers(pack_layers(sanitized, num_wires))
    mirrored = flatten_layers(pack_layers(mirror_network(packed, num_wires), num_wires))
    return mirrored if tuple(mirrored) < tuple(packed) else packed


def evaluate_network(
    network: Sequence[Comparator],
    base_wires: Tuple[int, ...],
) -> Tuple[int, int]:
    wires = list(base_wires)
    for i, j in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    invalid = 0
    for i in range(len(wires) - 1):
        invalid |= wires[i] & ~wires[i + 1]
    return invalid.bit_count(), invalid


def network_key(network: Sequence[Comparator]) -> Tuple[Comparator, ...]:
    return tuple(network)


def bitstring(case: int, num_wires: int) -> str:
    return "".join("1" if (case >> (num_wires - 1 - i)) & 1 else "0" for i in range(num_wires))


def simulate_case(case: int, network: Sequence[Comparator], num_wires: int) -> List[int]:
    wires = [(case >> (num_wires - 1 - idx)) & 1 for idx in range(num_wires)]
    for i, j in network:
        if wires[i] > wires[j]:
            wires[i], wires[j] = wires[j], wires[i]
    return wires


def failure_signature(
    network: Sequence[Comparator],
    num_wires: int,
    invalid_mask: int,
    limit_cases: int = 12,
) -> Dict[str, object]:
    hotspot_counter: Counter[int] = Counter()
    samples = []
    total_cases = 1 << num_wires
    for case in range(total_cases):
        if not ((invalid_mask >> case) & 1):
            continue
        output = simulate_case(case, network, num_wires)
        inversions = [idx for idx in range(num_wires - 1) if output[idx] > output[idx + 1]]
        for idx in inversions:
            hotspot_counter[idx] += 1
        if len(samples) < limit_cases:
            samples.append(
                {
                    "case_index": case,
                    "input": bitstring(case, num_wires),
                    "output": "".join(str(bit) for bit in output),
                    "inversion_positions": inversions,
                }
            )
    return {
        "hotspot_inversions": dict(sorted(hotspot_counter.items())),
        "sample_cases": samples,
    }


def batcher16() -> Network:
    out: Network = []

    def odd_even_merge(lo: int, n: int, r: int) -> None:
        step = r * 2
        if step < n:
            odd_even_merge(lo, n, step)
            odd_even_merge(lo + r, n, step)
            for idx in range(lo + r, lo + n - r, step):
                out.append((idx, idx + r))
        else:
            out.append((lo, lo + r))

    def odd_even_merge_sort(lo: int, n: int) -> None:
        if n > 1:
            m = n // 2
            odd_even_merge_sort(lo, m)
            odd_even_merge_sort(lo + m, m)
            odd_even_merge(lo, n, 1)

    odd_even_merge_sort(0, REFERENCE_WIRES)
    return out


def project_network(network: Sequence[Comparator], removed_wires: Sequence[int], num_wires: int) -> Network:
    removed = set(removed_wires)
    kept = [wire for wire in range(num_wires) if wire not in removed]
    mapping = {old: new for new, old in enumerate(kept)}
    out: Network = []
    for i, j in network:
        if i in removed or j in removed:
            continue
        out.append((mapping[i], mapping[j]))
    return out


def multiset_diff(a: Sequence[Comparator], b: Sequence[Comparator]) -> Dict[str, int]:
    diff = Counter(a) - Counter(b)
    return {f"{i}-{j}": count for (i, j), count in sorted(diff.items())}


def analyze_reference_candidate() -> Dict[str, object]:
    payload = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    network = sanitize_network(payload, REFERENCE_WIRES)
    network = canonicalize_network(network, REFERENCE_WIRES)
    fail_count, invalid_mask = evaluate_network(network, BASE_WIRES_16)
    batcher = batcher16()
    projected_seed = canonicalize_network(project_network(batcher, removed_wires=(0, 1), num_wires=16), TARGET_WIRES)
    projected_fail, _ = evaluate_network(projected_seed, BASE_WIRES_14)

    return {
        "source_candidate": REFERENCE_GIT_PATH,
        "source_problem": {"num_wires": REFERENCE_WIRES, "goal_comparators_lt": 63},
        "target_problem": {"num_wires": TARGET_WIRES, "goal_comparators_lt": TARGET_LT_COMPARATORS},
        "current_best_summary": {
            "comparator_count": len(network),
            "depth": depth_of(network, REFERENCE_WIRES),
            "fail_count": fail_count,
            "distance_histogram": {
                str(distance): count
                for distance, count in sorted(Counter(j - i for i, j in network).items())
            },
            "mirror_symmetric": network == canonicalize_network(mirror_network(network, REFERENCE_WIRES), REFERENCE_WIRES),
            "batcher_multiset_missing": multiset_diff(batcher, network),
            "batcher_multiset_extra": multiset_diff(network, batcher),
        },
        "layers": [
            [[i, j] for i, j in layer]
            for layer in pack_layers(network, REFERENCE_WIRES)
        ],
        "failure_signature": failure_signature(network, REFERENCE_WIRES, invalid_mask),
        "bottleneck_symmetries": [
            {
                "name": "pure power-of-two merge spine",
                "evidence": "all comparator spans are 1, 2, 4, or 8",
                "implication": "the search stayed inside a Batcher/Bose-Nelson family instead of introducing off-template bridges",
            },
            {
                "name": "missing boundary cleanup comparator",
                "evidence": "the 62-comparator candidate is exactly Batcher-16 minus one (1,2) comparator",
                "implication": "all 64 failures collapse to a single unresolved inversion boundary early in the left half",
            },
            {
                "name": "mirror locking",
                "evidence": "the network is fully mirror-symmetric after stage packing",
                "implication": "symmetry reduces search space, but it also blocks asymmetric repairs that often matter on 14 wires",
            },
        ],
        "projected_14w_seed": {
            "removed_wires_from_16w_batcher": [0, 1],
            "comparator_count": len(projected_seed),
            "depth": depth_of(projected_seed, TARGET_WIRES),
            "fail_count": projected_fail,
            "network": [[i, j] for i, j in projected_seed],
        },
        "proposed_modification": {
            "new_file": EXPERIMENT_GIT_PATH,
            "strategy": [
                "start from the correct 53-comparator 14-wire projection instead of mutating an arbitrary flat network",
                "compress with a deletion beam to target lengths 52, 51, and 50",
                "repair low-failure candidates with hotspot-guided replacements, moves, and delete-plus-insert mutations",
                "canonicalize under mirror symmetry, but do not require symmetric mutations so asymmetric escapes stay possible",
            ],
        },
    }


def load_state(state_file: Path) -> Dict[str, object]:
    if not state_file.exists():
        return {
            "num_wires": TARGET_WIRES,
            "goal_comparators_lt": TARGET_LT_COMPARATORS,
            "best_by_length": {},
        }
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {
            "num_wires": TARGET_WIRES,
            "goal_comparators_lt": TARGET_LT_COMPARATORS,
            "best_by_length": {},
        }


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_state(
    state_file: Path,
    target_len: int,
    network: Sequence[Comparator],
    fail_count: int,
    depth: int,
    metadata: Dict[str, object],
) -> None:
    state = load_state(state_file)
    best_by_length = state.setdefault("best_by_length", {})
    key = str(target_len)
    current = best_by_length.get(key)
    improved = current is None or (
        int(fail_count),
        int(depth),
    ) < (
        int(current.get("fail_count", 1 << 60)),
        int(current.get("depth", 1 << 60)),
    )
    if not improved:
        return
    best_by_length[key] = {
        "target_length": int(target_len),
        "fail_count": int(fail_count),
        "comparator_count": int(len(network)),
        "depth": int(depth),
        "updated_unix_ts": time.time(),
        "metadata": metadata,
        "network": [[i, j] for i, j in network],
    }
    write_json_atomic(state_file, state)


def update_run_state(state_file: Path, run_payload: Dict[str, object]) -> None:
    state = load_state(state_file)
    state["last_run"] = run_payload
    write_json_atomic(state_file, state)


def resolve_seed(cli_seed: int | None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    dt = datetime.now().isoformat(timespec="microseconds")
    entropy = f"{dt}|{time.time_ns()}|{os.getpid()}|{os.urandom(8).hex()}".encode("utf-8")
    digest = hashlib.blake2b(entropy, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def evaluate_cached(
    network: Sequence[Comparator],
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    key = tuple(network)
    cached = cache.get(key)
    if cached is not None:
        return cached
    fail_count, _ = evaluate_network(network, BASE_WIRES_14)
    depth = depth_of(network, TARGET_WIRES)
    cached = (fail_count, len(network), depth)
    cache[key] = cached
    return cached


def score_for_target(
    network: Sequence[Comparator],
    target_len: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    fail_count, _, depth = evaluate_cached(network, cache)
    length_penalty = abs(len(network) - target_len)
    return (fail_count, length_penalty, depth)


def descendant_aware_score(
    network: Sequence[Comparator],
    target_len: int,
    goal_length: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
    lookahead_cache: Dict[Tuple[Tuple[Comparator, ...], int, int], Tuple[int, ...]],
) -> Tuple[int, ...]:
    fail_count, _, depth = evaluate_cached(network, cache)
    length_penalty = abs(len(network) - target_len)
    if length_penalty != 0 or target_len <= goal_length or target_len != goal_length + 1:
        return (fail_count, length_penalty, depth)

    key = (tuple(network), target_len, goal_length)
    cached = lookahead_cache.get(key)
    if cached is not None:
        return cached

    child_target = target_len - 1
    children: Dict[Tuple[Comparator, ...], Network] = {}
    for idx in range(len(network)):
        child = canonicalize_network(network[:idx] + network[idx + 1 :], TARGET_WIRES)
        children.setdefault(tuple(child), child)

    best_child_score = min(
        descendant_aware_score(child, child_target, goal_length, cache, lookahead_cache)
        for child in children.values()
    )
    result = best_child_score + (fail_count, depth)
    lookahead_cache[key] = result
    return result


def seed_priority_for_target(
    network: Sequence[Comparator],
    target_len: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
) -> Tuple[int, int, int, int]:
    fail_count, _, depth = evaluate_cached(network, cache)
    length_gap = abs(len(network) - target_len)
    return (length_gap, fail_count, depth, len(network))


def dedupe_networks(networks: Iterable[Sequence[Comparator]]) -> List[Network]:
    unique: Dict[Tuple[Comparator, ...], Network] = {}
    for network in networks:
        candidate = canonicalize_network(network, TARGET_WIRES)
        if candidate:
            unique.setdefault(tuple(candidate), candidate)
    return list(unique.values())


def load_state_frontier(state_file: Path, goal_length: int) -> List[Network]:
    state = load_state(state_file)
    best_by_length = state.get("best_by_length", {})
    if not isinstance(best_by_length, dict):
        return []
    frontier: List[Network] = []
    for raw_len, payload in sorted(best_by_length.items(), key=lambda item: int(item[0]), reverse=True):
        try:
            target_len = int(raw_len)
        except (TypeError, ValueError):
            continue
        if target_len < goal_length:
            continue
        network = canonicalize_network(sanitize_network(payload, TARGET_WIRES), TARGET_WIRES)
        if network:
            frontier.append(network)
    return dedupe_networks(frontier)


def select_seed_frontier(
    networks: Iterable[Sequence[Comparator]],
    target_len: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
    limit: int,
) -> List[Network]:
    unique = dedupe_networks(networks)
    ranked = sorted(
        unique,
        key=lambda net: seed_priority_for_target(net, target_len, cache),
    )
    return ranked[:limit]


def summarize_best_scores(state: Dict[str, object]) -> Dict[str, Tuple[int, int]]:
    best_by_length = state.get("best_by_length", {})
    if not isinstance(best_by_length, dict):
        return {}
    out: Dict[str, Tuple[int, int]] = {}
    for raw_len, payload in best_by_length.items():
        if not isinstance(payload, dict):
            continue
        try:
            out[str(int(raw_len))] = (
                int(payload.get("fail_count", 1 << 60)),
                int(payload.get("depth", 1 << 60)),
            )
        except (TypeError, ValueError):
            continue
    return out


def allocate_target_budgets(
    search_lengths: Sequence[int],
    total_seconds: float,
    goal_length: int,
    previous_scores: Dict[str, Tuple[int, int]],
) -> Dict[int, float]:
    if not search_lengths:
        return {}
    weights: Dict[int, float] = {}
    for target_len in search_lengths:
        gap = target_len - goal_length
        if gap <= 0:
            weight = 6.0
        elif gap == 1:
            weight = 1.5
        else:
            weight = 0.75
        if str(target_len) not in previous_scores:
            weight *= 1.35
        weights[target_len] = weight
    total_weight = sum(weights.values())
    budgets = {
        target_len: max(1.0, total_seconds * (weights[target_len] / max(total_weight, 1e-9)))
        for target_len in search_lengths
    }
    return budgets


def ensure_results_file(results_file: Path) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    if results_file.exists() and results_file.stat().st_size > 0:
        return
    results_file.write_text(RESULTS_HEADER, encoding="utf-8")


def sanitize_tsv_field(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def current_commit_label() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", EXPERIMENT_GIT_PATH],
            capture_output=True,
            check=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "worktree"
    if not commit:
        return "worktree"
    return f"{commit}+dirty" if dirty else commit


def append_results_row(
    results_file: Path,
    commit_label: str,
    target_len: int,
    fail_count: int,
    depth: int,
    status: str,
    description: str,
) -> None:
    ensure_results_file(results_file)
    row = "\t".join(
        [
            sanitize_tsv_field(commit_label),
            str(int(target_len)),
            str(int(fail_count)),
            str(int(depth)),
            sanitize_tsv_field(status),
            sanitize_tsv_field(description),
        ]
    )
    with results_file.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def deletion_beam(
    seed: Sequence[Comparator],
    target_len: int,
    goal_length: int,
    beam_width: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
    lookahead_cache: Dict[Tuple[Tuple[Comparator, ...], int, int], Tuple[int, ...]],
) -> List[Network]:
    beam = [canonicalize_network(seed, TARGET_WIRES)]
    while beam and len(beam[0]) > target_len:
        next_map: Dict[Tuple[Comparator, ...], Network] = {}
        for network in beam:
            for idx in range(len(network)):
                candidate = canonicalize_network(network[:idx] + network[idx + 1 :], TARGET_WIRES)
                next_map.setdefault(tuple(candidate), candidate)
        ranked = sorted(
            next_map.values(),
            key=lambda net: descendant_aware_score(net, target_len, goal_length, cache, lookahead_cache),
        )
        beam = ranked[:beam_width]
    return beam


@functools.lru_cache(maxsize=65536)
def _hotspot_comparators_cached(network: Tuple[Comparator, ...], limit: int) -> Tuple[Comparator, ...]:
    fail_count, invalid_mask = evaluate_network(network, BASE_WIRES_14)
    if fail_count == 0:
        return ALL_COMPARATORS[:limit]
    counter: Counter[Comparator] = Counter()
    seen = 0
    for case in range(1 << TARGET_WIRES):
        if not ((invalid_mask >> case) & 1):
            continue
        output = simulate_case(case, network, TARGET_WIRES)
        inversions = [idx for idx in range(TARGET_WIRES - 1) if output[idx] > output[idx + 1]]
        for inv in inversions:
            for left in range(max(0, inv - 2), inv + 1):
                for right in range(inv + 1, min(TARGET_WIRES, inv + 4)):
                    counter[(left, right)] += 1
        seen += 1
        if seen >= 128:
            break
    ranked: List[Comparator] = [comp for comp, _ in counter.most_common(limit)]
    if len(ranked) < limit:
        for comp in ALL_COMPARATORS:
            if comp not in counter:
                ranked.append(comp)
            if len(ranked) >= limit:
                break
    return tuple(ranked[:limit])


def hotspot_comparators(network: Sequence[Comparator], limit: int = 24) -> List[Comparator]:
    return list(_hotspot_comparators_cached(network_key(network), limit))

@functools.lru_cache(maxsize=65536)
def _misplacement_comparators_cached(
    network: Tuple[Comparator, ...],
    limit_cases: int,
    limit: int,
) -> Tuple[Comparator, ...]:
    fail_count, invalid_mask = evaluate_network(network, BASE_WIRES_14)
    if fail_count == 0:
        return ALL_COMPARATORS[:limit]
    counter: Counter[Comparator] = Counter()
    seen = 0
    for case in range(1 << TARGET_WIRES):
        if not ((invalid_mask >> case) & 1):
            continue
        output = simulate_case(case, network, TARGET_WIRES)
        ones = case.bit_count()
        ideal = [0] * (TARGET_WIRES - ones) + [1] * ones
        wrong_ones = [idx for idx, (got, want) in enumerate(zip(output, ideal)) if got == 1 and want == 0]
        wrong_zeros = [idx for idx, (got, want) in enumerate(zip(output, ideal)) if got == 0 and want == 1]
        for left in wrong_ones:
            for right in wrong_zeros:
                if left < right:
                    counter[(left, right)] += 1
        seen += 1
        if seen >= limit_cases:
            break
    ranked: List[Comparator] = [comp for comp, _ in counter.most_common(limit)]
    if len(ranked) < limit:
        for comp in hotspot_comparators(network, limit=limit):
            if comp not in ranked:
                ranked.append(comp)
            if len(ranked) >= limit:
                break
    return tuple(ranked[:limit])


def misplacement_comparators(
    network: Sequence[Comparator],
    limit_cases: int = 96,
    limit: int = 24,
) -> List[Comparator]:
    return list(_misplacement_comparators_cached(network_key(network), limit_cases, limit))


@functools.lru_cache(maxsize=65536)
def _dominant_hotspot_key_cached(network: Tuple[Comparator, ...], limit_cases: int) -> Tuple[int, ...]:
    fail_count, invalid_mask = evaluate_network(network, BASE_WIRES_14)
    if fail_count == 0:
        return ()
    counter: Counter[int] = Counter()
    seen = 0
    for case in range(1 << TARGET_WIRES):
        if not ((invalid_mask >> case) & 1):
            continue
        output = simulate_case(case, network, TARGET_WIRES)
        for idx in range(TARGET_WIRES - 1):
            if output[idx] > output[idx + 1]:
                counter[idx] += 1
        seen += 1
        if seen >= limit_cases:
            break
    return tuple(idx for idx, _ in counter.most_common(3))


def dominant_hotspot_key(network: Sequence[Comparator], limit_cases: int = 64) -> Tuple[int, ...]:
    return _dominant_hotspot_key_cached(network_key(network), limit_cases)


def comparator_pool(
    network: Sequence[Comparator],
    anchors: Sequence[Sequence[Comparator]],
    rng: random.Random,
    limit: int = 32,
) -> List[Comparator]:
    pool: List[Comparator] = []
    for comp in misplacement_comparators(network, limit=12):
        if comp not in pool:
            pool.append(comp)
    for comp in hotspot_comparators(network, limit=16):
        if comp not in pool:
            pool.append(comp)
    for anchor in anchors:
        for comp in anchor:
            if comp not in pool:
                pool.append(comp)
            if len(pool) >= limit:
                return pool
    short_spans = [comp for comp in ALL_COMPARATORS if (comp[1] - comp[0]) <= 4]
    rng.shuffle(short_spans)
    for comp in short_spans:
        if comp not in pool:
            pool.append(comp)
        if len(pool) >= limit:
            return pool
    return pool[:limit]


def focused_bridge_mutation(
    network: Sequence[Comparator],
    anchors: Sequence[Sequence[Comparator]],
    rng: random.Random,
) -> Network:
    candidate = list(network)
    hotspot_key = dominant_hotspot_key(candidate)
    hotspot_wires = set()
    for pos in hotspot_key:
        hotspot_wires.update((max(0, pos - 1), pos, pos + 1, min(TARGET_WIRES - 1, pos + 2)))
    preferred_positions = [
        idx
        for idx, (left, right) in enumerate(candidate)
        if left in hotspot_wires or right in hotspot_wires
    ]
    if not preferred_positions:
        preferred_positions = list(range(len(candidate)))
    bridge_pool = [
        comp for comp in misplacement_comparators(candidate, limit=16) if (comp[1] - comp[0]) not in (1, 2, 4, 8)
    ]
    general_pool = comparator_pool(candidate, anchors, rng, limit=28)
    if not bridge_pool:
        bridge_pool = general_pool[:]

    edits = 2 + int(rng.random() < 0.7) + int(rng.random() < 0.35)
    for _ in range(edits):
        if rng.random() < 0.7:
            idx = preferred_positions[rng.randrange(len(preferred_positions))]
        else:
            idx = rng.randrange(len(candidate))
        replacement_pool = bridge_pool if rng.random() < 0.75 else general_pool
        candidate[idx] = replacement_pool[rng.randrange(len(replacement_pool))]

    if rng.random() < 0.55:
        src = preferred_positions[rng.randrange(len(preferred_positions))]
        comp = candidate.pop(src)
        lo = len(candidate) // 2 if hotspot_key and max(hotspot_key) >= TARGET_WIRES // 2 else 0
        dst = rng.randrange(lo, len(candidate) + 1)
        candidate.insert(dst, comp)
    return canonicalize_network(candidate, TARGET_WIRES)


def mutate_fixed_length(
    network: Sequence[Comparator],
    target_len: int,
    anchors: Sequence[Sequence[Comparator]],
    rng: random.Random,
) -> Network:
    candidate = list(network)
    fail_count, _ = evaluate_network(candidate, BASE_WIRES_14)
    if fail_count <= 256 and rng.random() < 0.42:
        candidate = focused_bridge_mutation(candidate, anchors, rng)
    pool = comparator_pool(candidate, anchors, rng)
    move = rng.random()
    if move < 0.34:
        idx = rng.randrange(len(candidate))
        candidate[idx] = pool[rng.randrange(len(pool))]
    elif move < 0.56:
        src = rng.randrange(len(candidate))
        comp = candidate.pop(src)
        dst = rng.randrange(len(candidate) + 1)
        candidate.insert(dst, comp)
    elif move < 0.74:
        a = rng.randrange(len(candidate))
        b = rng.randrange(len(candidate))
        candidate[a], candidate[b] = candidate[b], candidate[a]
    elif move < 0.90:
        idx = rng.randrange(len(candidate))
        del candidate[idx]
        candidate.insert(rng.randrange(len(candidate) + 1), pool[rng.randrange(len(pool))])
    else:
        width = max(1, min(4, 1 + int(rng.expovariate(1.0 / 1.8))))
        lo = rng.randrange(0, len(candidate) - width + 1)
        for idx in range(lo, lo + width):
            candidate[idx] = pool[rng.randrange(len(pool))]
    while len(candidate) > target_len:
        del candidate[rng.randrange(len(candidate))]
    while len(candidate) < target_len:
        candidate.insert(rng.randrange(len(candidate) + 1), pool[rng.randrange(len(pool))])
    return canonicalize_network(candidate, TARGET_WIRES)


def heavy_mutation(
    network: Sequence[Comparator],
    target_len: int,
    anchors: Sequence[Sequence[Comparator]],
    rng: random.Random,
) -> Network:
    candidate = canonicalize_network(network, TARGET_WIRES)
    rounds = 2 + int(rng.random() < 0.75) + int(rng.random() < 0.35)
    for _ in range(rounds):
        if rng.random() < 0.6:
            candidate = focused_bridge_mutation(candidate, anchors, rng)
        candidate = mutate_fixed_length(candidate, target_len, anchors, rng)
    return canonicalize_network(candidate, TARGET_WIRES)


def clamp_network_length(
    network: Sequence[Comparator],
    target_len: int,
    anchors: Sequence[Sequence[Comparator]],
    rng: random.Random,
) -> Network:
    candidate = canonicalize_network(network, TARGET_WIRES)
    while len(candidate) > target_len:
        idx = rng.randrange(len(candidate))
        candidate = canonicalize_network(candidate[:idx] + candidate[idx + 1 :], TARGET_WIRES)
    while len(candidate) < target_len:
        pool = comparator_pool(candidate, anchors, rng)
        expanded = list(candidate)
        expanded.insert(rng.randrange(len(expanded) + 1), pool[rng.randrange(len(pool))])
        candidate = canonicalize_network(expanded, TARGET_WIRES)
    return candidate


def best_single_delete_child(
    parent: Sequence[Comparator],
    child_len: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
) -> Tuple[Network, Tuple[int, int, int]]:
    parent_net = canonicalize_network(parent, TARGET_WIRES)
    if len(parent_net) <= child_len:
        child = clamp_network_length(parent_net, child_len, [parent_net], random.Random(0))
        return child, evaluate_cached(child, cache)
    if len(parent_net) != child_len + 1:
        reduced = deletion_beam(parent_net, child_len, beam_width=8, cache=cache)
        child = min(reduced, key=lambda net: score_for_target(net, child_len, cache))
        return child, evaluate_cached(child, cache)

    best_child = canonicalize_network(parent_net[:-1], TARGET_WIRES)
    best_score = evaluate_cached(best_child, cache)
    for idx in range(len(parent_net)):
        child = canonicalize_network(parent_net[:idx] + parent_net[idx + 1 :], TARGET_WIRES)
        score = evaluate_cached(child, cache)
        if (score[0], score[2]) < (best_score[0], best_score[2]):
            best_child = child
            best_score = score
    return best_child, best_score


def crossover_fixed_length(
    parent_a: Sequence[Comparator],
    parent_b: Sequence[Comparator],
    target_len: int,
    anchors: Sequence[Sequence[Comparator]],
    rng: random.Random,
) -> Network:
    a = canonicalize_network(parent_a, TARGET_WIRES)
    b = canonicalize_network(parent_b, TARGET_WIRES)
    mode = rng.random()
    if mode < 0.34:
        cut_a = rng.randrange(len(a) + 1)
        cut_b = rng.randrange(len(b) + 1)
        child = list(a[:cut_a]) + list(b[cut_b:])
    elif mode < 0.67:
        child = []
        for idx in range(max(len(a), len(b))):
            preferred = a if rng.random() < 0.5 else b
            fallback = b if preferred is a else a
            if idx < len(preferred):
                child.append(preferred[idx])
            elif idx < len(fallback):
                child.append(fallback[idx])
    else:
        layers_a = pack_layers(a, TARGET_WIRES)
        layers_b = pack_layers(b, TARGET_WIRES)
        child_layers: List[List[Comparator]] = []
        i = 0
        j = 0
        while i < len(layers_a) or j < len(layers_b):
            if i < len(layers_a) and (j >= len(layers_b) or rng.random() < 0.5):
                child_layers.append(list(layers_a[i]))
                i += 1
            if j < len(layers_b) and (i >= len(layers_a) or rng.random() < 0.5):
                child_layers.append(list(layers_b[j]))
                j += 1
        child = flatten_layers(child_layers)
    return clamp_network_length(child, target_len=target_len, anchors=[a, b] + list(anchors), rng=rng)


def select_diverse_beam(
    candidates: Sequence[Network],
    target_len: int,
    goal_length: int,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
    lookahead_cache: Dict[Tuple[Tuple[Comparator, ...], int, int], Tuple[int, ...]],
    beam_width: int,
) -> List[Network]:
    ranked = sorted(
        candidates,
        key=lambda net: descendant_aware_score(net, target_len, goal_length, cache, lookahead_cache),
    )
    shortlist = ranked[: max(beam_width * 3, beam_width)]
    selected: List[Network] = []
    signature_counts: Counter[Tuple[int, ...]] = Counter()
    soft_limit = max(2, beam_width // 6)
    for network in shortlist:
        signature = dominant_hotspot_key(network)
        if signature_counts[signature] >= soft_limit and len(selected) < beam_width // 2:
            continue
        selected.append(network)
        signature_counts[signature] += 1
        if len(selected) >= beam_width:
            return selected
    for network in ranked:
        if network in selected:
            continue
        selected.append(network)
        if len(selected) >= beam_width:
            break
    return selected


def targeted_replacement_pass(
    network: Sequence[Comparator],
    target_len: int,
    goal_length: int,
    anchors: Sequence[Sequence[Comparator]],
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
    lookahead_cache: Dict[Tuple[Tuple[Comparator, ...], int, int], Tuple[int, ...]],
) -> List[Network]:
    pool = comparator_pool(network, anchors, random.Random(0), limit=22)
    baseline = descendant_aware_score(network, target_len, goal_length, cache, lookahead_cache)
    best_map: Dict[Tuple[Comparator, ...], Network] = {}

    for idx in range(len(network)):
        old = network[idx]
        for comp in pool:
            if comp == old:
                continue
            candidate = list(network)
            candidate[idx] = comp
            candidate = canonicalize_network(candidate, TARGET_WIRES)
            if descendant_aware_score(candidate, target_len, goal_length, cache, lookahead_cache) < baseline:
                best_map.setdefault(tuple(candidate), candidate)

    if not best_map:
        return []
    return sorted(
        best_map.values(),
        key=lambda net: descendant_aware_score(net, target_len, goal_length, cache, lookahead_cache),
    )[:6]


def best_seed_networks() -> List[Network]:
    seeds = []
    batcher_seed = canonicalize_network(project_network(batcher16(), (0, 1), REFERENCE_WIRES), TARGET_WIRES)
    seeds.append(batcher_seed)
    if REFERENCE_JSON.exists():
        payload = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
        ref_network = canonicalize_network(sanitize_network(payload, REFERENCE_WIRES), REFERENCE_WIRES)
        for removed in ((0, 1), (14, 15), (0, 15)):
            projected = canonicalize_network(
                project_network(ref_network, removed_wires=removed, num_wires=REFERENCE_WIRES),
                TARGET_WIRES,
            )
            if projected:
                seeds.append(projected)
    unique: Dict[Tuple[Comparator, ...], Network] = {}
    for seed in seeds:
        unique.setdefault(tuple(seed), seed)
    correct = []
    for seed in unique.values():
        fail_count, _ = evaluate_network(seed, BASE_WIRES_14)
        if fail_count == 0:
            correct.append(seed)
    return sorted(correct, key=lambda net: (len(net), depth_of(net, TARGET_WIRES)))


def search_target_length(
    target_len: int,
    goal_length: int,
    seeds: Sequence[Sequence[Comparator]],
    state_file: Path,
    rng: random.Random,
    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]],
    lookahead_cache: Dict[Tuple[Tuple[Comparator, ...], int, int], Tuple[int, ...]],
    beam_width: int,
    children_per_parent: int,
    max_seconds: float,
    heartbeat_seconds: float,
) -> Tuple[Network, Tuple[int, int, int]]:
    deadline = time.time() + max_seconds
    last_heartbeat = time.time()
    anchors = [canonicalize_network(seed, TARGET_WIRES) for seed in seeds]
    initial_map: Dict[Tuple[Comparator, ...], Network] = {}
    for seed in anchors:
        for candidate in deletion_beam(
            seed,
            target_len,
            goal_length,
            beam_width=max(beam_width, 8),
            cache=cache,
            lookahead_cache=lookahead_cache,
        ):
            initial_map.setdefault(tuple(candidate), candidate)
    state = load_state(state_file)
    persisted = state.get("best_by_length", {}).get(str(target_len))
    if isinstance(persisted, dict):
        persisted_network = canonicalize_network(sanitize_network(persisted, TARGET_WIRES), TARGET_WIRES)
        if persisted_network:
            initial_map.setdefault(tuple(persisted_network), persisted_network)
    beam = sorted(
        select_diverse_beam(list(initial_map.values()), target_len, goal_length, cache, lookahead_cache, beam_width),
        key=lambda net: descendant_aware_score(net, target_len, goal_length, cache, lookahead_cache),
    )[:beam_width]
    if not beam:
        raise RuntimeError(f"no initial candidates for target length {target_len}")

    best = beam[0]
    best_score = evaluate_cached(best, cache)
    save_state(
        state_file,
        target_len=target_len,
        network=best,
        fail_count=best_score[0],
        depth=best_score[2],
        metadata={"stage": "initial_beam"},
    )
    print(
        f"[target {target_len}] init fail={best_score[0]} depth={best_score[2]} beam={len(beam)}",
        flush=True,
    )

    generation = 0
    stagnation = 0
    while time.time() < deadline:
        generation += 1
        next_map: Dict[Tuple[Comparator, ...], Network] = {tuple(net): net for net in beam}
        parent_support = [anchor for anchor in anchors if len(anchor) >= target_len + 1]
        if target_len == TARGET_LEN and parent_support:
            parent_candidates: List[Network] = []
            for parent in parent_support[: min(6, len(parent_support))]:
                parent_candidates.append(clamp_network_length(parent, target_len + 1, anchors, rng))
            for parent in parent_candidates:
                child, _ = best_single_delete_child(parent, target_len, cache)
                next_map.setdefault(tuple(child), child)
            parent_trials = max(4, len(parent_candidates))
            for _ in range(parent_trials):
                parent_a = parent_candidates[rng.randrange(len(parent_candidates))]
                parent_b = parent_candidates[rng.randrange(len(parent_candidates))]
                if rng.random() < 0.5:
                    parent_child = crossover_fixed_length(
                        parent_a,
                        parent_b,
                        target_len=target_len + 1,
                        anchors=anchors,
                        rng=rng,
                    )
                else:
                    parent_child = mutate_fixed_length(
                        parent_a,
                        target_len=target_len + 1,
                        anchors=anchors,
                        rng=rng,
                    )
                child, child_score = best_single_delete_child(parent_child, target_len, cache)
                next_map.setdefault(tuple(child), child)
                worst_idx = max(
                    range(len(parent_candidates)),
                    key=lambda idx: best_single_delete_child(parent_candidates[idx], target_len, cache)[1],
                )
                worst_score = best_single_delete_child(parent_candidates[worst_idx], target_len, cache)[1]
                if (child_score[0], child_score[2]) <= (worst_score[0], worst_score[2]):
                    parent_candidates[worst_idx] = parent_child
        for network in beam[: min(3, len(beam))]:
            for candidate in targeted_replacement_pass(
                network,
                target_len=target_len,
                goal_length=goal_length,
                anchors=anchors,
                cache=cache,
                lookahead_cache=lookahead_cache,
            ):
                next_map.setdefault(tuple(candidate), candidate)
        crossover_trials = max(2, len(beam) // 3)
        for _ in range(crossover_trials):
            parent_a = beam[rng.randrange(len(beam))]
            parent_b = beam[rng.randrange(len(beam))]
            child = crossover_fixed_length(
                parent_a,
                parent_b,
                target_len=target_len,
                anchors=anchors,
                rng=rng,
            )
            next_map.setdefault(tuple(child), child)
        for network in beam:
            for _ in range(children_per_parent):
                child = mutate_fixed_length(
                    network,
                    target_len=target_len,
                    anchors=anchors,
                    rng=rng,
                )
                next_map.setdefault(tuple(child), child)
        ranked = sorted(
            select_diverse_beam(list(next_map.values()), target_len, goal_length, cache, lookahead_cache, beam_width),
            key=lambda net: descendant_aware_score(net, target_len, goal_length, cache, lookahead_cache),
        )
        beam = ranked[:beam_width]
        candidate = beam[0]
        candidate_score = evaluate_cached(candidate, cache)
        if (candidate_score[0], candidate_score[2]) < (best_score[0], best_score[2]):
            best = candidate
            best_score = candidate_score
            stagnation = 0
            save_state(
                state_file,
                target_len=target_len,
                network=best,
                fail_count=best_score[0],
                depth=best_score[2],
                metadata={"generation": generation},
            )
            print(
                f"[target {target_len}] improve fail={best_score[0]} depth={best_score[2]} gen={generation}",
                flush=True,
            )
            if best_score[0] == 0:
                break
        else:
            stagnation += 1

        restart_interval = 10 if target_len <= TARGET_LEN else 14
        if stagnation >= restart_interval:
            restart_map: Dict[Tuple[Comparator, ...], Network] = {tuple(best): best}
            for seed in anchors[: min(4, len(anchors))]:
                for candidate in deletion_beam(
                    seed,
                    target_len=target_len,
                    goal_length=goal_length,
                    beam_width=max(8, beam_width // 2),
                    cache=cache,
                    lookahead_cache=lookahead_cache,
                ):
                    restart_map.setdefault(tuple(candidate), candidate)
            restart_sources = list(beam[: min(6, len(beam))]) + [best]
            for source in restart_sources:
                for _ in range(max(beam_width // 2, 8)):
                    candidate = heavy_mutation(
                        source,
                        target_len=target_len,
                        anchors=anchors,
                        rng=rng,
                    )
                    restart_map.setdefault(tuple(candidate), candidate)
            beam = sorted(
                select_diverse_beam(
                    list(restart_map.values()),
                    target_len,
                    goal_length,
                    cache,
                    lookahead_cache,
                    beam_width,
                ),
                key=lambda net: descendant_aware_score(net, target_len, goal_length, cache, lookahead_cache),
            )[:beam_width]
            print(
                f"[target {target_len}] diversify fail={best_score[0]} depth={best_score[2]} gen={generation}",
                flush=True,
            )
            stagnation = 0
        now = time.time()
        if now - last_heartbeat >= heartbeat_seconds:
            unique = len(next_map)
            print(
                f"[target {target_len}] heartbeat fail={best_score[0]} depth={best_score[2]} "
                f"gen={generation} unique={unique}",
                flush=True,
            )
            last_heartbeat = now
    return best, best_score


def run_search(args: argparse.Namespace) -> int:
    analysis = analyze_reference_candidate()
    args.analysis_file.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_file.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    seeds = best_seed_networks()
    if not seeds:
        raise RuntimeError("failed to construct any correct 14-wire seed networks")
    print(
        f"[analysis] wrote {args.analysis_file} and found {len(seeds)} correct seed family members",
        flush=True,
    )
    for idx, seed in enumerate(seeds[:4], start=1):
        fail_count, _ = evaluate_network(seed, BASE_WIRES_14)
        print(
            f"[seed {idx}] len={len(seed)} depth={depth_of(seed, TARGET_WIRES)} fail={fail_count}",
            flush=True,
        )

    seed_value = resolve_seed(args.seed)
    rng = random.Random(seed_value)
    print(f"[seed] {seed_value}", flush=True)

    state_before = load_state(args.state_file)
    previous_scores = summarize_best_scores(state_before)
    commit_label = current_commit_label()
    run_started = time.time()
    update_run_state(
        args.state_file,
        {
            "status": "running",
            "commit": commit_label,
            "seed": seed_value,
            "goal_length": int(args.goal_length),
            "max_seconds": float(args.max_seconds),
            "started_unix_ts": run_started,
        },
    )

    cache: Dict[Tuple[Comparator, ...], Tuple[int, int, int]] = {}
    lookahead_cache: Dict[Tuple[Tuple[Comparator, ...], int, int], Tuple[int, ...]] = {}
    search_lengths = list(range(min(len(seed) for seed in seeds) - 1, args.goal_length - 1, -1))
    if args.goal_length not in search_lengths:
        search_lengths.append(args.goal_length)
    search_lengths = sorted(set(length for length in search_lengths if length >= args.goal_length), reverse=True)

    frontier = dedupe_networks(load_state_frontier(args.state_file, args.goal_length) + list(seeds))
    results = []
    target_budgets = allocate_target_budgets(
        search_lengths=search_lengths,
        total_seconds=float(args.max_seconds),
        goal_length=int(args.goal_length),
        previous_scores=previous_scores,
    )
    for target_len in search_lengths:
        seed_frontier = select_seed_frontier(
            frontier,
            target_len=target_len,
            cache=cache,
            limit=max(4, args.seed_frontier),
        )
        target_budget = target_budgets.get(target_len, max(1.0, args.max_seconds / max(1, len(search_lengths))))
        target_children = args.children_per_parent + (3 if target_len == args.goal_length else 0)
        best, best_score = search_target_length(
            target_len=target_len,
            goal_length=args.goal_length,
            seeds=seed_frontier,
            state_file=args.state_file,
            rng=rng,
            cache=cache,
            lookahead_cache=lookahead_cache,
            beam_width=args.beam_width,
            children_per_parent=target_children,
            max_seconds=target_budget,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        results.append((target_len, best_score, best, len(seed_frontier), target_budget))
        frontier = dedupe_networks([best] + frontier)

    print("[summary]", flush=True)
    for target_len, best_score, _, _, target_budget in results:
        print(
            f"  target_len={target_len} fail={best_score[0]} depth={best_score[2]} budget={target_budget:.1f}s",
            flush=True,
        )
    for target_len, best_score, _, frontier_size, target_budget in results:
        previous = previous_scores.get(str(target_len))
        improved = previous is None or (best_score[0], best_score[2]) < previous
        status = "discard"
        if improved:
            status = "breakthrough" if best_score[0] == 0 and target_len < TARGET_LT_COMPARATORS else "keep"
        prev_text = "none" if previous is None else f"{previous[0]}/{previous[1]}"
        append_results_row(
            args.results_file,
            commit_label=commit_label,
            target_len=target_len,
            fail_count=best_score[0],
            depth=best_score[2],
            status=status,
            description=(
                f"seed={seed_value} prev={prev_text} new={best_score[0]}/{best_score[2]} "
                f"frontier={frontier_size} budget={target_budget:.1f}s"
            ),
        )

    update_run_state(
        args.state_file,
        {
            "status": "completed",
            "commit": commit_label,
            "seed": seed_value,
            "goal_length": int(args.goal_length),
            "max_seconds": float(args.max_seconds),
            "target_budgets": {str(target_len): float(target_budgets[target_len]) for target_len in search_lengths},
            "search_lengths": search_lengths,
            "started_unix_ts": run_started,
            "finished_unix_ts": time.time(),
            "summary": {
                str(target_len): {
                    "fail_count": int(best_score[0]),
                    "depth": int(best_score[2]),
                }
                for target_len, best_score, _, _, _ in results
            },
        },
    )

    final_target, final_score, final_network, _, _ = results[-1]
    if final_score[0] == 0 and final_target < TARGET_LT_COMPARATORS:
        print("[found] correct network under 51 comparators", flush=True)
        print(json.dumps([[i, j] for i, j in final_network]), flush=True)
        return 0
    print(
        f"[best-under-{TARGET_LT_COMPARATORS}] target_len={final_target} fail={final_score[0]} depth={final_score[2]}",
        flush=True,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--goal-length", type=int, default=TARGET_LEN)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--children-per-parent", type=int, default=6)
    parser.add_argument("--seed-frontier", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    analysis = analyze_reference_candidate()
    args.analysis_file.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_file.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    if args.analyze_only:
        print(f"[analysis] wrote {args.analysis_file}", flush=True)
        return 0
    return run_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
