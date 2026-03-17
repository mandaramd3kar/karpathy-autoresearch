"""Graph-based Lamarckian-Baldwinian hybrid search for n=16 sorting networks <63.

This keeps the external persisted-state interface compatible with _search_under63.py:
- reads best candidate from --state-file (default: ga_best_candidate.json)
- writes improved best candidate back to --state-file

Additionally, it maintains a side copy file next to state_file so best state is not lost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Comparator = Tuple[int, int]
Network = List[Comparator]
Layers = List[List[Comparator]]
Score = Tuple[int, int, int]  # (fail_count, comparator_count, depth)

N = 16


def build_wire_masks(num_wires: int):
    total = 1 << num_wires
    wires = [0] * num_wires
    for case in range(total):
        for wire in range(num_wires):
            if (case >> (num_wires - 1 - wire)) & 1:
                wires[wire] |= 1 << case
    return tuple(wires)


BASE_WIRES = build_wire_masks(N)


def fail_count(network: Sequence[Comparator]) -> int:
    wires = list(BASE_WIRES)
    for i, j in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    invalid = 0
    for i in range(N - 1):
        invalid |= wires[i] & ~wires[i + 1]
    return invalid.bit_count()


def batcher16() -> Network:
    out: Network = []

    def odd_even_merge(lo: int, n: int, r: int):
        step = r * 2
        if step < n:
            odd_even_merge(lo, n, step)
            odd_even_merge(lo + r, n, step)
            for i in range(lo + r, lo + n - r, step):
                out.append((i, i + r))
        else:
            out.append((lo, lo + r))

    def odd_even_merge_sort(lo: int, n: int):
        if n > 1:
            m = n // 2
            odd_even_merge_sort(lo, m)
            odd_even_merge_sort(lo + m, m)
            odd_even_merge(lo, n, 1)

    odd_even_merge_sort(0, 16)
    return out


def random_comp(rng: random.Random) -> Comparator:
    i = rng.randrange(N)
    j = rng.randrange(N - 1)
    if j >= i:
        j += 1
    if j < i:
        i, j = j, i
    return (i, j)


def _normalize_comp(raw) -> Comparator | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        i = int(raw[0])
        j = int(raw[1])
    except (TypeError, ValueError):
        return None
    if i == j:
        return None
    if not (0 <= i < N and 0 <= j < N):
        return None
    if j < i:
        i, j = j, i
    return (i, j)


def _sanitize_network(raw) -> Network:
    if not isinstance(raw, (list, tuple)):
        return []
    out: Network = []
    for item in raw:
        comp = _normalize_comp(item)
        if comp is not None:
            out.append(comp)
    return out


def state_copy_path(state_file: Path) -> Path:
    return state_file.with_name(f"{state_file.stem}.copy{state_file.suffix}")


def load_state_network(state_file: Path) -> Tuple[Network | None, Score | None, Path | None]:
    candidates = [state_file, state_copy_path(state_file)]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                network = _sanitize_network(payload.get("network"))
                score = None
                if (
                    "fail_count" in payload
                    and "comparator_count" in payload
                    and "depth" in payload
                ):
                    score = (
                        int(payload["fail_count"]),
                        int(payload["comparator_count"]),
                        int(payload["depth"]),
                    )
            else:
                network = _sanitize_network(payload)
                score = None
            if network:
                return network, score, path
        except Exception:
            continue
    return None, None, None


def save_state_network(
    state_file: Path,
    network: Network,
    score: Score,
    generation: int,
    evals: int,
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_wires": N,
        "fail_count": int(score[0]),
        "comparator_count": int(score[1]),
        "depth": int(score[2]),
        "generation": int(generation),
        "evaluations": int(evals),
        "updated_unix_ts": time.time(),
        "network": [[i, j] for i, j in network],
    }

    # Primary state file (atomic replace)
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(state_file)

    # Copy state file (loss-protection mirror)
    copy_path = state_copy_path(state_file)
    tmp_copy = copy_path.with_suffix(copy_path.suffix + ".tmp")
    with tmp_copy.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_copy.replace(copy_path)


def resolve_seed(cli_seed: int | None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    dt = datetime.now().isoformat(timespec="microseconds")
    entropy = f"{dt}|{time.time_ns()}|{os.getpid()}|{os.urandom(8).hex()}".encode("utf-8")
    digest = hashlib.blake2b(entropy, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def to_layers(network: Sequence[Comparator]) -> Layers:
    # Dependency-preserving stage assignment:
    # A comparator can only be placed after the last stage touching either wire.
    layers: Layers = []
    last_stage = [-1] * N
    for i, j in network:
        stage = max(last_stage[i], last_stage[j]) + 1
        while len(layers) <= stage:
            layers.append([])
        layers[stage].append((i, j))
        last_stage[i] = stage
        last_stage[j] = stage
    return layers


def flatten_layers(layers: Layers) -> Network:
    out: Network = []
    for layer in layers:
        out.extend(layer)
    return out


def canonicalize_network(network: Sequence[Comparator]) -> Network:
    sanitized = []
    for comp in network:
        c = _normalize_comp(comp)
        if c is not None:
            sanitized.append(c)
    return flatten_layers(to_layers(sanitized))


def _clamp_len(network: Network, rng: random.Random, min_len: int, max_len: int) -> Network:
    out = list(network)
    while len(out) > max_len:
        del out[rng.randrange(len(out))]
    while len(out) < min_len:
        out.insert(rng.randrange(len(out) + 1), random_comp(rng))
    return canonicalize_network(out)


def score_of(network: Network, cache: Dict[Tuple[Comparator, ...], Score]) -> Score:
    key = tuple(network)
    got = cache.get(key)
    if got is not None:
        return got
    layers = to_layers(network)
    result = (fail_count(network), len(network), len(layers))
    cache[key] = result
    return result


def is_better(a: Score, b: Score) -> bool:
    return a < b


def mutate_graph(
    network: Network,
    rng: random.Random,
    min_len: int,
    max_len: int,
    steps: int,
) -> Network:
    layers = [list(layer) for layer in to_layers(network)]
    if not layers:
        layers = [[random_comp(rng)]]

    for _ in range(max(1, steps)):
        move = rng.random()
        flat = flatten_layers(layers)
        if not flat:
            flat = [random_comp(rng)]

        if move < 0.20:
            # Replace a comparator endpoint pair.
            idx = rng.randrange(len(flat))
            flat[idx] = random_comp(rng)
            layers = to_layers(flat)
        elif move < 0.34:
            # Swap two comparators in linearized order.
            if len(flat) >= 2:
                a = rng.randrange(len(flat))
                b = rng.randrange(len(flat))
                flat[a], flat[b] = flat[b], flat[a]
                layers = to_layers(flat)
        elif move < 0.50:
            # Delete comparator.
            if len(flat) > min_len:
                del flat[rng.randrange(len(flat))]
                layers = to_layers(flat)
        elif move < 0.64:
            # Insert comparator.
            if len(flat) < max_len:
                flat.insert(rng.randrange(len(flat) + 1), random_comp(rng))
                layers = to_layers(flat)
        elif move < 0.78:
            # Segment reverse in flattened graph.
            if len(flat) >= 2:
                w = max(1, min(len(flat), 1 + int(rng.expovariate(1.0 / 3.0))))
                lo = rng.randrange(0, len(flat) - w + 1)
                hi = lo + w
                flat[lo:hi] = reversed(flat[lo:hi])
                layers = to_layers(flat)
        elif move < 0.90:
            # Move one comparator to another layer position.
            src_l = rng.randrange(len(layers))
            if layers[src_l]:
                src_i = rng.randrange(len(layers[src_l]))
                comp = layers[src_l].pop(src_i)
                if not layers[src_l]:
                    del layers[src_l]
                dst_l = rng.randrange(len(layers) + 1)
                if dst_l == len(layers):
                    layers.append([comp])
                else:
                    dst_i = rng.randrange(len(layers[dst_l]) + 1)
                    layers[dst_l].insert(dst_i, comp)
                layers = to_layers(flatten_layers(layers))
        else:
            # Burst rewrite.
            rewrite = max(1, min(len(flat), 1 + int(rng.expovariate(1.0 / 4.0))))
            for _ in range(rewrite):
                flat[rng.randrange(len(flat))] = random_comp(rng)
            layers = to_layers(flat)

    return _clamp_len(flatten_layers(layers), rng, min_len=min_len, max_len=max_len)


def crossover_graph(
    parent_a: Network,
    parent_b: Network,
    rng: random.Random,
    min_len: int,
    max_len: int,
) -> Network:
    layers_a = to_layers(parent_a)
    layers_b = to_layers(parent_b)

    mode = rng.random()
    if mode < 0.45:
        # Prefix/suffix by layer cuts.
        cut_a = rng.randrange(len(layers_a) + 1)
        cut_b = rng.randrange(len(layers_b) + 1)
        child_layers = layers_a[:cut_a] + layers_b[cut_b:]
        child = flatten_layers(child_layers)
    elif mode < 0.8:
        # Interleave layers.
        child_layers: Layers = []
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
    else:
        # Uniform comparator mix.
        child = []
        max_n = max(len(parent_a), len(parent_b))
        for k in range(max_n):
            pick_a = rng.random() < 0.5
            if pick_a and k < len(parent_a):
                child.append(parent_a[k])
            elif (not pick_a) and k < len(parent_b):
                child.append(parent_b[k])
            elif k < len(parent_a):
                child.append(parent_a[k])
            elif k < len(parent_b):
                child.append(parent_b[k])

    return _clamp_len(canonicalize_network(child), rng, min_len=min_len, max_len=max_len)


def local_learning(
    network: Network,
    rng: random.Random,
    min_len: int,
    max_len: int,
    budget_steps: int,
    cache: Dict[Tuple[Comparator, ...], Score],
) -> Tuple[Network, Score]:
    # Learning operator used for both Baldwin (phenotype-only) and Lamarck (genotype update).
    current = list(network)
    current_score = score_of(current, cache)
    best = list(current)
    best_score = current_score

    for _ in range(max(1, budget_steps)):
        # Prefer small local edits, sometimes stronger jumps.
        if rng.random() < 0.75:
            steps = 1 + int(rng.expovariate(1.0 / 2.0))
        else:
            steps = 3 + int(rng.expovariate(1.0 / 5.0))
        cand = mutate_graph(best, rng, min_len=min_len, max_len=max_len, steps=steps)
        cand_score = score_of(cand, cache)
        if is_better(cand_score, best_score):
            best = cand
            best_score = cand_score

    return best, best_score


def make_seed_candidate(
    rng: random.Random,
    base: Network,
    min_len: int,
    max_len: int,
    persisted: Network | None,
    anchor: Network | None,
    heavy: bool,
) -> Network:
    roll = rng.random()
    if persisted is not None and roll < 0.4:
        seed = list(persisted)
    elif anchor is not None and roll < 0.8:
        seed = list(anchor)
    else:
        seed = list(base)

    target_len = min_len + rng.randrange(max_len - min_len + 1)
    seed = _clamp_len(seed, rng, min_len=target_len, max_len=target_len)
    if rng.random() < 0.35:
        rng.shuffle(seed)

    if heavy:
        steps = 4 + int(rng.expovariate(1.0 / 7.0))
    else:
        steps = 1 + int(rng.expovariate(1.0 / 3.0))
    return mutate_graph(seed, rng, min_len=min_len, max_len=max_len, steps=steps)


def tournament_pick(
    population: Sequence[Network],
    fitness: Sequence[Tuple[int, int, int, int]],
    rng: random.Random,
    k: int,
) -> Network:
    best_idx = None
    for _ in range(max(2, k)):
        idx = rng.randrange(len(population))
        if best_idx is None or fitness[idx] < fitness[best_idx]:
            best_idx = idx
    return list(population[best_idx])


def select_parent(
    population: Sequence[Network],
    fitness: Sequence[Tuple[int, int, int, int]],
    ranked: Sequence[int],
    rng: random.Random,
    stagnation: int,
) -> Network:
    roll = rng.random()
    if roll < 0.7:
        if stagnation < 400:
            t = 5
        elif stagnation < 1600:
            t = 4
        else:
            t = 3
        return tournament_pick(population, fitness, rng, k=t)
    if roll < 0.9:
        top = max(4, len(population) // 4)
        return list(population[ranked[rng.randrange(top)]])
    return list(population[rng.randrange(len(population))])


def run_lbh_graph_search(
    seed: int,
    population_size: int,
    generations: int,
    heartbeat_seconds: float,
    max_seconds: float,
    state_file: Path,
    min_len: int,
    max_len: int,
    lamarck_rate: float,
    baldwin_rate: float,
    learning_steps: int,
) -> Network | None:
    rng = random.Random(seed)
    base = batcher16()
    assert len(base) == 63 and fail_count(base) == 0

    start_ts = time.time()
    last_heartbeat = start_ts
    eval_cache: Dict[Tuple[Comparator, ...], Score] = {}
    eval_count = 0

    def score_cached(net: Network) -> Score:
        nonlocal eval_count
        key = tuple(net)
        if key not in eval_cache:
            eval_cache[key] = score_of(net, eval_cache)
            eval_count += 1
        return eval_cache[key]

    persisted, persisted_score_meta, persisted_source = load_state_network(state_file)
    persisted_seed = None
    if persisted:
        persisted_sanitized = canonicalize_network(persisted)
        persisted_eval_score = score_cached(persisted_sanitized)
        persisted_seed = (
            list(persisted_sanitized)
            if min_len <= len(persisted_sanitized) <= max_len
            else _clamp_len(persisted_sanitized, rng, min_len, max_len)
        )
        print(
            f"[state] loaded candidate len={len(persisted_seed)} "
            f"fail={persisted_eval_score[0]} from '{persisted_source}'",
            flush=True,
        )
    else:
        print(f"[state] no prior state at '{state_file}'", flush=True)

    population: List[Network] = []
    if persisted_seed is not None:
        population.append(list(persisted_seed))
    while len(population) < population_size:
        population.append(
            make_seed_candidate(
                rng,
                base=base,
                min_len=min_len,
                max_len=max_len,
                persisted=persisted_seed,
                anchor=None,
                heavy=False,
            )
        )

    best = min(population, key=score_cached)
    best_score = score_cached(best)
    if persisted and persisted_score_meta is not None:
        # Guard against any mismatch: use the better of metadata and recomputed score.
        saved_score = persisted_score_meta if is_better(persisted_score_meta, persisted_eval_score) else persisted_eval_score
    elif persisted:
        saved_score = persisted_eval_score
    else:
        saved_score = (1 << 60, 1 << 60, 1 << 60)
    stagnation = 0

    def maybe_save(candidate: Network, sc: Score, generation: int):
        nonlocal saved_score, persisted_seed
        if is_better(sc, saved_score):
            save_state_network(
                state_file=state_file,
                network=candidate,
                score=sc,
                generation=generation,
                evals=eval_count,
            )
            saved_score = sc
            persisted_seed = list(candidate)
            print(
                f"[state] saved best fail={sc[0]} len={sc[1]} depth={sc[2]} "
                f"to '{state_file}' and '{state_copy_path(state_file)}'",
                flush=True,
            )

    print(
        "[start] graph-based Lamarckian-Baldwinian hybrid GA search",
        flush=True,
    )
    print(
        f"[init] population={population_size} best_fail={best_score[0]} "
        f"best_len={best_score[1]} best_depth={best_score[2]}",
        flush=True,
    )
    maybe_save(best, best_score, generation=0)

    for gen in range(1, generations + 1):
        elapsed = time.time() - start_ts
        if elapsed >= max_seconds:
            print(f"[stop] max_seconds reached ({elapsed:.1f}s)", flush=True)
            break

        raw_scores = [score_cached(net) for net in population]
        raw_ranked = sorted(range(len(population)), key=lambda i: raw_scores[i])

        # Baldwin learning pool: top raw performers + random sample.
        baldwin_pool = set(raw_ranked[: max(2, int(population_size * baldwin_rate * 0.6))])
        need_random = max(0, int(population_size * baldwin_rate) - len(baldwin_pool))
        for _ in range(need_random):
            baldwin_pool.add(rng.randrange(len(population)))

        learned_networks: List[Network | None] = [None] * len(population)
        learned_scores: List[Score | None] = [None] * len(population)
        learn_budget = learning_steps + min(10, stagnation // 700)

        for idx in baldwin_pool:
            improved_net, improved_score = local_learning(
                population[idx],
                rng,
                min_len=min_len,
                max_len=max_len,
                budget_steps=learn_budget,
                cache=eval_cache,
            )
            learned_networks[idx] = improved_net
            learned_scores[idx] = improved_score

        # Baldwinian ranking: after-learning score dominates, raw score tie-breaker.
        combined_fitness: List[Tuple[int, int, int, int]] = []
        for i, raw in enumerate(raw_scores):
            learned = learned_scores[i] if learned_scores[i] is not None else raw
            combined_fitness.append((learned[0], raw[0], learned[1], learned[2]))

        ranked = sorted(range(len(population)), key=lambda i: combined_fitness[i])

        # Lamarckian assimilation on top-ranked individuals.
        lamarck_count = max(1, int(population_size * lamarck_rate))
        for idx in ranked[:lamarck_count]:
            ln = learned_networks[idx]
            ls = learned_scores[idx]
            if ln is not None and ls is not None and is_better(ls, raw_scores[idx]):
                population[idx] = ln
                raw_scores[idx] = ls

        current_best = population[ranked[0]]
        current_best_score = score_cached(current_best)
        # Also consider top learned phenotype even if not written genetically.
        learned_best_idx = min(ranked[: max(5, population_size // 10)], key=lambda i: combined_fitness[i])
        lb_score = learned_scores[learned_best_idx]
        lb_net = learned_networks[learned_best_idx]
        if lb_score is not None and lb_net is not None and is_better(lb_score, current_best_score):
            current_best = lb_net
            current_best_score = lb_score

        if is_better(current_best_score, best_score):
            best = list(current_best)
            best_score = current_best_score
            stagnation = 0
            print(
                f"[improve] gen={gen} fail={best_score[0]} len={best_score[1]} "
                f"depth={best_score[2]} elapsed={elapsed:.1f}s evals={eval_count}",
                flush=True,
            )
            maybe_save(best, best_score, generation=gen)
            if best_score[0] == 0 and best_score[1] < 63:
                print(f"[found] comparators={best_score[1]} depth={best_score[2]} gen={gen}", flush=True)
                return best
        else:
            stagnation += 1

        if stagnation > 5000:
            print(
                f"[diversify] gen={gen} stagnation={stagnation}; performing heavy population refresh",
                flush=True,
            )
            keep = max(2, population_size // 25)
            survivors = [list(best)]
            if persisted_seed is not None:
                survivors.append(list(persisted_seed))
            for idx in ranked[1 : 1 + keep]:
                survivors.append(list(population[idx]))
            population = survivors[:]
            while len(population) < population_size:
                population.append(
                    make_seed_candidate(
                        rng,
                        base=base,
                        min_len=min_len,
                        max_len=max_len,
                        persisted=persisted_seed,
                        anchor=best,
                        heavy=True,
                    )
                )
            stagnation = 1200
            continue

        if stagnation < 400:
            elite_count = max(2, population_size // 10)
        elif stagnation < 1800:
            elite_count = max(2, population_size // 14)
        else:
            elite_count = max(2, population_size // 20)

        immigrants = min(0.55, 0.08 + 0.00006 * stagnation)
        next_population: List[Network] = [list(population[idx]) for idx in ranked[:elite_count]]
        while len(next_population) < population_size:
            if rng.random() < immigrants:
                child = make_seed_candidate(
                    rng,
                    base=base,
                    min_len=min_len,
                    max_len=max_len,
                    persisted=persisted_seed,
                    anchor=best,
                    heavy=stagnation > 1400,
                )
            else:
                p1 = select_parent(population, combined_fitness, ranked, rng, stagnation)
                p2 = select_parent(population, combined_fitness, ranked, rng, stagnation)
                child = crossover_graph(p1, p2, rng, min_len=min_len, max_len=max_len)
                mut_steps = 1 + int(rng.expovariate(1.0 / 2.5)) + min(10, stagnation // 500)
                if rng.random() < min(0.35, stagnation / 7000.0):
                    mut_steps += 2 + int(rng.expovariate(1.0 / 3.5))
                child = mutate_graph(child, rng, min_len=min_len, max_len=max_len, steps=mut_steps)
            next_population.append(child)
        population = next_population

        now = time.time()
        if now - last_heartbeat >= heartbeat_seconds:
            unique = len({tuple(net) for net in population})
            print(
                f"[heartbeat] gen={gen}/{generations} fail={best_score[0]} len={best_score[1]} "
                f"depth={best_score[2]} unique={unique}/{population_size} "
                f"stagnation={stagnation} immigrants={immigrants:.2f} "
                f"learn_budget={learn_budget} evals={eval_count} elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_heartbeat = now

        if len(eval_cache) > 700000:
            keep_keys = {tuple(best)}
            if persisted_seed is not None:
                keep_keys.add(tuple(persisted_seed))
            for net in population[: min(300, len(population))]:
                keep_keys.add(tuple(net))
            eval_cache = {k: v for k, v in eval_cache.items() if k in keep_keys}

    maybe_save(best, best_score, generation=generations)
    print(
        f"[done] best_fail={best_score[0]} best_len={best_score[1]} best_depth={best_score[2]} "
        f"evals={eval_count} elapsed={time.time() - start_ts:.1f}s",
        flush=True,
    )
    if best_score[0] == 0 and best_score[1] < 63:
        return best
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=140)
    parser.add_argument("--generations", type=int, default=3000)
    parser.add_argument("--heartbeat-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--state-file", type=str, default="ga_best_candidate.json")
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=62)
    parser.add_argument("--lamarck-rate", type=float, default=0.22)
    parser.add_argument("--baldwin-rate", type=float, default=0.65)
    parser.add_argument("--learning-steps", type=int, default=7)
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    if args.seed is None:
        print(f"[seed] auto-generated seed={seed}", flush=True)
    else:
        print(f"[seed] using provided seed={seed}", flush=True)

    result = run_lbh_graph_search(
        seed=seed,
        population_size=max(20, args.population_size),
        generations=max(1, args.generations),
        heartbeat_seconds=args.heartbeat_seconds,
        max_seconds=args.max_seconds,
        state_file=Path(args.state_file),
        min_len=max(1, args.min_len),
        max_len=max(1, args.max_len),
        lamarck_rate=max(0.0, min(1.0, args.lamarck_rate)),
        baldwin_rate=max(0.0, min(1.0, args.baldwin_rate)),
        learning_steps=max(1, args.learning_steps),
    )
    if result is None:
        print("NOT_FOUND", flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
