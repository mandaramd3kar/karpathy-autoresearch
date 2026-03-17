"""Genetic Algorithm search utility to find a correct n=16 network under 63 comparators."""

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


def load_state_network(state_file: Path) -> Network | None:
    if not state_file.exists():
        return None
    try:
        with state_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            network = _sanitize_network(payload.get("network"))
        else:
            network = _sanitize_network(payload)
        return network or None
    except Exception as exc:
        print(f"[state] failed to load '{state_file}': {exc}", flush=True)
        return None


def save_state_network(
    state_file: Path,
    network: Network,
    fail: int,
    generation: int,
    evals: int,
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_wires": N,
        "fail_count": int(fail),
        "comparator_count": int(len(network)),
        "generation": int(generation),
        "evaluations": int(evals),
        "updated_unix_ts": time.time(),
        "network": [[i, j] for i, j in network],
    }
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(state_file)


def is_better(candidate_fail: int, candidate_len: int, best_fail: int, best_len: int) -> bool:
    return (candidate_fail, candidate_len) < (best_fail, best_len)


def _clamp_length(net: Network, rng: random.Random, min_len: int, max_len: int) -> Network:
    out = list(net)
    while len(out) > max_len:
        del out[rng.randrange(len(out))]
    while len(out) < min_len:
        out.insert(rng.randrange(len(out) + 1), random_comp(rng))
    return out


def mutate_network(
    network: Network,
    rng: random.Random,
    min_len: int,
    max_len: int,
    mutation_steps: int,
) -> Network:
    cand = list(network)
    if not cand:
        return _clamp_length(cand, rng, min_len=min_len, max_len=max_len)

    for _ in range(max(1, mutation_steps)):
        n = len(cand)
        move = rng.random()
        if move < 0.28:
            # Point replacement.
            idx = rng.randrange(n)
            cand[idx] = random_comp(rng)
        elif move < 0.48:
            # Position swap.
            a = rng.randrange(n)
            b = rng.randrange(n)
            cand[a], cand[b] = cand[b], cand[a]
        elif move < 0.62:
            # Segment reverse.
            width = max(1, min(n, 1 + int(rng.expovariate(1.0 / 3.0))))
            lo = rng.randrange(0, n - width + 1)
            hi = lo + width
            cand[lo:hi] = reversed(cand[lo:hi])
        elif move < 0.78:
            # Segment shuffle.
            width = max(1, min(n, 1 + int(rng.expovariate(1.0 / 3.0))))
            lo = rng.randrange(0, n - width + 1)
            hi = lo + width
            chunk = cand[lo:hi]
            rng.shuffle(chunk)
            cand[lo:hi] = chunk
        elif move < 0.90 and len(cand) > min_len:
            # Deletion (enables discovery of shorter networks).
            del cand[rng.randrange(len(cand))]
        elif move < 0.98 and len(cand) < max_len:
            # Insertion.
            cand.insert(rng.randrange(len(cand) + 1), random_comp(rng))
        else:
            # Burst rewrite mutation.
            rewrites = max(1, min(len(cand), 1 + int(rng.expovariate(1.0 / 4.0))))
            for _ in range(rewrites):
                cand[rng.randrange(n)] = random_comp(rng)

    return _clamp_length(cand, rng, min_len=min_len, max_len=max_len)


def crossover(parent_a: Network, parent_b: Network, rng: random.Random) -> Network:
    if not parent_a:
        return list(parent_b)
    if not parent_b:
        return list(parent_a)

    mix = rng.random()
    if mix < 0.34:
        # One-point crossover.
        cut_a = rng.randrange(len(parent_a) + 1)
        cut_b = rng.randrange(len(parent_b) + 1)
        return list(parent_a[:cut_a] + parent_b[cut_b:])
    if mix < 0.67:
        # Two-segment splice.
        cut1_a = rng.randrange(len(parent_a) + 1)
        cut2_a = rng.randrange(cut1_a, len(parent_a) + 1)
        cut1_b = rng.randrange(len(parent_b) + 1)
        cut2_b = rng.randrange(cut1_b, len(parent_b) + 1)
        return list(parent_a[:cut1_a] + parent_b[cut1_b:cut2_b] + parent_a[cut2_a:])

    # Uniform crossover with order perturbation.
    n = max(len(parent_a), len(parent_b))
    child: Network = []
    for i in range(n):
        pick_a = rng.random() < 0.5
        gene = None
        if pick_a and i < len(parent_a):
            gene = parent_a[i]
        elif (not pick_a) and i < len(parent_b):
            gene = parent_b[i]
        elif i < len(parent_a):
            gene = parent_a[i]
        elif i < len(parent_b):
            gene = parent_b[i]
        if gene is not None:
            child.append(gene)
    if child and rng.random() < 0.3:
        a = rng.randrange(len(child))
        b = rng.randrange(len(child))
        child[a], child[b] = child[b], child[a]
    return child


def make_candidate_from_seed(
    rng: random.Random,
    base: Network,
    min_len: int,
    max_len: int,
    persisted: Network | None = None,
    anchor: Network | None = None,
    heavy: bool = False,
) -> Network:
    source_roll = rng.random()
    if persisted and source_roll < 0.35:
        seed_net = list(persisted)
    elif anchor and source_roll < 0.75:
        seed_net = list(anchor)
    else:
        seed_net = list(base)

    target_len = min_len + rng.randrange(max_len - min_len + 1)
    seed_net = _clamp_length(seed_net, rng, min_len=target_len, max_len=target_len)
    if rng.random() < 0.35:
        rng.shuffle(seed_net)

    if heavy:
        steps = 3 + int(rng.expovariate(1.0 / 7.0))
    else:
        steps = 1 + int(rng.expovariate(1.0 / 3.0))
    return mutate_network(
        seed_net,
        rng,
        min_len=min_len,
        max_len=max_len,
        mutation_steps=steps,
    )


def tournament_select(
    population: Sequence[Network],
    fitness: Sequence[Tuple[int, int]],
    rng: random.Random,
    tournament_size: int,
) -> Network:
    best_idx = None
    for _ in range(max(2, tournament_size)):
        idx = rng.randrange(len(population))
        if best_idx is None or fitness[idx] < fitness[best_idx]:
            best_idx = idx
    return list(population[best_idx])


def select_parent(
    population: Sequence[Network],
    fitness: Sequence[Tuple[int, int]],
    ranked_indices: Sequence[int],
    rng: random.Random,
    stagnation_gens: int,
) -> Network:
    roll = rng.random()
    if roll < 0.7:
        # Lower tournament pressure when stale to preserve diversity.
        if stagnation_gens < 300:
            tsize = 5
        elif stagnation_gens < 1200:
            tsize = 4
        elif stagnation_gens < 3000:
            tsize = 3
        else:
            tsize = 2
        return tournament_select(population, fitness, rng, tournament_size=tsize)
    if roll < 0.9:
        top_k = max(4, len(population) // 4)
        idx = ranked_indices[rng.randrange(top_k)]
        return list(population[idx])
    return list(population[rng.randrange(len(population))])


def run_genetic_search(
    seed: int,
    population_size: int,
    generations: int,
    heartbeat_seconds: float,
    max_seconds: float,
    state_file: Path,
) -> Network | None:
    rng = random.Random(seed)
    base = batcher16()
    assert len(base) == 63 and fail_count(base) == 0

    min_len = 60
    max_len = 62

    start_time = time.time()
    last_heartbeat = start_time
    eval_cache: Dict[Tuple[Comparator, ...], int] = {}
    evals = 0

    def score(net: Network) -> int:
        nonlocal evals
        key = tuple(net)
        if key not in eval_cache:
            eval_cache[key] = fail_count(net)
            evals += 1
        return eval_cache[key]

    def rank_tuple(net: Network) -> Tuple[int, int]:
        # Primary objective: failures. Secondary objective: shorter network.
        return (score(net), len(net))

    persisted = load_state_network(state_file)
    persisted_seed: Network | None = None
    if persisted:
        persisted_seed = _clamp_length(persisted, rng, min_len=min_len, max_len=max_len)
        print(
            f"[state] loaded candidate len={len(persisted_seed)} fail={score(persisted_seed)} from '{state_file}'",
            flush=True,
        )
    else:
        print(f"[state] no prior state at '{state_file}'", flush=True)

    population: List[Network] = []
    if persisted_seed:
        population.append(list(persisted_seed))
    while len(population) < population_size:
        population.append(
            make_candidate_from_seed(
                rng,
                base=base,
                min_len=min_len,
                max_len=max_len,
                persisted=persisted_seed,
                anchor=None,
                heavy=False,
            )
        )

    best = min(population, key=rank_tuple)
    best_fail = score(best)
    persisted_fail = 1 << 60
    persisted_len = 1 << 60
    stagnation_gens = 0

    def maybe_persist(candidate: Network, candidate_fail: int, generation: int) -> None:
        nonlocal persisted_fail, persisted_len, persisted_seed
        if is_better(candidate_fail, len(candidate), persisted_fail, persisted_len):
            save_state_network(
                state_file=state_file,
                network=candidate,
                fail=candidate_fail,
                generation=generation,
                evals=evals,
            )
            persisted_fail = candidate_fail
            persisted_len = len(candidate)
            persisted_seed = list(candidate)
            print(
                f"[state] saved best fail={candidate_fail} len={len(candidate)} to '{state_file}'",
                flush=True,
            )

    print(
        "[start] target=<63 comparators, genetic algorithm search",
        flush=True,
    )
    print(
        f"[init] population={population_size} best_fail={best_fail} best_len={len(best)}",
        flush=True,
    )
    maybe_persist(best, best_fail, generation=0)

    for gen in range(1, generations + 1):
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"[stop] max_seconds reached ({elapsed:.1f}s)", flush=True)
            break

        fitness = [rank_tuple(net) for net in population]
        ranked_indices = sorted(range(len(population)), key=lambda i: fitness[i])
        # Decay elitism when stale; too much elitism causes collapse.
        if stagnation_gens < 400:
            elite_count = max(2, population_size // 10)
        elif stagnation_gens < 2000:
            elite_count = max(2, population_size // 14)
        else:
            elite_count = max(2, population_size // 20)
        next_population: List[Network] = [
            list(population[idx]) for idx in ranked_indices[:elite_count]
        ]

        current_best = population[ranked_indices[0]]
        current_fail = score(current_best)
        if (current_fail, len(current_best)) < (best_fail, len(best)):
            best = list(current_best)
            best_fail = current_fail
            print(
                f"[improve] gen={gen} best_fail={best_fail} best_len={len(best)} "
                f"elapsed={elapsed:.1f}s evals={evals}",
                flush=True,
            )
            maybe_persist(best, best_fail, generation=gen)
            stagnation_gens = 0
            if best_fail == 0 and len(best) < 63:
                print(f"[found] comparators={len(best)} fail=0 gen={gen}", flush=True)
                return best
        else:
            stagnation_gens += 1

        # Periodic diversity reset if fully stalled.
        if stagnation_gens > 5000:
            print(
                f"[diversify] gen={gen} no improvement for {stagnation_gens} generations; "
                "injecting heavy-randomized population refresh",
                flush=True,
            )
            keep_count = max(2, population_size // 25)
            survivors: List[Network] = [list(best)]
            if persisted_seed:
                survivors.append(list(persisted_seed))
            for idx in ranked_indices[1 : 1 + keep_count]:
                survivors.append(list(population[idx]))
            population = survivors[:]
            while len(population) < population_size:
                population.append(
                    make_candidate_from_seed(
                        rng,
                        base=base,
                        min_len=min_len,
                        max_len=max_len,
                        persisted=persisted_seed,
                        anchor=best,
                        heavy=True,
                    )
                )
            next_population = population[: max(2, population_size // 20)]
            stagnation_gens = 1200

        # Higher immigrant share as stagnation grows.
        immigrant_fraction = min(0.5, 0.06 + 0.00005 * stagnation_gens)
        while len(next_population) < population_size:
            if rng.random() < immigrant_fraction:
                child = make_candidate_from_seed(
                    rng,
                    base=base,
                    min_len=min_len,
                    max_len=max_len,
                    persisted=persisted_seed,
                    anchor=best,
                    heavy=stagnation_gens > 1500,
                )
            else:
                p1 = select_parent(population, fitness, ranked_indices, rng, stagnation_gens)
                p2 = select_parent(population, fitness, ranked_indices, rng, stagnation_gens)
                child = crossover(p1, p2, rng)
                child = _clamp_length(child, rng, min_len=min_len, max_len=max_len)
                mutation_steps = 1 + int(rng.expovariate(1.0 / 2.5))
                mutation_steps += min(10, stagnation_gens // 500)
                if rng.random() < min(0.3, stagnation_gens / 8000.0):
                    mutation_steps += 2 + int(rng.expovariate(1.0 / 3.0))
                child = mutate_network(
                    child,
                    rng,
                    min_len=min_len,
                    max_len=max_len,
                    mutation_steps=mutation_steps,
                )
            next_population.append(child)

        population = next_population

        now = time.time()
        if now - last_heartbeat >= heartbeat_seconds:
            unique = len({tuple(net) for net in population})
            unique_ratio = unique / max(1, population_size)
            print(
                f"[heartbeat] gen={gen}/{generations} best_fail={best_fail} "
                f"best_len={len(best)} unique={unique}/{population_size} "
                f"stagnation={stagnation_gens} immigrant_frac={immigrant_fraction:.2f} "
                f"evals={evals} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if unique_ratio < 0.35:
                print(
                    f"[warn] low diversity unique_ratio={unique_ratio:.2f}; consider larger population",
                    flush=True,
                )
            last_heartbeat = now

        # Keep cache bounded for very long runs.
        if len(eval_cache) > 600000:
            keep_keys = {tuple(best)}
            if persisted_seed:
                keep_keys.add(tuple(persisted_seed))
            for net in population[: min(250, len(population))]:
                keep_keys.add(tuple(net))
            eval_cache = {k: v for k, v in eval_cache.items() if k in keep_keys}

    if best_fail == 0 and len(best) < 63:
        print(f"[found] comparators={len(best)} fail=0", flush=True)
        return best

    maybe_persist(best, best_fail, generation=generations)
    print(
        f"[done] best_fail={best_fail} best_len={len(best)} evals={evals} "
        f"elapsed={time.time() - start_time:.1f}s",
        flush=True,
    )
    return None


def resolve_seed(cli_seed: int | None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    # Mix date/time with process info to vary seeds across runs.
    dt = datetime.now().isoformat(timespec="microseconds")
    entropy = f"{dt}|{time.time_ns()}|{os.getpid()}|{os.urandom(8).hex()}".encode("utf-8")
    digest = hashlib.blake2b(entropy, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--heartbeat-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--state-file", type=str, default="ga_best_candidate.json")
    args = parser.parse_args()

    seed = resolve_seed(args.seed)
    if args.seed is None:
        print(f"[seed] auto-generated seed={seed}", flush=True)
    else:
        print(f"[seed] using provided seed={seed}", flush=True)

    result = run_genetic_search(
        seed=seed,
        population_size=max(20, args.population_size),
        generations=max(1, args.generations),
        heartbeat_seconds=args.heartbeat_seconds,
        max_seconds=args.max_seconds,
        state_file=Path(args.state_file),
    )
    if result is None:
        print("NOT_FOUND", flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
