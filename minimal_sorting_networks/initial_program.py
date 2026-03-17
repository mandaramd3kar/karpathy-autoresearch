# EVOLVE-BLOCK-START
"""Initial program for evolving minimal sorting networks."""

import random


def search_network(num_wires: int = 16, iterations: int = 4000, seed: int = 0):
    """
    Start from a guaranteed-correct odd-even transposition network and
    stochastically delete/replace comparators while preserving correctness.
    """
    rng = random.Random(seed)
    base_wires = _build_wire_masks(num_wires)

    best_network = _odd_even_transposition_network(num_wires)
    best_failures = _count_failures(best_network, base_wires)

    for _ in range(max(1, iterations)):
        candidate = list(best_network)
        if not candidate:
            break

        move = rng.random()
        if move < 0.78 and len(candidate) > 1:
            # Aggressive pruning: remove one comparator.
            del candidate[rng.randrange(len(candidate))]
        elif move < 0.92:
            # Replace one comparator with a random valid pair.
            idx = rng.randrange(len(candidate))
            candidate[idx] = _random_comparator(num_wires, rng)
        else:
            # Small permutation move to improve ordering.
            a = rng.randrange(len(candidate))
            b = rng.randrange(len(candidate))
            candidate[a], candidate[b] = candidate[b], candidate[a]

        failures = _count_failures(candidate, base_wires)
        if failures < best_failures:
            best_failures = failures
            best_network = candidate
        elif failures == best_failures and len(candidate) < len(best_network):
            best_network = candidate

    return best_network


def _odd_even_transposition_network(num_wires: int):
    network = []
    for phase in range(num_wires):
        start = phase % 2
        for i in range(start, num_wires - 1, 2):
            network.append((i, i + 1))
    return network


def _random_comparator(num_wires: int, rng: random.Random):
    i = rng.randrange(num_wires)
    j = rng.randrange(num_wires - 1)
    if j >= i:
        j += 1
    return (j, i) if j < i else (i, j)


def _build_wire_masks(num_wires: int):
    total = 1 << num_wires
    wires = [0] * num_wires
    for case in range(total):
        for wire in range(num_wires):
            if (case >> (num_wires - 1 - wire)) & 1:
                wires[wire] |= 1 << case
    return tuple(wires)


def _count_failures(network, base_wires):
    wires = list(base_wires)
    for i, j in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b

    invalid = 0
    for i in range(len(wires) - 1):
        invalid |= wires[i] & ~wires[i + 1]
    return invalid.bit_count()


# EVOLVE-BLOCK-END


def _sanitize_network(network, num_wires: int):
    if not isinstance(network, (list, tuple)):
        return []

    sanitized = []
    for item in network:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            i = int(item[0])
            j = int(item[1])
        except (TypeError, ValueError):
            continue
        if i == j:
            continue
        if not (0 <= i < num_wires and 0 <= j < num_wires):
            continue
        if j < i:
            i, j = j, i
        sanitized.append((i, j))
    return sanitized


def solve(num_wires: int = 16, seed: int = 0):
    """Return a candidate sorting network as a list of (i, j) comparators."""
    network = search_network(num_wires=num_wires, seed=seed)
    return _sanitize_network(network, num_wires)


if __name__ == "__main__":
    net = solve(num_wires=16, seed=0)
    print(f"comparators={len(net)}")
    print(net[:10])
