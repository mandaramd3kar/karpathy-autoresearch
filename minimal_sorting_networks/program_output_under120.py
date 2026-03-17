"""Deterministic sorting-network candidate with <120 comparators for n=16."""

from typing import List, Tuple


Comparator = Tuple[int, int]


def _odd_even_merge(lo: int, n: int, r: int, out: List[Comparator]) -> None:
    step = r * 2
    if step < n:
        _odd_even_merge(lo, n, step, out)
        _odd_even_merge(lo + r, n, step, out)
        for i in range(lo + r, lo + n - r, step):
            out.append((i, i + r))
    else:
        out.append((lo, lo + r))


def _odd_even_merge_sort(lo: int, n: int, out: List[Comparator]) -> None:
    if n > 1:
        m = n // 2
        _odd_even_merge_sort(lo, m, out)
        _odd_even_merge_sort(lo + m, m, out)
        _odd_even_merge(lo, n, 1, out)


def build_network(num_wires: int = 16) -> List[Comparator]:
    if num_wires <= 0 or (num_wires & (num_wires - 1)) != 0:
        raise ValueError("Batcher odd-even mergesort network requires num_wires to be a power of 2")
    network: List[Comparator] = []
    _odd_even_merge_sort(0, num_wires, network)
    return network


def solve(num_wires: int = 16, seed: int = 0) -> List[Comparator]:
    del seed  # Deterministic construction.
    return build_network(num_wires=num_wires)


if __name__ == "__main__":
    net = solve(16)
    print(f"comparators={len(net)}")
    print(net[:20])
