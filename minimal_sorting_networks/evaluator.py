"""Evaluator for minimal sorting-network optimization."""

import concurrent.futures
import importlib.util
import traceback
from typing import Any, Iterable, List, Sequence, Tuple

from openevolve.evaluation_result import EvaluationResult

TARGET_WIRES = 16
TOTAL_CASES = 1 << TARGET_WIRES
TARGET_MIN_COMPARATORS = 60
TARGET_MIN_DEPTH = 10
MAX_COMPARATORS = 256
SOLVER_TIMEOUT_SECONDS = 20


def _build_wire_masks(num_wires: int) -> Tuple[int, ...]:
    wires = [0] * num_wires
    total = 1 << num_wires
    for case in range(total):
        for wire in range(num_wires):
            if (case >> (num_wires - 1 - wire)) & 1:
                wires[wire] |= 1 << case
    return tuple(wires)


BASE_WIRES = _build_wire_masks(TARGET_WIRES)


def _normalize_comparator(item: Sequence[Any], num_wires: int) -> Tuple[int, int] | None:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return None
    try:
        i = int(item[0])
        j = int(item[1])
    except (TypeError, ValueError):
        return None
    if i == j:
        return None
    if not (0 <= i < num_wires and 0 <= j < num_wires):
        return None
    if j < i:
        i, j = j, i
    return (i, j)


def _sanitize_network(raw: Any, num_wires: int) -> List[Tuple[int, int]]:
    if isinstance(raw, dict):
        for key in ("network", "comparators", "pairs"):
            if key in raw:
                raw = raw[key]
                break

    if not isinstance(raw, (list, tuple)):
        return []

    network: List[Tuple[int, int]] = []
    for item in raw:
        comp = _normalize_comparator(item, num_wires)
        if comp is not None:
            network.append(comp)
        if len(network) >= MAX_COMPARATORS:
            break
    return network


def _count_failures(network: Iterable[Tuple[int, int]], base_wires: Tuple[int, ...]) -> int:
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


def _network_depth(network: Sequence[Tuple[int, int]], num_wires: int) -> int:
    levels: List[set[int]] = []
    for i, j in network:
        placed = False
        for used_wires in levels:
            if i not in used_wires and j not in used_wires:
                used_wires.add(i)
                used_wires.add(j)
                placed = True
                break
        if not placed:
            levels.append({i, j})
    return len(levels)


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate_program", program_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _call_solver(program) -> Any:
    if hasattr(program, "solve"):
        fn = program.solve
    elif hasattr(program, "search_network"):
        fn = program.search_network
    else:
        raise AttributeError("Program is missing required solver (solve or search_network)")

    call_patterns = (
        {"num_wires": TARGET_WIRES, "seed": 0},
        {"n": TARGET_WIRES, "seed": 0},
        {"num_wires": TARGET_WIRES},
        {"n": TARGET_WIRES},
        {},
    )
    for kwargs in call_patterns:
        try:
            return fn(**kwargs)
        except TypeError:
            continue
    return fn()


def _run_solver_with_timeout(program):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_solver, program)
        return future.result(timeout=SOLVER_TIMEOUT_SECONDS)


def evaluate(program_path: str) -> EvaluationResult:
    try:
        program = _load_program(program_path)
        raw_network = _run_solver_with_timeout(program)
        network = _sanitize_network(raw_network, TARGET_WIRES)

        if not network:
            return EvaluationResult(
                metrics={
                    "combined_score": 0.0,
                    "correctness": 0.0,
                    "comparator_count": float(MAX_COMPARATORS),
                    "depth": float(MAX_COMPARATORS),
                    "fail_count": float(TOTAL_CASES),
                },
                artifacts={"error": "Solver returned an empty or invalid network"},
            )

        fail_count = _count_failures(network, BASE_WIRES)
        correctness = 1.0 - (fail_count / TOTAL_CASES)
        comparator_count = len(network)
        depth = _network_depth(network, TARGET_WIRES)

        if fail_count > 0:
            # Lexicographic preference: correctness dominates all else.
            combined_score = correctness**6
        else:
            comparator_efficiency = min(
                1.0, TARGET_MIN_COMPARATORS / max(comparator_count, 1)
            )
            depth_efficiency = min(1.0, TARGET_MIN_DEPTH / max(depth, 1))
            length_penalty = max(
                0.0, (comparator_count - TARGET_MIN_COMPARATORS) / TARGET_MIN_COMPARATORS
            )
            combined_score = 1.0 + 0.85 * comparator_efficiency + 0.15 * depth_efficiency
            combined_score -= 0.05 * length_penalty

        return EvaluationResult(
            metrics={
                "combined_score": float(combined_score),
                "correctness": float(correctness),
                "fail_count": float(fail_count),
                "comparator_count": float(comparator_count),
                "depth": float(depth),
            },
            artifacts={
                "first_comparators": network[:20],
                "target_wires": TARGET_WIRES,
                "target_min_comparators": TARGET_MIN_COMPARATORS,
                "target_min_depth": TARGET_MIN_DEPTH,
            },
        )
    except concurrent.futures.TimeoutError:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "correctness": 0.0,
                "comparator_count": float(MAX_COMPARATORS),
                "depth": float(MAX_COMPARATORS),
                "fail_count": float(TOTAL_CASES),
            },
            artifacts={"error": f"Solver timeout after {SOLVER_TIMEOUT_SECONDS} seconds"},
        )
    except Exception as exc:
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "correctness": 0.0,
                "comparator_count": float(MAX_COMPARATORS),
                "depth": float(MAX_COMPARATORS),
                "fail_count": float(TOTAL_CASES),
            },
            artifacts={
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    import sys

    result = evaluate(sys.argv[1])
    print(result.metrics)
