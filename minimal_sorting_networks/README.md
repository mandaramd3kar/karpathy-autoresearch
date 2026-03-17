# Minimal Sorting Networks (n=16)

This example sets up OpenEvolve for a hard combinatorial optimization target:
discovering shorter, correct sorting networks on 16 wires.

## Problem

A sorting network is a fixed sequence of comparators `(i, j)` that must sort
every input, independent of data values. By the zero-one principle, correctness
for all real inputs is equivalent to correctness for all `2^n` binary inputs.

For `n=16`, the known comparator optimum is `60`. This example optimizes:

1. Correctness on all `2^16` binary inputs (must pass first)
2. Comparator count (smaller is better)
3. Depth (tie-breaker)

## Files

- `initial_program.py`: Baseline network search algorithm (single EVOLVE-BLOCK)
- `evaluator.py`: Exact bitset-based zero-one evaluator for `n=16`
- `config.yaml`: OpenEvolve configuration tuned for this task

## Why the evaluator is fast

The evaluator encodes each wire as a Python integer bitset over all `65,536`
binary cases, applies each comparator with bitwise `AND/OR`, and computes exact
failures in one pass. That keeps full exhaustive correctness checks practical.

## Run

```bash
cd examples/minimal_sorting_networks
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 150
```

## Quick local check

```bash
cd examples/minimal_sorting_networks
python evaluator.py initial_program.py
```
