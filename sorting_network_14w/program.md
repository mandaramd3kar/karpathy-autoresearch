# sorting_network_14w autoresearch program

This experiment lives under `sorting_network_14w/` and uses Karpathy's
autoresearch style on the 14-wire sorting-network problem without touching his
original Python files.

## Goal

Find a correct 14-wire sorting network with fewer than `51` comparators.

Priority order:

1. Correctness on all `2^14` binary inputs.
2. Comparator count below `51` if possible (target `50`).
3. Lower depth as the tie-breaker.

## In-scope files

Read these before starting:

- `README.md`
- `program.md`
- `minimal_sorting_networks/README.md`
- `minimal_sorting_networks/evaluator.py`
- `minimal_sorting_networks/_search_under63_lbh_graph.py`
- `minimal_sorting_networks/ga_best_candidate.json`
- `sorting_network_14w/experiment.py`
- `sorting_network_14w/analysis.json`

## The key insight

The current saved 16-wire JSON candidate is not an arbitrary near miss. It is
the canonical 63-comparator Batcher network with exactly one comparator missing:
`(1, 2)`. Every failure comes from that single unresolved boundary. That means
the old search is stuck in a rigid power-of-two merge family.

For 14 wires, avoid doing more of the same. The experiment file already does
three deliberate things:

1. Projects a correct 53-comparator 14-wire seed out of the 16-wire family.
2. Compresses toward shorter targets with a deletion beam.
3. Repairs low-failure candidates using hotspot-guided local search.

## Hard constraints

- Do not modify `prepare.py`.
- Do not modify `train.py`.
- Do not modify the Python files under `minimal_sorting_networks/`.
- Only edit `sorting_network_14w/experiment.py` unless a non-Python helper file
  is clearly needed for logging or reporting.
- Keep the experiment runnable with the single command below.

## Baseline command

```bash
python sorting_network_14w/experiment.py --max-seconds 60 --goal-length 50
```

If you want an even shorter Windows command, use:

```bash
powershell -ExecutionPolicy Bypass -File .\sorting_network_14w\run.ps1 --max-seconds 60 --goal-length 50
```

If you specifically want to run it through `uv`, first point `uv` at a writable local cache and sync:

```bash
$env:UV_CACHE_DIR = ".uv-cache"
python -m uv sync
python -m uv run sorting-network-14w --max-seconds 60 --goal-length 50
```

## Logging

Keep a local TSV called `sorting_network_14w/results.tsv` with:

```text
commit	target_len	fail_count	depth	status	description
```

Use `keep` for improvements, `discard` for regressions, and `breakthrough` for a
correct network below 51 comparators.

## Experiment loop

1. Read the current git state.
2. Run the baseline experiment once and record its best result per target length.
3. Edit only `sorting_network_14w/experiment.py`.
4. Commit the change.
5. Run:

```bash
python sorting_network_14w/experiment.py --max-seconds 120 --goal-length 50 > sorting_network_14w/sorting14.log 2>&1
```

6. Inspect the summary and the updated `sorting_network_14w/state.json`.
7. If the best `fail_count` at target length `50` improves, keep the commit.
8. If not, revert only your own last experiment commit and try another idea.

## High-value ideas

- Run a descendant-aware search, not just a per-length search. Score every
  `52`-comparator candidate by its best exact `51` descendant, and every `51`
  candidate by its best exact `50` descendant. Optimize for compressibility,
  because a network that is slightly worse at `52` may be dramatically better
  after one or two exact deletions.
- Replace single-seed thinking with a large prefix portfolio. Start from many
  structurally distinct `3`- to `5`-layer prefixes mined from known optimal and
  near-optimal families, then grow and compress each family separately. Treat
  prefix diversity as a first-class search axis, not a side effect.
- Add SAT-guided local impossibility learning. For stubborn `52 -> 51` and
  `51 -> 50` transitions, encode the surviving prefix and ask a SAT subsolver
  whether a short suffix can repair it. If the answer is no, extract the failed
  local pattern as a hard taboo constraint for future search.
- Build a failure-hypergraph archive. Represent each candidate by the exact set
  of counterexamples and inversion boundaries it fails on, and preserve a
  quality-diversity frontier over those signatures. Stop keeping only the
  numerically best fail count; keep candidates that fail in genuinely different
  ways so crossover can combine complementary fixes.
- Use e-graph style rewrite saturation on short comparator windows. Canonicalize
  semantically equivalent subnets, mine recurring useful motifs, and search in a
  quotient space of behaviors rather than raw comparator sequences. This should
  expose compressible local structures that ordinary mutation never notices.
- Run meet-in-the-middle layer synthesis for the last few layers. Enumerate
  reachable partially ordered states from a fixed prefix and backwards from the
  sorted outputs, then solve the middle connection problem exactly. Use this to
  discover whether a candidate is blocked by the late suffix rather than the
  early scaffold.
- Replace generic mutation pressure with bridge-construction pressure derived
  from exact misplacement transport plans. For each failing case, compute where
  misplaced `1`s and `0`s must cross, aggregate those crossings, and propose
  comparators that explicitly move mass across those bottlenecks, including
  spans that never appear in Batcher-style families.
- Add learned guidance instead of hand-tuned heuristics alone. Train a small
  online surrogate on evaluated candidates to predict exact fail count, best
  child fail count, and hotspot signature. Use it only for ranking and pruning,
  never as ground truth, so the exact evaluator remains the authority.
- Introduce branch-and-bound over the final comparator budget. When a candidate
  reaches `52` or `51` with low fail count, launch an exact bounded search over
  the remaining delete/insert/swap decisions under a strict wall-clock cap.
  Treat this as a tactical solver pass layered on top of the stochastic search.
- Search for reusable motifs, not only full networks. Mine short comparator
  blocks that repeatedly appear in the best `52`, `51`, and `50` near-misses,
  cluster them by semantic effect, and let the search splice motif-level units
  instead of raw comparators. The goal is to evolve a library of compressible
  building blocks that can break the current basin.

## Stop condition

Keep going until interrupted by the human or until you find a correct
`50`-comparator network.
