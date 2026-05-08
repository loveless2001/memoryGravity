# Previous Experiment Artifacts

This index records the older Memory Gravity experiment branch that predated the
geometric-commitment paper work. It is intentionally separate from the Modal
Pythia/larger-model artifacts used by `docs/geometric_commitment_signatures_paper.md`.

## Result Groups

- `results/mg_core/`: early TinyShakespeare and ablation checks for refined MG
  attention dynamics.
- `results/mg_recall_tasks/`: delayed-recall and associative-recall probes.
- `results/glyph_memory_arena/`: glyph binding, query-gated local attention,
  selective-reach, and v2 activation-boundary experiments.

## Main Code Paths

- `train/mg_core.py`: core Memory Gravity language-model blocks used by the toy
  and glyph-memory arenas.
- `train/glyph_memory_data.py`: synthetic glyph binding and selective-reach data.
- `train/run_glyph_memory_arena.py`: general arena runner.
- `train/run-v2-reduced-overnight-with-checkpoints.py`: single-seed reduced v2
  checkpointed run.
- `train/run-v2-capacity-map-3seed-and-bridge.py`: 3-seed capacity-map follow-up.
- `scripts/phase-a-optimization-sweep-v2-activation-reliability.sh`: paused
  stabilization sweep script.

## Current Interpretation

The most complete writeup is `docs/memory_gravity_phase_transition.md`. The
defensible claim is that query-gated Memory Gravity creates a real beyond-window
retrieval route, but the hard `seq_len=512` selective-reach regime is activation
fragile: one bottlenecked seed solved it, while follow-up seeds did not.
