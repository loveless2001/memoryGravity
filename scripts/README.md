# Scripts

This directory contains long-running or operational wrappers that are not the
primary Python entrypoints.

- `phase-a-optimization-sweep-v2-activation-reliability.sh`: stabilization sweep
  for the reduced v2 selective-reach activation boundary. It writes to
  `results/glyph_memory_arena/v2_optim_sweep/`.
- `kill-sweep-after-run2.sh`: local one-off process watcher for pausing the
  above sweep after run 2. The embedded PID is historical and should be updated
  before reuse.
