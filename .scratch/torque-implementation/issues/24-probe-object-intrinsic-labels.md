# 24: Probe Object-Intrinsic Labels

**What to build:** Measure object-disjoint accessibility of mass, volume, density, principal inertia, and COM offset from frozen context and predicted representations.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Linear and `256 -> 256 -> target` probes train without backbone updates on same object split.
- [ ] Target normalization uses training objects only and preserves documented physical units for inverse transform.
- [ ] Window predictions aggregate within trajectory and object before object-level MAE and R2 reporting.
- [ ] Context-summary and per-horizon predicted-representation sources are reported separately.
- [ ] Conclusions state geometry/physics confounding and avoid causal parameter-identification claims.
