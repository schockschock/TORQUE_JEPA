# 07: Establish State Baseline Ladder

**What to build:** Benchmark simpler supervised state models under same Forecasting Sample, rollout, normalization, and evaluation contract as TORQUE-Sup.

**Blocked by:** 05: Train State-Only TORQUE-Sup.

**Status:** ready-for-agent

- [ ] MLP, recurrent, direct-Transformer, and autoregressive-Transformer baselines are trainable from same snapshot and splits.
- [ ] Every baseline declares inputs and emits all four target state families.
- [ ] Search budget and checkpoint selection are documented and comparable.
- [ ] Report compares models against controls and TORQUE-Sup with identical metrics and object-macro aggregation.
