# 23: Run Visual And Fusion Ablations

**What to build:** Test whether visual fine-tuning and alternative fusion improve forecasting beyond canonical frozen pooled-token hierarchical model.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Frozen visual baseline is compared with declared fine-tuning policy under matched search budget.
- [ ] Hierarchical causal fusion is compared with Perceiver-style alternative under declared parameter/FLOP differences.
- [ ] Any patch/local-token extension reports extra compute and uses same Observable Context.
- [ ] Results separate representation changes from increased capacity.
