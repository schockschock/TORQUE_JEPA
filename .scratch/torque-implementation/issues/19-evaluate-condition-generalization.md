# 19: Evaluate Condition Generalization

**What to build:** Evaluate locked models under separate Initial-Condition-Disjoint and Scenario-Condition-Disjoint protocols without changing object or scenario axes simultaneously.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Initial-condition protocol holds out joint combinations while retaining each marginal value in training.
- [ ] Scenario-condition protocols report interpolation and extrapolation separately for applicable controls.
- [ ] Protocol artifacts reference stable initial/scenario condition identities from final manifest.
- [ ] Models are selected without held-out condition test leakage and results use same object-macro metric contract.
