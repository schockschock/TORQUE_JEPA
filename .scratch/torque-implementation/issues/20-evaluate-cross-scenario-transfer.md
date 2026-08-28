# 20: Evaluate Cross-Scenario Transfer

**What to build:** Run leave-one-scenario-out transfer for locked models using same object population and explicit multimodal viewpoint-shift interpretation.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Four held-out-scenario folds use same object population and no scenario token.
- [ ] Each transfer direction is reported separately rather than hidden behind one average.
- [ ] State-only results are distinguished from multimodal results that also encounter Camera A viewpoint shift.
- [ ] Checkpoint selection and normalization use only represented training scenarios and validation objects.
