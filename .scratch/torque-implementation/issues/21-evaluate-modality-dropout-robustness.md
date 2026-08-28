# 21: Evaluate Modality-Dropout Robustness

**What to build:** Train secondary universal multimodal model and measure graceful degradation under declared RGB, depth, and geometry availability patterns.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Training dropout operates only on optional observable modalities and preserves state input.
- [ ] Availability masks distinguish state, RGB, depth, and static geometry semantics correctly.
- [ ] Evaluation includes complete input and each planned missing-modality pattern.
- [ ] Results remain secondary to separately trained modality-track models.
