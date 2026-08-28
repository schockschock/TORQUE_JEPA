# 30: Adapt DINO-WM

**What to build:** Adapt DINO-WM pretrained visual dynamics to TORQUE forecasting and representation evaluation.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Adaptation defines how DINO patch/frame features map into TORQUE causal context and future prediction.
- [ ] Observable inputs, target representation, and frozen/trainable components are declared.
- [ ] Forecast outputs and probes evaluate all four kinematic-state families.
- [ ] Results report compute and compare against TORQUE-JEPA pretrained-visual track under same snapshot and metrics.
