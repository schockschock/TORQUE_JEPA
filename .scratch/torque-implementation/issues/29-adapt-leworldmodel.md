# 29: Adapt LeWorldModel

**What to build:** Adapt LeWorldModel-style latent prediction to TORQUE windows and compare its predictive representations through declared forecast protocol.

**Blocked by:** 18: Run Locked Final Main Benchmark.

**Status:** ready-for-agent

- [ ] Adaptation documents target, context, teacher-forcing, autoregressive inference, and SIGReg differences from TORQUE-JEPA.
- [ ] Input modalities and future targets obey one declared TORQUE information track.
- [ ] Forecast decoding/probing protocol is explicit and does not silently fine-tune frozen representations.
- [ ] Results use same final snapshot, metric pipeline, and object-macro aggregation.
