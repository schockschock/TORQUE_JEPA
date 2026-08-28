# 17: Train Full Multimodal TORQUE-JEPA

**What to build:** Deliver complete state+RGB-D+geometry TORQUE-JEPA from multimodal Observable Context through latent rollout and frozen Forecast Probes.

**Blocked by:** 12: Train Full Multimodal TORQUE-Sup; 15: Train RGB-D TORQUE-JEPA; 16: Train Geometry TORQUE-JEPA.

**Status:** ready-for-agent

- [ ] Full model uses canonical 256-D hierarchical fusion core and 32-D recursive objective latent.
- [ ] Target remains future state only; no physical, contact, future-visual, consistency, or mechanics loss enters initial method.
- [ ] Three cumulative JEPA modality paths and state+geometry diagnostic are reproducible from same configuration family.
- [ ] Frozen-probe report compares all modality-matched TORQUE-Sup models.
