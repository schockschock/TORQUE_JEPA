# 15: Train RGB-D TORQUE-JEPA

**What to build:** Add metric depth context to RGB TORQUE-JEPA while preserving representation-first objective and frozen-probe evaluation.

**Blocked by:** 10: Train RGB-D TORQUE-Sup; 14: Train RGB TORQUE-JEPA.

**Status:** ready-for-agent

- [ ] RGB-D preprocessing, validity masks, encoders, and fusion match supervised RGB-D information contract.
- [ ] Missing/invalid depth cannot leak segmentation or simulator metadata.
- [ ] State-grounded JEPA/SIGReg and rollout remain unchanged except added Observable Context.
- [ ] Report isolates depth contribution against state+RGB JEPA and matched TORQUE-Sup.
