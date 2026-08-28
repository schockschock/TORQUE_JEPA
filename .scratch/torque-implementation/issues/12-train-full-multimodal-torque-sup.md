# 12: Train Full Multimodal TORQUE-Sup

**What to build:** Deliver state+RGB-D+geometry TORQUE-Sup using canonical hierarchical causal fusion from complete Observable Context to 30-step kinematic-state forecast.

**Blocked by:** 10: Train RGB-D TORQUE-Sup; 11: Train Geometry TORQUE-Sup.

**Status:** ready-for-agent

- [ ] Dynamic state/RGB/depth tokens and persistent geometry token follow documented fusion order.
- [ ] Model excludes scenario labels, calibration, contact, segmentation, physical parameters, and explicit environment representation.
- [ ] Training and evaluation remain matched to state-only TORQUE-Sup core and rollout contract.
- [ ] Report includes all cumulative modality tracks plus state+geometry diagnostic.
