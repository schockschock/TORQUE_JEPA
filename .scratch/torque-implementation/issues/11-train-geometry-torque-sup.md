# 11: Train Geometry TORQUE-Sup

**What to build:** Add reproducibly audited Utonia global geometry summaries to state-only TORQUE-Sup through persistent static conditioning.

**Blocked by:** 06: Run Ramp And Joint TORQUE-Sup Pilots.

**Status:** ready-for-agent

- [ ] Geometry loader verifies artifact identity, extraction provenance, expected 1,386-D summaries, and object mapping.
- [ ] Global mean/max summaries use training-object standardization and explicit availability state.
- [ ] Gated geometry cross-attention conditions dynamic tokens without physical parameters.
- [ ] Report compares state+geometry against state-only TORQUE-Sup with matched core and object-macro metrics.
