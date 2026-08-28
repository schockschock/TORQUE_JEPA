# 06: Run Ramp And Joint TORQUE-Sup Pilots

**What to build:** Extend state-only TORQUE-Sup to ramp-specific and joint freefall/ramp engineering pilots without scenario labels as model inputs.

**Blocked by:** 05: Train State-Only TORQUE-Sup.

**Status:** ready-for-agent

- [ ] Separate freefall and ramp models train and evaluate under identical state contract.
- [ ] Joint scenario-agnostic model trains on both scenarios without scenario token.
- [ ] Report contains per-scenario and equally weighted joint object-macro metrics.
- [ ] Pilot outputs are marked engineering evidence, not publication benchmark claims.
