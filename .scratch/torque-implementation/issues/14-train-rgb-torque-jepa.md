# 14: Train RGB TORQUE-JEPA

**What to build:** Add Camera A RGB context to TORQUE-JEPA and test whether visual predictive representations improve frozen-probe forecasting over state-only JEPA and matched supervised RGB models.

**Blocked by:** 09: Train RGB TORQUE-Sup Tracks; 13: Train State-Only TORQUE-JEPA.

**Status:** ready-for-agent

- [ ] Selected frozen/scratch RGB encoder follows established RGB track and same-timestep fusion contract.
- [ ] JEPA objective remains state-grounded with no future visual target or decoded-state loss.
- [ ] Backbone and visual policy freeze exactly as declared before Forecast Probe training.
- [ ] Report includes modality-matched TORQUE-Sup and state-only JEPA comparisons.
