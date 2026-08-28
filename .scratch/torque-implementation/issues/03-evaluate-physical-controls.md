# 03: Evaluate Persistence And Constant-Velocity Controls

**What to build:** Forecast pilot windows with persistence and constant world linear/angular velocity, then evaluate them through same world-reconstruction and metric path used by learned methods.

**Blocked by:** 02: Load Canonical State Forecasting Samples.

**Status:** ready-for-agent

- [ ] Persistence emits all four kinematic-state families for every future horizon.
- [ ] Constant-velocity control integrates world position and orientation from observation-end world linear/angular velocity.
- [ ] Reports include position, orientation, linear-velocity, and angular-velocity horizon curves plus ADE/FDE-style summaries.
- [ ] Reduction follows window, trajectory, scenario, object hierarchy with object-bootstrap uncertainty.
- [ ] Control outputs and metrics are reproducible from frozen pilot snapshot.
