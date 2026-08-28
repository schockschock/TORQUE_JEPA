# 02: Load Canonical State Forecasting Samples

**What to build:** Turn pilot trajectories into validated state-only Forecasting Samples using canonical 16-observation/30-target timing and Observation-End Frame semantics. Build the dataset class and the Dataloader to do so.
It must be simple.

**Blocked by:** 01: Freeze Versioned Pilot Trajectories.

**Status:** ready-for-agent

- [ ] Samples use stride-two 120-to-60 Hz indexing with exact 0.25 s observation and 0.50 s forecast horizons.
- [ ] Observed and target tensors follow documented COM, rotation, linear-velocity, and actor angular-velocity conventions.
- [ ] Observation-End World Location and rotation reconstruct original world trajectories within numerical tolerance.
- [ ] Camera A sampled-frame visibility, complete-future, train-only normalization, and invalid-window policies are enforced and tested.
- [ ] Window indexing is deterministic and supports at most one sampled window per trajectory in a training batch.
