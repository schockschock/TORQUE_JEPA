# 05: Train State-Only TORQUE-Sup

**What to build:** Deliver first learned freefall forecast using state-only TORQUE-Sup from canonical samples through training, checkpoint selection, 30-step rollout, and control comparison.

**Blocked by:** 03: Evaluate Persistence And Constant-Velocity Controls.

**Status:** ready-for-agent

- [ ] Model consumes only declared state Observable Context plus Observation-End World Location and sequence-index encoding.
- [ ] Training implements component-aware equal-family loss and reproducible teacher-forcing/free-running curriculum.
- [ ] Horizon-one feedback uses observation-end state converted to target coordinates.
- [ ] Checkpoint selection uses object-macro validation and never test objects.
- [ ] Evaluation reports reconstructed-world curves against physical controls on frozen pilot snapshot.
