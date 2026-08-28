# 10: Train RGB-D TORQUE-Sup

**What to build:** Add metric Camera A depth to state+RGB TORQUE-Sup with explicit invalid-depth handling and separate depth representation.

**Blocked by:** 09: Train RGB TORQUE-Sup Tracks.

**Status:** ready-for-agent

- [ ] Depth preprocessing preserves metric finite values, supplies validity mask, and uses training-object normalization.
- [ ] Separate scratch depth encoder produces pooled token per observed timestep.
- [ ] Same-timestep state-to-RGB/depth fusion respects availability masks.
- [ ] Report isolates depth contribution over selected state+RGB track using matched core and metrics.
