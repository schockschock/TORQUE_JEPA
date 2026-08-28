# 08: Complete And Freeze Final Dataset Snapshot

**What to build:** Produce final immutable TORQUE Dataset snapshot spanning scheduled object population and four scenarios, accounting for every scheduled trajectory as included or excluded.

**Blocked by:** 01: Freeze Versioned Pilot Trajectories.

**Status:** ready-for-agent

- [ ] Snapshot covers all 337 object identities and freefall, object-throw, ramp, and conveyor schedules.
- [ ] Every scheduled trajectory is included or has machine-readable exclusion/failure reason.
- [ ] Included trajectories pass completion, validation, synchronization, visibility-policy, and duration gates.
- [ ] Snapshot records schedule, split, content, Utonia artifact, extractor-revision, and checkpoint hashes.
- [ ] Final snapshot is immutable and distinct from engineering-pilot version.
