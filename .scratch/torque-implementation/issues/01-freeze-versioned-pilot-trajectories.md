# 01: Freeze Versioned Pilot Trajectories

**What to build:** Produce an immutable freefall/ramp engineering-pilot snapshot from validated TORQUE Dataset trajectories so every later sample and result can be traced to exact source artifacts.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Snapshot includes only completed trajectories with successful validation, complete synchronized artifacts, and sufficient provenance.
- [ ] Every trajectory has stable scenario/object/condition identity, decomposed initial/scenario conditions, object split, hashes, and explicit exclusion reasons.
- [ ] Repeating snapshot generation against unchanged inputs produces identical contents and summary counts.
- [ ] Human-readable audit reports included objects, scenarios, conditions, termination reasons, warnings, and excluded runs.
