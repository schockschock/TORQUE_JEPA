# 04: Add Ballistic Free-Flight Diagnostic

**What to build:** Add gravity-aware ballistic extrapolation as contact-stratified diagnostic for verified free-flight intervals without mixing it into fair state-only headline controls.

**Blocked by:** 03: Evaluate Persistence And Constant-Velocity Controls.

**Status:** ready-for-agent

- [ ] Diagnostic runs only on windows whose forecast interval satisfies validated free-flight policy.
- [ ] Position and world linear velocity include fixed-gravity acceleration while orientation follows documented angular extrapolation.
- [ ] Results are reported separately from persistence and constant-velocity headline controls.
- [ ] Event-mask errors or contact-crossing windows are rejected with explicit reasons.
