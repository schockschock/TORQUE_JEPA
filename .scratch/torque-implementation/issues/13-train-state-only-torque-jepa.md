# 13: Train State-Only TORQUE-JEPA

**What to build:** Pretrain state-only TORQUE-JEPA with trainable state targets and SIGReg, then evaluate predicted representations through linear and shallow Forecast Probes.

**Blocked by:** 12: Train Full Multimodal TORQUE-Sup.

**Status:** ready-for-agent

- [ ] Predictor emits 256-D hidden representation and recursive 32-D objective latent with learned BOS and full-context access.
- [ ] Shared target MLP is trainable without EMA/stop-gradient; per-horizon target-only SIGReg follows locked estimator contract.
- [ ] Curriculum implements documented teacher-forced and 2/4/8-step free-running prefixes before 30-step open-loop evaluation.
- [ ] Backbone freezes before linear and `256 -> 256 -> outputs` Forecast Probe training.
- [ ] Report compares frozen-representation forecasts with matched state-only TORQUE-Sup under equal backbone search budget.
