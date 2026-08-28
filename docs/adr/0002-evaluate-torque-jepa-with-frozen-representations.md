# Evaluate TORQUE-JEPA with Forecast Probes on frozen representations

TORQUE-JEPA is representation-first: it trains future-predictive latent representations with a state-grounded joint-embedding objective, then freezes the context encoder and latent predictor before fitting linear and shallow nonlinear Forecast Probes. Joint end-to-end forecast fine-tuning would make representation quality inseparable from supervised adaptation and blur the distinction from TORQUE-Sup.

## Considered Options

- Joint forecast-decoder training improves direct task optimization but changes TORQUE-JEPA into a hybrid supervised forecasting method.
- Reporting frozen-probe and fine-tuned variants as co-primary weakens method identity and doubles the main comparison surface.
- Frozen probes preserve the representation-learning claim and provide a consistent accessibility test.

## Consequences

TORQUE-Sup trains the matched backbone end-to-end with direct kinematic-state loss, while TORQUE-JEPA pretrains, freezes, and probes. Comparisons match backbone token count, hidden widths, optimizer updates, rollout protocol, and validation search budget; Forecast Probe cost is reported separately.
