# Condition on observation-end world location

Primary TORQUE methods receive observation-end COM world location while predicting position and orientation relative to the observation end. Purely relative state makes collision timing unidentifiable for otherwise identical windows at different heights or distances from fixed surfaces. Full world location can also expose fixed-layout shortcuts, so no explicit support/environment geometry is supplied, a no-world-location ablation is required, and claims explicitly exclude layout transfer.

## Considered Options

- Evaluator-only world location preserves translation invariance but leaves state-only contact forecasting partially non-identifiable.
- Derived surface distances provide stronger physical semantics but disclose environment-derived information excluded from the primary contract.
- World-location conditioning preserves current data and supports state-only forecasting, at the cost of fixed-layout dependence.

## Consequences

TORQUE methods are world-location-conditioned, not translation-invariant. Observation-end world location is observable context; scenario labels and explicit environment representations are not. Multimodal cross-scenario evaluation remains additionally confounded by scenario-specific viewpoint shift.
