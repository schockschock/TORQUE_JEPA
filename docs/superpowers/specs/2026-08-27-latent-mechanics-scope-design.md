# Latent-Mechanics Scope for Initial TORQUE-JEPA Baseline

## Context

TORQUE-JEPA is a representation-learning method whose future-predictive latent representations are trained against per-timestep kinematic-state targets. A mechanics-inspired latent objective could encourage coherent long-horizon dynamics or interpretable latent structure, but contact events make rigid-body trajectories piecewise smooth and introduce assumptions that are not yet validated.

## Decision

The initial TORQUE-JEPA implementation uses only the per-step joint-embedding prediction loss and SIGReg:

```text
L = L_JEPA + lambda_sigreg * L_SIGReg
```

No mechanics-inspired regularizer is part of the initial method. This excludes decoded finite-difference residuals, free-flight conservation losses, latent semigroup constraints, Hamiltonian or Koopman structure, contact-regime gates, and hybrid impulse transitions.

## Rationale

First training run must establish an interpretable baseline before adding assumptions about latent dynamics. Mechanics losses would otherwise confound whether improvements come from state-grounded joint-embedding learning, direct kinematic supervision, privileged physical information, or manually imposed dynamics. Impact, rolling, sliding, and free-flight intervals also require different physical assumptions; applying one unvalidated residual across them risks rewarding incorrect behavior.

## Evaluation Boundary

Forecast quality is measured through a shallow Forecast Probe trained on frozen TORQUE-JEPA representations. Mechanical accessibility is measured through separate probes on frozen representations and correlational analyses. Neither evaluation path backpropagates a mechanics loss into the initial TORQUE-JEPA backbone.

## Deferred Work

Latent-space mechanics remains an open research direction after baseline results exist. No follow-up loss family or implementation order is selected yet.
