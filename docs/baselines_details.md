# Baseline Tracks

Keep information tracks separate. Privileged simulator references, state-only models, and multimodal models do not belong in one undifferentiated leaderboard.

## Implementation Order

1. Physical controls.
2. State-only TORQUE-Sup.
3. Multimodal TORQUE-Sup and modality progression.
4. State-only then multimodal TORQUE-JEPA with Forecast Probes on frozen representations.
5. Generalization, ablations, intrinsic probes, and violation-of-expectation analysis.
6. External adaptations and privileged references.

## Physical Controls

Required before learned models:

1. Persistence.
2. Constant world linear and angular velocity.

Add ballistic extrapolation after free-flight event masks are validated. Add training-object-only kNN and convex/exact simulator references later. Simulator references are privileged, not fair state-only competitors.

## State-Only Supervised Track

Initial milestone:

1. TORQUE-Sup.

Subsequent state baselines:

1. MLP.
2. GRU or LSTM.
3. Direct State Transformer.
4. Autoregressive State Transformer.

TORQUE-Sup is matched to TORQUE-JEPA in shared backbone capacity, data, rollout, backbone updates, and search budget, but differs in direct supervised training versus representation pretraining and frozen probing.

## Multimodal Supervised Tracks

Train separate models for:

```text
state
state + RGB
state + RGB-D
state + geometry
state + RGB-D + geometry
```

RGB comparisons:

1. scratch ResNet-50;
2. frozen ImageNet ResNet-50;
3. frozen DINOv2 ViT-S/14.

Depth uses separate scratch ResNet-18. Visual-from-scratch results are required before interpreting pretrained features. A modality-dropout model is secondary robustness experiment.

## Geometry-Aware And Related Methods

After internal TORQUE baselines:

1. geometry-conditioned Transformer;
2. ObjectForesight adapted to TORQUE Dataset information contract and outputs;
3. FIGNet-style rigid-body or interaction-graph model;
4. LeWorldModel adapted to TORQUE Dataset;
5. DINO-WM adapted with Forecast Probes on frozen representations where applicable.

Every adaptation must declare exact observable and privileged inputs. Do not claim task novelty from future 6-DoF forecasting alone; ObjectForesight is explicit prior work.

## Privileged Track

Later references may condition on physical parameters or simulator information:

- parameter-conditioned State Transformer;
- parameter-conditioned TORQUE-JEPA variant;
- convex or exact simulator reference.

Report these separately as privileged ceilings or analyses.
