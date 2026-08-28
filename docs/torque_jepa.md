# TORQUE-JEPA

## Method Identity

TORQUE-JEPA is a representation-learning method. It predicts future latent representations from observable context and aligns them to training-only kinematic-state targets. Forecasting uses trainable Forecast Probes on frozen representations after backbone training.

It is state-grounded joint-embedding learning, not self-supervised visual JEPA and not end-to-end supervised forecasting.

## Observable Inputs

```text
past COM-referenced kinematic state
Camera A RGB history
Camera A metric-depth history
Utonia global object-geometry summaries
observation-end COM world location
sample indices through sinusoidal encoding
availability masks
```

The model does not receive Camera A calibration, scenario label, contact state, simulator segmentation, physical parameters, gravity, or explicit environment geometry. RGB-D may depict visible surfaces. Observation-end world location is concatenated into every state token. A no-world-location ablation is required.

## First Shared Core

```text
hidden dimension: 256
context blocks: 4
predictor blocks: 6
attention heads: 8
FFN dimension: 1024
dropout: 0.1
```

Encoders and fusion:

1. Encode one kinematic token per observed timestep.
2. Encode one pooled Camera A RGB token and one pooled depth token per timestep.
3. Let each state token query visual tokens from same timestep.
4. Project Utonia global mean/max into persistent geometry token.
5. Apply gated geometry cross-attention to dynamic tokens.
6. Fuse dynamic sequence with causal Transformer.
7. Let every future query cross-attend full observed context.

RGB tracks compare scratch ResNet-50, frozen ImageNet ResNet-50, and frozen DINOv2 ViT-S/14. Depth uses separate scratch ResNet-18. Main information tracks use separately trained models; modality dropout is secondary robustness experiment.

Hierarchical causal fusion is canonical first architecture. Perceiver fusion, local Utonia tokens, patch-level visual tokens, and explicit static/dynamic latent factorization are ablations.

## State-Grounded Target

One shared trainable MLP with BatchNorm maps every future kinematic state to `z_target` in 32-D objective space. It receives state only, not horizon. It has no stop-gradient or EMA.

The target branch exists only during training and is removed for forecasting.

## Autoregressive Latent Predictor

At horizon `h`:

```text
context + previous objective latent + future index
    -> predictor
    -> h_pred[h] in R^256
    -> MLP + BatchNorm
    -> z_pred[h] in R^32
```

Horizon one uses learned BOS token. Teacher forcing feeds `z_target[h-1]`; free-running prediction feeds `z_pred[h-1]`. Forecast and mechanical probes consume 256-D `h_pred`, not objective latent.

Rollout curriculum uses fixed backbone-update fractions:

```text
40% teacher forcing
20% maximum free-running prefix length 2
20% maximum free-running prefix length 4
20% maximum free-running prefix length 8
```

At each free-running stage, prefix length is sampled uniformly from one through stage maximum. Prediction recurses from horizon one for this prefix, then teacher forcing resumes; losses are computed at all 30 horizons. Full evaluation is 30-step open loop. All-at-once prediction is an ablation.

## Objective

```text
L_JEPA = mean_{batch,horizon} ||z_pred - z_target||_2^2

L = L_JEPA
  + lambda_sigreg * mean_h SIGReg({z_target[b, h]} over batch b)
```

SIGReg applies independently at each horizon to target projector outputs only.

Initial SIGReg contract:

```text
effective global batch: 128
random projections: 1024
quadrature knots: 17
lambda sweep: 0.01, 0.03, 0.1, 0.2
```

No consistency, decoded-state, latent-mechanics, contact, or future-visual loss enters first method. Mechanics-inspired losses remain deferred with no selected roadmap.

## Forecast Probes

After TORQUE-JEPA training:

1. freeze context encoder and latent predictor;
2. remove target branch;
3. train shared-horizon probes on predicted 256-D hidden states;
4. never fine-tune backbone with forecast labels.

Report:

- linear Forecast Probe;
- two-layer MLP Forecast Probe `256 -> 256 -> state outputs` with no horizon input.

Probe outputs are:

```text
future COM displacement from observation-end COM
future rotation relative to observation-end actor orientation
future world COM linear velocity
future actor-frame angular velocity
```

Predict orientation as rot6d, project to SO(3), and train with geodesic loss. Contact prediction is a later auxiliary ablation.

## TORQUE-Sup Comparison

TORQUE-Sup uses same observable inputs, encoder choices, fusion core, predictor width/depth, rollout schedule, split, backbone token count, optimizer updates, and equally sized validation search budget. TORQUE-Sup is trained end-to-end using direct component-aware kinematic-state loss; TORQUE-JEPA is pretrained then frozen and probed. Report Forecast Probe cost separately.

At horizon one, TORQUE-Sup receives observation-end state converted to target coordinates: zero displacement, identity relative rotation, world COM velocity, and actor angular velocity. Later teacher-forced steps receive true previous future state; free-running steps receive decoded previous prediction.

## Information Tracks

Train separate main models for:

```text
state
state + RGB
state + RGB-D
state + geometry
state + RGB-D + geometry
```

Train state-only TORQUE-JEPA first, then cumulative multimodal tracks.

## Mechanical Representation Analysis

Static intrinsic probes use observation-end context summary. Dynamic and collision analyses use per-horizon predicted hidden states.

Required object-disjoint intrinsic probes:

- mass;
- volume;
- density;
- principal inertia;
- COM offset.

Freeze representations and train both linear and `256 -> 256 -> target` MLP probes. Use same object train/validation/test split, training-object target normalization, and equal object weighting. Aggregate predictions within trajectory and object before reporting object-level MAE in physical units and R2. Targets are mass in kg, volume in m3, density in kg/m3, three principal inertia values in kg m2, and three actor-frame COM-offset values in m.

Because each geometry has one fixed physical record, probe transfer remains correlational with geometry. Do not claim causal parameter identification. Friction, restitution, and gravity probes are deferred because current data lacks independent variation.

Violation-of-expectation analysis is required after core forecasting and probes. It is not part of first engineering pilot. Perturbations, matched controls, surprise score, and statistical test require separate preregistered design before implementation.

## Evaluation

Compute forecasts after world reconstruction and report:

- position ADE/FDE in metres;
- mean/final SO(3) geodesic orientation error in degrees;
- world linear-velocity L2 error in m/s;
- angular-velocity L2 error in rad/s;
- dense horizon curves;
- object-bootstrap confidence intervals.

Aggregate windows within trajectory, trajectories within scenario/object, scenarios equally within object, then objects equally. Select checkpoints using object-macro validation loss. Use one engineering-pilot seed and three locked final seeds.

Primary generalization protocols vary one axis at a time. Multimodal cross-scenario transfer includes viewpoint shift because Camera A pose changes and calibration is withheld.
