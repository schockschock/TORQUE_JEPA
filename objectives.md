# TORQUE Implementation Brief

## Project And Task

TORQUE Project comprises TORQUE Dataset and methods trained and evaluated on it. TORQUE Dataset supports **rigid-body kinematic-state forecasting**: predicting future centre-of-mass (COM) position, orientation, COM linear velocity, and angular velocity from a causal observation window.

Primary methods are:

- **TORQUE-Sup**: supervised forecasting with direct kinematic-state losses;
- **TORQUE-JEPA**: state-grounded representation learning with autoregressive latent prediction and Forecast Probes trained on frozen representations.

TORQUE-JEPA is representation-first. Its backbone is not fine-tuned by forecast labels after pretraining.

## Information Contract

Primary observable context contains:

- past COM-referenced kinematic state;
- Camera A RGB and metric depth history;
- object-normalized geometry through precomputed Utonia features;
- timestamps, represented to the model with sinusoidal sample-index encoding;
- observation-end COM world location.

Scenario labels, Camera A calibration, simulator segmentation, contact labels, physical parameters, and explicit support or environment geometry are not primary model inputs. RGB-D observations may depict visible surfaces. Contact and segmentation may be used for sampling, validation, and stratified evaluation. Mass, volume, density, inertia, COM offset, friction, restitution, gravity, and simulator-only parameters are reserved for privileged references or representation analysis.

The observation-end world location is intentionally included because state-only collision timing is otherwise not identifiable in fixed world layouts. Forecast outputs remain relative. A no-world-location ablation is required.

## Dataset And Dataloader

### Source Data

```text
dataset root:     /data2/adrien/TORQUE/
assets:           /data2/adrien/TORQUE/assets
Utonia features:  /data2/adrien/TORQUE/3D_features
simulations:      /data2/adrien/TORQUE/v2
paper draft:      /home/adrien/TORQUE_JEPA/docs/iclr_paper
```

The simulator captures synchronized state, RGB, and depth at 120 Hz. Models use a synchronized 60 Hz stride-two view.

Canonical window:

```yaml
observation_seconds: 0.25
prediction_seconds: 0.50
observation_samples: 16
prediction_samples: 30
capture_frequency_hz: 120
model_frequency_hz: 60
position_reference: com
coordinate_frame: observation_end_translated_world
prediction_mode: autoregressive
rotation_representation: rot6d
```

For even 120 Hz observation-end frame `e`:

```text
observation: e-30, e-28, ..., e
future:      e+2,  e+4,  ..., e+60
```

### Versioned Manifests

Use two immutable artifacts.

Trajectory manifest, created before window extraction:

```text
dataset_version
schedule_hash
split_hash
trajectory_id                 # {scenario}/{object_id}/{condition_id}
object_id
scenario
condition_id
condition_parameters
initial_condition_id
initial_condition_parameters
scenario_condition_id
scenario_condition_parameters
split
geometry_path
utonia_path
state_path
rgb_a_path
depth_a_path
metadata_path
validation_path
contact_path
frame_count
termination_reason
validity_flags
content_hashes
```

Window index:

```text
trajectory_id
start_frame
observation_end_frame
future_end_frame
observation_timestamps
future_timestamps
observation_visibility
contact_phase
validity_flags
```

Do not create `shape_id`: each `object_id` identifies one rigid-body asset with fixed geometry and intrinsic physical properties.

A trajectory is eligible only when metadata reports completion, `validation.json` reports `valid: true`, required synchronized files exist, and the sequence is long enough. Observation eligibility requires Camera A frustum visibility at all 16 sampled input frames. Future visibility is recorded but not required.

Freeze separate versioned pilot and final manifests. Never train against a live directory scan.

### Object Split

Keep the existing object-disjoint split:

```text
seed:       1729
train:      236 objects
validation: 51 objects
test:       50 objects
```

All trajectories from one object remain in one partition. Fit normalization statistics on training objects only.

### Sample Structure

```python
{
    "past": {
        "state": FloatTensor[16, 15],
        "rgb_a": FloatTensor[16, 3, 224, 224] | None,
        "depth_a": FloatTensor[16, 1, 224, 224] | None,
        "depth_valid_a": BoolTensor[16, 1, 224, 224] | None,
        "timestamps": FloatTensor[16],
        "availability_mask": BoolTensor[16, 3],  # state, rgb_a, depth_a
    },
    "static": {
        "utonia_global_mean": FloatTensor[1386] | None,
        "utonia_global_max": FloatTensor[1386] | None,
        "geometry_available": bool,
    },
    "observation_end": {
        "p_com_world": FloatTensor[3],
        "r_actor_to_world": FloatTensor[3, 3],
    },
    "target": {
        "state": FloatTensor[30, 15],
        "timestamps": FloatTensor[30],
    },
    "metadata": {
        "object_id": str,
        "trajectory_id": str,
        "scenario": str,
        "condition_id": str,
        "split": str,
    },
}
```

Resize complete Camera A frames to 224 x 224 without augmentation. Use encoder-native RGB normalization. Replace invalid depth with zero, return an explicit validity mask, and normalize finite depth using training-object statistics with mask-aware resizing.

### State Convention

Let `e` denote observation end and `R_t` the active actor-to-world rotation.

Observed state at time `t <= e`:

```text
p_com_world[t] - p_com_world[e]
rot6d(R_t)
v_com_world[t]
omega_actor[t]
```

Future target at time `t > e`:

```text
p_com_world[t] - p_com_world[e]
rot6d(R_e^T R_t)
v_com_world[t]
omega_actor[t]
```

The evaluator reconstructs `p_com_world[t]` by adding the observation-end world location and reconstructs orientation as `R_e R_relative[t]`. Keep original matrices and timestamps for evaluation. Project rot6d predictions to valid SO(3) matrices before geodesic loss or metrics.

Normalize displacement, world linear velocity, actor angular velocity, observation-end world location, finite depth, and Utonia summaries by separate training-object statistics. Do not scalar-standardize rot6d.

### Window Sampling

- Index every eligible 60 Hz observation endpoint for training.
- Sample at most one window per trajectory in each global batch.
- Balance training windows across no-contact, contact-onset, sustained-contact, and post-contact phases using validated contact labels; labels never enter primary models.
- Evaluate deterministic endpoints at a 30-sample stride.
- Aggregate windows within trajectory, trajectories within scenario/object, scenarios equally within object, then objects equally.

## Physical Controls

Implement before learned methods:

1. Persistence.
2. Constant linear and angular velocity.

The constant-velocity control holds observation-end world linear and world angular velocity fixed. It integrates orientation as `R(t) = Exp(t [omega_world]) R_e`.

Ballistic free-flight extrapolation follows after event masks are validated. kNN retrieval and simulator references are later baselines; simulator references are privileged.

## TORQUE-Sup

TORQUE-Sup is the first learned method.

Milestones:

1. state-only TORQUE-Sup;
2. full multimodal TORQUE-Sup;
3. cumulative information tracks.

Information tracks use separate main models:

```text
state
state + RGB
state + RGB-D
state + geometry
state + RGB-D + geometry
```

A universal modality-dropout model is a secondary robustness experiment, not the main modality comparison.

RGB comparisons include a scratch ResNet-50, frozen ImageNet ResNet-50, and frozen DINOv2 ViT-S/14. Depth uses a separate scratch ResNet-18. First fusion uses one pooled RGB token and one pooled depth token per timestep. First geometry fusion projects concatenated Utonia global mean/max into a static token.

TORQUE-Sup uses the observation-end state converted to target coordinates at horizon one: zero COM displacement, identity relative rotation, observation-end world COM velocity, and observation-end actor angular velocity. Later teacher-forced steps use true previous future state; free-running steps use decoded previous prediction. Its direct loss assigns equal initial weight to:

- normalized COM displacement regression;
- SO(3) geodesic orientation loss;
- normalized world linear-velocity regression;
- normalized actor angular-velocity regression.

## Shared Multimodal Core

First reproducible core:

```text
hidden dimension: 256
context blocks: 4
autoregressive predictor blocks: 6
attention heads: 8
FFN dimension: 1024
dropout: 0.1
```

- one kinematic token per observed timestep;
- sinusoidal encoding over sampled sequence indices;
- observation-end world location concatenated into each state token;
- same-timestep state-to-vision cross-attention;
- gated cross-attention from dynamic tokens to persistent geometry token;
- causal Transformer over fused dynamic sequence;
- full observed context available to every autoregressive future query.

Keep core width, depth, optimizer updates, and search budget matched across information tracks and TORQUE-Sup/TORQUE-JEPA. Report encoder parameters and FLOPs rather than padding smaller tracks.

## TORQUE-JEPA

### Method Identity

TORQUE-JEPA learns future-predictive representations using training-only future kinematic states. It is state-grounded representation learning, not self-supervised visual JEPA and not end-to-end supervised forecasting.

### Target And Predictor

At each future timestep:

```text
future target state -> one shared trainable target MLP + BatchNorm -> z_target[32]

context + previous z + future index -> autoregressive predictor -> h_pred[256]
h_pred -> predictor MLP + BatchNorm -> z_pred[32]
```

The Forecast Probe reads `h_pred`, not 32-D objective latent `z_pred`.

At horizon one, previous latent is a learned BOS token. Teacher forcing feeds the previous target `z_target`; free-running prediction feeds previous `z_pred`. The target encoder receives state only, not horizon or timestamp. It is trainable with no stop-gradient or EMA.

### Objective

```text
L = mean_h,b ||z_pred[b, h] - z_target[b, h]||_2^2
  + lambda_sigreg * mean_h SIGReg({z_target[b, h]} over batch b)
```

SIGReg applies independently per horizon to target projector outputs only. Initial contract:

```text
effective global batch: 128
random projections: 1024
quadrature knots: 17
lambda sweep: [0.01, 0.03, 0.1, 0.2]
```

No state-consistency, latent-mechanics, contact, or future-visual loss enters the initial method. These are later ablations. Explicit static/dynamic latent factorization and Perceiver fusion are also later ablations.

### Rollout Curriculum

Use fixed fractions of matched backbone updates:

```text
40% teacher forcing
20% maximum free-running prefix length 2
20% maximum free-running prefix length 4
20% maximum free-running prefix length 8
```

In each free-running stage, sample prefix length uniformly from one through stage maximum, recurse from horizon one for that prefix, then resume teacher forcing for remaining horizons. Compute losses at all 30 horizons. Evaluate all 30 future steps open loop. All-at-once prediction is an ablation.

### Forecast And Mechanical Probes

Freeze context encoder and latent predictor after TORQUE-JEPA training. Report:

- linear Forecast Probe;
- shared-horizon two-layer Forecast Probe `256 -> 256 -> state outputs`, with no horizon input and no backbone updates.

Static intrinsic probes read observation-end context summary. Dynamic and collision analyses read per-horizon predicted hidden states. Mandatory object-disjoint intrinsic probes cover mass, volume, density, principal inertia, and COM offset. Because each geometry has one fixed physics record, these results are correlational with geometry and do not establish causal parameter identification.

For intrinsic probes, freeze representations and train both linear and `256 -> 256 -> target` MLP probes. Use same object train/validation/test split, training-object target normalization, and equal object weighting. Aggregate window predictions within trajectory and object before reporting object-level MAE in physical units and R2. Targets are mass in kg, volume in m3, density in kg/m3, three principal inertia values in kg m2, and three actor-frame COM-offset values in m.

Violation-of-expectation analysis is required after forecasting and probes, but not in first engineering pilot. Its perturbations, matched controls, surprise score, and statistical test require separate preregistered design before implementation; current brief does not claim that protocol is defined.

## Generalization And Evaluation

Primary protocols vary one axis at a time:

1. object-disjoint split;
2. initial-condition-disjoint split using held-out joint combinations while retaining marginal values;
3. scenario-condition-disjoint interpolation and extrapolation;
4. leave-one-scenario-out transfer using the same object population.

Combined-axis stress tests are secondary. There is no shape-disjoint protocol.

Camera A pose differs by scenario and calibration is withheld. Multimodal cross-scenario transfer therefore includes viewpoint shift and must not be described as isolated scenario transfer.

Primary metrics:

- position ADE and FDE in metres;
- mean and final SO(3) geodesic orientation error in degrees;
- world linear-velocity L2 error in m/s;
- angular-velocity L2 error in rad/s;
- dense per-horizon curves.

Compute metrics after world reconstruction. Use object-macro validation for checkpoint selection, object bootstrap confidence intervals, one pilot seed, and three locked final seeds. Do not combine state families into one headline score.

TORQUE-Sup and TORQUE-JEPA share backbone token count, optimizer updates, hidden widths, rollout protocol, and equally sized validation search budgets. Report Forecast Probe cost separately.

## Scientific Scope

TORQUE Dataset supplies exact simulator targets, but primary forecasting is partially observed because physical parameters and explicit environment representations are withheld. Describe simulator dynamics as deterministic; do not claim observable context admits only one future.

The contribution is controlled multimodal kinematic-state forecasting across rigid-body interaction regimes and state-grounded predictive representation analysis. Do not claim first future 6-DoF forecasting; compare against adapted ObjectForesight. Use **rigid-body mechanics**, not **solid mechanics**, for central claims.

Current fixed scene layouts are accepted scope. World-location conditioning may exploit layout regularities, so claims do not cover layout transfer. Physical properties are fixed per object, so object-disjoint probes do not disentangle mechanics from geometry.

## Implementation Order

1. Dataset audit, versioned manifests, transforms, reconstruction tests, and metrics.
2. Persistence and constant-velocity controls.
3. State-only TORQUE-Sup.
4. Joint and scenario-specific freefall/ramp engineering pilots.
5. Full multimodal TORQUE-Sup and cumulative modality tracks.
6. State-only TORQUE-JEPA and Forecast Probes on frozen representations.
7. Multimodal TORQUE-JEPA and modality tracks.
8. Generalization protocols, required ablations, intrinsic probes, and violation-of-expectation analysis.
9. Adapted LeWorldModel, DINO-WM, ObjectForesight/FIGNet-style, and privileged references.

Freefall/ramp pilot results are engineering evidence only. Publication claims require a frozen final manifest covering the completed object population and all four scenarios.
