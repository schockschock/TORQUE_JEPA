# TORQUE Dataset Requirements

## Scenarios

TORQUE Dataset contains four controlled interaction regimes:

- **Free fall and rebound:** gravity, inertia, impact, restitution, and rotational response.
- **Object throw:** nonzero initial velocity, wall impact, and rotational response.
- **Ramp:** sustained support contact, friction, sliding, and rolling.
- **Conveyor:** relative motion, frictional entrainment, slip, and rolling on a moving support.

Use the same object population across scenarios. Scenario labels are experimental metadata, not observable model context.

## Object Identity

`object_id` identifies one rigid-body asset with fixed geometry and intrinsic physical properties. TORQUE Dataset has no separate shape identity; do not create `shape_id`. One object has multiple trajectories under different initial and scenario conditions.

COM position is canonical. Actor-origin position remains compatibility metadata only.

## Coordinate And State Contract

Capture conventions:

- world frame is right-handed with +Z up;
- orientation is active actor-to-world WXYZ quaternion;
- linear velocity is COM velocity in world coordinates;
- angular velocity is represented in the authored actor frame for model state;
- original matrices, quaternions, world angular velocity, and timestamps remain available for validation.

For each sample, define the Observation-End Frame as world-aligned with origin at final observed COM position `p_e`.

Observed state at `t <= e`:

```text
p_com_world[t] - p_e
rot6d(R_actor_to_world[t])
v_com_world[t]
omega_actor[t]
```

Future target at `t > e`:

```text
p_com_world[t] - p_e
rot6d(R_actor_to_world[e]^T R_actor_to_world[t])
v_com_world[t]
omega_actor[t]
```

World reconstruction is:

```text
p_com_world[t] = p_e + displacement[t]
R_actor_to_world[t] = R_actor_to_world[e] R_relative[t]
```

## Information Classes

### Observable Context

- past kinematic state;
- Camera A RGB and metric depth;
- object-normalized geometry through Utonia features;
- timestamps;
- observation-end COM world location.

Camera calibration, scenario label, simulator segmentation, contact state, and gravity are not primary model inputs. Camera A differs across scenarios; multimodal cross-scenario transfer therefore includes viewpoint shift.

### Privileged Context

- mass, volume, density, inertia, and COM offset;
- friction and restitution;
- gravity and scenario parameters;
- support or environment geometry;
- simulator-only quantities.

Privileged context supports references and representation analysis, never primary methods.

RGB-D observations may depict visible surfaces, but no explicit support/environment representation is supplied. Contact and segmentation may support validation, sampling, and stratified evaluation. They are not observable context.

## Sampling Frequency And Windows

State, RGB, and depth are captured synchronously at 120 Hz. Models use stride-two 60 Hz samples.

```yaml
observation_seconds: 0.25
prediction_seconds: 0.50
observation_samples: 16
prediction_samples: 30
capture_frequency_hz: 120
model_frequency_hz: 60
```

For even native frame `e`:

```text
observation: e-30, e-28, ..., e
future:      e+2,  e+4,  ..., e+60
```

Require Camera A frustum visibility at each of 16 sampled observation frames. Future visibility is recorded but does not invalidate a window.

## Versioned Manifests

Create immutable trajectory and window artifacts. Each version records generation time, schedule hash, object-split hash, included trajectory hashes, and policy configuration.

Trajectory ID is the relative run key:

```text
{scenario}/{object_id}/{condition_id}
```

Trajectory manifest stores identity, paths, object split, `initial_condition_id`, `scenario_condition_id`, their parameter records, validation result, termination reason, frame count, and content hashes. Protocol-specific partition artifacts reference these stable condition IDs. A run is eligible only when:

```text
metadata.status == "completed"
AND validation.valid == true
AND all required synchronized files exist
AND frame count supports complete observation and future windows
```

Window index stores trajectory ID, native frame bounds, exact timestamps, observation visibility, contact phase, and validity flags.

Freeze separate pilot and final dataset versions. Never infer an experiment dataset from a mutable live directory scan.

## Object Split

Use existing object-disjoint partition:

```text
split_seed: 1729
train_objects: 236
validation_objects: 51
test_objects: 50
```

All trajectories from one object remain in one partition. Fit every normalization statistic using training objects only.

## Sample API

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

First geometry path loads Utonia `global_mean` and `global_max`, not local feature arrays. First visual path resizes complete Camera A frames to 224 x 224 without augmentation. Invalid depth becomes zero with explicit validity mask; finite values use training-object normalization and mask-aware resizing.

### Utonia Feature Provenance

Existing feature artifacts were produced by `/home/adrien/TORQUE/data_processing/extract_utonia_features.py` with:

```text
source mesh variant: original OBJ
surface sampling: 32,768 area-weighted points
base seed: 53,124, deterministically specialized per object
checkpoint: utonia
repository: Pointcept/Utonia
backbone: Point Transformer V3
object scale: 1.0 after unit-sphere normalization
grid sampling: 0.01
upcast levels: 4 concatenated hierarchy levels
feature dimension: 1,386
pooled outputs: surface-mapped global mean and global max
```

Utonia preprocessing centres sampled geometry by sampled-point mean and unit-sphere normalization; it does not centre geometry at simulator COM. Each NPZ stores source mesh, normalization centre/radius, extraction settings, and local/global arrays. Final trajectory manifest must record NPZ SHA-256, exact Utonia source revision, and downloaded checkpoint SHA-256. Standardize pooled features using training objects only.

## Window Sampling

- Index every eligible 60 Hz endpoint for training.
- Keep at most one window per trajectory in each global batch.
- Balance no-contact, contact-onset, sustained-contact, and post-contact windows with validated simulator labels.
- Use deterministic evaluation endpoints at 30-sample stride.
- Treat windows as correlated observations, not independent statistical samples.

## Required Validation

Before training, verify:

- COM position and velocity use same reference point;
- inertia is measured about COM;
- mesh/body frame and COM offset conventions are documented;
- active actor-to-world WXYZ orientation reconstructs stored COM position;
- actor angular velocity equals `R_actor_to_world^T omega_world`;
- finite-difference COM velocity matches recorded COM velocity away from impulses;
- every modality timestamp is synchronized;
- 120-to-60 Hz indexing yields 16 observed and 30 future samples;
- relative state reconstructs world trajectories;
- rot6d projection yields valid SO(3) matrices;
- contact and visibility labels agree with stored events and rendered observations.

## Generalization Protocols

Run primary protocols separately so each changes one axis:

- object-disjoint;
- initial-condition-disjoint with held-out joint combinations and represented marginals;
- scenario-condition-disjoint interpolation and extrapolation;
- leave-one-scenario-out transfer with same object population.

Combined-axis tests are secondary. There is no shape-disjoint protocol.
