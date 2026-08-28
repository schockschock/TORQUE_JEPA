# TORQUE

TORQUE studies representation learning for forecasting rigid-body motion from controlled multimodal observations.

## Project And Methods

**TORQUE Project**:
Research project comprising the TORQUE Dataset and methods trained and evaluated on it.
_Avoid_: TORQUE method

**TORQUE Dataset**:
Dataset for future rigid-body kinematic-state forecasting across controlled interaction scenarios.
_Avoid_: TORQUE-JEPA dataset

**TORQUE-Sup**:
Supervised-learning method trained and evaluated on the TORQUE Dataset.
_Avoid_: TORQUE supervised dataset

**TORQUE-JEPA**:
Representation-learning method whose future-predictive latent representations are trained using a state-grounded joint-embedding objective.
_Avoid_: TORQUE-JePA, TORQUE_JEPA

**Forecast Probe**:
Shallow decoder trained on frozen TORQUE-JEPA representations to map predicted latents to future kinematic states.
_Avoid_: TORQUE-JEPA decoder, fine-tuning head

## Forecasting

**Kinematic State**:
Rigid body's centre-of-mass position, orientation, centre-of-mass linear velocity, and angular velocity at one time.
_Avoid_: 6-DoF state, physical state

**Kinematic-State Forecasting**:
Prediction of future rigid-body kinematic states from a causal observation window.
_Avoid_: state prediction, pose forecasting

**Observable Context**:
Information available to primary methods: past kinematic state, single-view RGB-D observations, object geometry, timestamps, and observation-end world location.
_Avoid_: main input, ordinary metadata

**Privileged Context**:
Explicit physical parameters or environment representations withheld from primary methods and reserved for privileged references or representation analysis.
_Avoid_: static context, observable context

## Dataset Identity

**Shape**:
Geometry of an object. TORQUE Dataset does not assign shapes a separate identity.
_Avoid_: object, `shape_id`, shape cluster

**Object**:
Rigid-body asset identified by `object_id`, with fixed geometry and intrinsic physical properties.
_Avoid_: shape, trajectory

**Trajectory**:
One simulated rollout of one object under one scenario and initial condition.
_Avoid_: object, window, shape

**Initial Condition**:
Object's kinematic state at trajectory start.
_Avoid_: scenario condition, trajectory condition

**Scenario Condition**:
Scenario-specific configuration governing an interaction regime, such as ramp angle or conveyor speed.
_Avoid_: initial condition, scenario

**Trajectory Condition**:
Combination of initial condition and scenario condition that configures one trajectory.
_Avoid_: initial condition, scenario condition

**Scenario**:
Controlled interaction regime used to generate and group trajectories. Scenario labels are experimental metadata, not observable context.
_Avoid_: model input, trajectory

**Observation-End World Location**:
Object's centre-of-mass world position at final observed timestamp. It conditions primary methods and anchors reconstruction, while forecast outputs remain relative.
_Avoid_: initial position, trajectory origin

**Observation-End Frame**:
World-aligned coordinate frame whose origin is observation-end world location.
_Avoid_: gravity-aligned frame, body frame

## Generalization

**Object-Disjoint Split**:
Dataset partition in which no object identity appears in more than one partition.
_Avoid_: shape-disjoint split, trajectory split

**Cross-Scenario Transfer**:
Evaluation on an interaction scenario absent from training while retaining same object population. With current scenario-specific cameras, multimodal transfer also includes viewpoint shift.
_Avoid_: object-disjoint transfer, isolated scenario transfer

**Initial-Condition-Disjoint Split**:
Trajectory partition that holds out initial kinematic states while retaining represented scenario conditions.
_Avoid_: scenario-condition split, object-disjoint split

**Scenario-Condition-Disjoint Split**:
Trajectory partition that holds out scenario-specific configurations while retaining represented initial conditions.
_Avoid_: cross-scenario transfer, initial-condition split
