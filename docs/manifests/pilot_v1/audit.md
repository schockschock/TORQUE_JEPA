# TORQUE Pilot Trajectory Snapshot Audit

- scope: `pilot-freefall-ramp`
- dataset_version: `2.0`
- format_version: `1`
- schedule_hash: `539474f75d2433cec51866a588152cb50c404394e8c8b0d6c6bc116a745f9868`
- split_hash: `71345f97c4988c76a7bd1db3bb546e5d8742d9144e6efd03515fee4262cd7fa7`

## Summary

- scheduled: 6066 (included 2491, excluded 3575)

| scenario | scheduled | included | excluded |
| --- | ---: | ---: | ---: |
| freefall | 3033 | 1227 | 1806 |
| ramp | 3033 | 1264 | 1769 |

## Object Split

| split | scheduled | included |
| --- | ---: | ---: |
| test | 900 | 245 |
| train | 4248 | 1841 |
| validation | 918 | 405 |

## Objects

- objects with included trajectories: 142
- objects without pilot trajectories: 195

| split | objects | included trajectories |
| --- | ---: | ---: |
| test | 15 | 245 |
| train | 104 | 1841 |
| validation | 23 | 405 |

Objects without pilot trajectories: 4R5-4, 4R5-5, 4R5-6, 4R5-7, 4R5-8, 4R5-9, 4R6-1, 4R6-10, 4R6-2, 4R6-3, 4R6-4, 4R6-5, 4R6-6, 4R6-7, 4R6-8, 4R6-9, 4R8-1, 4R8-10, 4R8-2, 4R8-3, 4R8-4, 4R8-5, 4R8-6, 4R8-7, 4R8-8, 4R8-9, 4R9-1, 4R9-10, 4R9-2, 4R9-3, 4R9-4, 4R9-5, 4R9-6, 4R9-7, 4R9-8, 4R9-9, 5R1-1, 5R1-10, 5R1-2, 5R1-3, 5R1-4, 5R1-5, 5R1-6, 5R1-7, 5R1-8, 5R1-9, 5R2-1, 5R2-10, 5R2-2, 5R2-3, 5R2-4, 5R2-5, 5R2-6, 5R2-7, 5R2-8, 5R2-9, 5R3-1, 5R3-10, 5R3-2, 5R3-3, 5R3-4, 5R3-5, 5R3-6, 5R3-7, 5R3-8, 5R3-9, R1-1, R1-10, R1-2, R1-3, R1-4, R1-5, R1-6, R1-7, R1-8, R1-9, R10-1, R10-10, R10-2, R10-3, R10-4, R10-5, R10-7, R10-8, R10-9, R11-1, R11-10, R11-2, R11-3, R11-4, R11-5, R11-6, R11-7, R11-8, R11-9, R12-1, R12-10, R12-2, R12-3, R12-4, R12-5, R12-6, R12-7, R12-8, R12-9, R13-1, R13-10, R13-2, R13-3, R13-4, R13-5, R13-6, R13-7, R13-8, R13-9, R2-1, R2-10, R2-2, R2-3, R2-4, R2-5, R2-6, R2-7, R2-8, R2-9, R3-1, R3-10, R3-2, R3-3, R3-4, R3-5, R3-6, R3-7, R3-8, R3-9, R4-1, R4-10, R4-2, R4-3, R4-4, R4-5, R4-6, R4-7, R4-8, R4-9, R5-1, R5-10, R5-2, R5-3, R5-4, R5-5, R5-6, R5-7, R5-8, R5-9, R6-1, R6-10, R6-2, R6-3, R6-4, R6-5, R6-6, R6-7, R6-8, R6-9, R7-1, R7-10, R7-2, R7-3, R7-4, R7-5, R7-6, R7-7, R7-8, R7-9, R8-1, R8-10, R8-2, R8-3, R8-4, R8-5, R8-6, R8-7, R8-8, R8-9, R9-1, R9-10, R9-2, R9-3, R9-4, R9-5, R9-6, R9-7, R9-8, R9-9

## Conditions

### freefall

| condition_id | scheduled | included |
| --- | ---: | ---: |
| pose0_angvel0 | 337 | 134 |
| pose0_angvel1 | 337 | 134 |
| pose0_angvel2 | 337 | 136 |
| pose1_angvel0 | 337 | 136 |
| pose1_angvel1 | 337 | 135 |
| pose1_angvel2 | 337 | 140 |
| pose2_angvel0 | 337 | 135 |
| pose2_angvel1 | 337 | 137 |
| pose2_angvel2 | 337 | 140 |

### ramp

| condition_id | scheduled | included |
| --- | ---: | ---: |
| angle20deg_pose0 | 337 | 139 |
| angle20deg_pose1 | 337 | 140 |
| angle20deg_pose2 | 337 | 139 |
| angle30deg_pose0 | 337 | 141 |
| angle30deg_pose1 | 337 | 141 |
| angle30deg_pose2 | 337 | 141 |
| angle40deg_pose0 | 337 | 141 |
| angle40deg_pose1 | 337 | 141 |
| angle40deg_pose2 | 337 | 141 |

## Termination Reasons

| termination_reason | count |
| --- | ---: |
| max_duration | 2 |
| out_of_view | 755 |
| settled | 1734 |

## Warnings

- trajectories with warnings: 755

| warning | count |
| --- | ---: |
| Object left both cameras; run retained for later policy decision | 755 |

## Excluded Runs

| reason | count |
| --- | ---: |
| insufficient_duration | 52 |
| missing_run_directory | 3523 |

### insufficient_duration

| trajectory_id |
| --- |
| freefall/2R1-1/pose0_angvel2 |
| freefall/2R1-11/pose2_angvel0 |
| freefall/2R1-11/pose2_angvel1 |
| freefall/2R1-2/pose2_angvel0 |
| freefall/2R1-5/pose2_angvel0 |
| freefall/2R2-7/pose0_angvel0 |
| freefall/2R3-4/pose1_angvel1 |
| freefall/2R3-5/pose0_angvel1 |
| freefall/2R3-5/pose1_angvel0 |
| freefall/2R3-6/pose1_angvel0 |
| freefall/2R3-7/pose2_angvel2 |
| freefall/3R1-10/pose0_angvel2 |
| freefall/3R1-2/pose0_angvel2 |
| freefall/3R2-4/pose0_angvel0 |
| freefall/3R2-5/pose0_angvel0 |
| freefall/3R2-8/pose0_angvel0 |
| freefall/3R2-8/pose0_angvel1 |
| freefall/3R3-9/pose0_angvel1 |
| freefall/3R4-10/pose0_angvel0 |
| freefall/3R4-10/pose1_angvel1 |
| freefall/3R4-4/pose0_angvel2 |
| freefall/3R5-6/pose1_angvel2 |
| freefall/3R6-10/pose2_angvel1 |
| freefall/3R6-4/pose0_angvel1 |
| freefall/3R6-4/pose1_angvel0 |
| freefall/3R6-5/pose2_angvel1 |
| freefall/3R7-1/pose0_angvel0 |
| freefall/3R7-1/pose0_angvel1 |
| freefall/3R7-10/pose1_angvel0 |
| freefall/3R7-3/pose1_angvel1 |
| freefall/4R1-3/pose2_angvel0 |
| freefall/4R1-4/pose2_angvel0 |
| freefall/4R1-4/pose2_angvel1 |
| freefall/4R1-6/pose1_angvel1 |
| freefall/4R1-7/pose0_angvel2 |
| freefall/4R2-2/pose0_angvel0 |
| freefall/4R2-3/pose1_angvel1 |
| freefall/4R2-6/pose2_angvel0 |
| freefall/4R2-8/pose1_angvel0 |
| freefall/4R2-8/pose1_angvel1 |
| freefall/4R3-1/pose1_angvel0 |
| freefall/4R3-1/pose1_angvel1 |
| freefall/4R4-2/pose0_angvel2 |
| freefall/4R4-4/pose0_angvel1 |
| freefall/4R4-7/pose0_angvel1 |
| freefall/4R5-3/pose0_angvel0 |
| freefall/4R5-3/pose0_angvel1 |
| ramp/2R1-1/angle20deg_pose2 |
| ramp/2R2-8/angle20deg_pose0 |
| ramp/2R3-1/angle20deg_pose0 |
| ramp/3R7-8/angle20deg_pose1 |
| ramp/4R2-8/angle20deg_pose2 |

### missing_run_directory (objects not yet simulated)

- freefall: 4R5-3, 4R5-4, 4R5-5, 4R5-6, 4R5-7, 4R5-8, 4R5-9, 4R6-1, 4R6-10, 4R6-2, 4R6-3, 4R6-4, 4R6-5, 4R6-6, 4R6-7, 4R6-8, 4R6-9, 4R8-1, 4R8-10, 4R8-2, 4R8-3, 4R8-4, 4R8-5, 4R8-6, 4R8-7, 4R8-8, 4R8-9, 4R9-1, 4R9-10, 4R9-2, 4R9-3, 4R9-4, 4R9-5, 4R9-6, 4R9-7, 4R9-8, 4R9-9, 5R1-1, 5R1-10, 5R1-2, 5R1-3, 5R1-4, 5R1-5, 5R1-6, 5R1-7, 5R1-8, 5R1-9, 5R2-1, 5R2-10, 5R2-2, 5R2-3, 5R2-4, 5R2-5, 5R2-6, 5R2-7, 5R2-8, 5R2-9, 5R3-1, 5R3-10, 5R3-2, 5R3-3, 5R3-4, 5R3-5, 5R3-6, 5R3-7, 5R3-8, 5R3-9, R1-1, R1-10, R1-2, R1-3, R1-4, R1-5, R1-6, R1-7, R1-8, R1-9, R10-1, R10-10, R10-2, R10-3, R10-4, R10-5, R10-7, R10-8, R10-9, R11-1, R11-10, R11-2, R11-3, R11-4, R11-5, R11-6, R11-7, R11-8, R11-9, R12-1, R12-10, R12-2, R12-3, R12-4, R12-5, R12-6, R12-7, R12-8, R12-9, R13-1, R13-10, R13-2, R13-3, R13-4, R13-5, R13-6, R13-7, R13-8, R13-9, R2-1, R2-10, R2-2, R2-3, R2-4, R2-5, R2-6, R2-7, R2-8, R2-9, R3-1, R3-10, R3-2, R3-3, R3-4, R3-5, R3-6, R3-7, R3-8, R3-9, R4-1, R4-10, R4-2, R4-3, R4-4, R4-5, R4-6, R4-7, R4-8, R4-9, R5-1, R5-10, R5-2, R5-3, R5-4, R5-5, R5-6, R5-7, R5-8, R5-9, R6-1, R6-10, R6-2, R6-3, R6-4, R6-5, R6-6, R6-7, R6-8, R6-9, R7-1, R7-10, R7-2, R7-3, R7-4, R7-5, R7-6, R7-7, R7-8, R7-9, R8-1, R8-10, R8-2, R8-3, R8-4, R8-5, R8-6, R8-7, R8-8, R8-9, R9-1, R9-10, R9-2, R9-3, R9-4, R9-5, R9-6, R9-7, R9-8, R9-9
- ramp: 4R5-3, 4R5-4, 4R5-5, 4R5-6, 4R5-7, 4R5-8, 4R5-9, 4R6-1, 4R6-10, 4R6-2, 4R6-3, 4R6-4, 4R6-5, 4R6-6, 4R6-7, 4R6-8, 4R6-9, 4R8-1, 4R8-10, 4R8-2, 4R8-3, 4R8-4, 4R8-5, 4R8-6, 4R8-7, 4R8-8, 4R8-9, 4R9-1, 4R9-10, 4R9-2, 4R9-3, 4R9-4, 4R9-5, 4R9-6, 4R9-7, 4R9-8, 4R9-9, 5R1-1, 5R1-10, 5R1-2, 5R1-3, 5R1-4, 5R1-5, 5R1-6, 5R1-7, 5R1-8, 5R1-9, 5R2-1, 5R2-10, 5R2-2, 5R2-3, 5R2-4, 5R2-5, 5R2-6, 5R2-7, 5R2-8, 5R2-9, 5R3-1, 5R3-10, 5R3-2, 5R3-3, 5R3-4, 5R3-5, 5R3-6, 5R3-7, 5R3-8, 5R3-9, R1-1, R1-10, R1-2, R1-3, R1-4, R1-5, R1-6, R1-7, R1-8, R1-9, R10-1, R10-10, R10-2, R10-3, R10-4, R10-5, R10-7, R10-8, R10-9, R11-1, R11-10, R11-2, R11-3, R11-4, R11-5, R11-6, R11-7, R11-8, R11-9, R12-1, R12-10, R12-2, R12-3, R12-4, R12-5, R12-6, R12-7, R12-8, R12-9, R13-1, R13-10, R13-2, R13-3, R13-4, R13-5, R13-6, R13-7, R13-8, R13-9, R2-1, R2-10, R2-2, R2-3, R2-4, R2-5, R2-6, R2-7, R2-8, R2-9, R3-1, R3-10, R3-2, R3-3, R3-4, R3-5, R3-6, R3-7, R3-8, R3-9, R4-1, R4-10, R4-2, R4-3, R4-4, R4-5, R4-6, R4-7, R4-8, R4-9, R5-1, R5-10, R5-2, R5-3, R5-4, R5-5, R5-6, R5-7, R5-8, R5-9, R6-1, R6-10, R6-2, R6-3, R6-4, R6-5, R6-6, R6-7, R6-8, R6-9, R7-1, R7-10, R7-2, R7-3, R7-4, R7-5, R7-6, R7-7, R7-8, R7-9, R8-1, R8-10, R8-2, R8-3, R8-4, R8-5, R8-6, R8-7, R8-8, R8-9, R9-1, R9-10, R9-2, R9-3, R9-4, R9-5, R9-6, R9-7, R9-8, R9-9
