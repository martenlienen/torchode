# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.2 - 2026-11-02

### Fixed

- Both adjoint backward pass implementations now produce the correct gradients if `t_eval` is explicitly set but does not include `t_start`. The bug was that the adjoint backward passes would only propagate the gradients back to the first evaluation point and thus discount the effect between `t_start` and `t_eval[:, 0]`.
- A failing solve no longer stops the whole batch. Previously, the other solves were cut short but still reported `Status.SUCCESS` with uninitialized evaluations. Now every solve continues until it finishes or fails itself.
- Solves, that were erroneously reported as failed because they finished on the last possible step or experienced errors after they were already done, now correctly finish with `Status.SUCCESS`.
- Zero-length intervals (`t_start == t_end`) now ignore errors from step methods and controllers as they are not involved in the "solve".
- `PIDController` no longer returns NaN for zero-length intervals when `t_eval` is not set.
- Gradients through the initial step size selection are no longer NaN if any solve has a zero state or a zero-length interval.
- `BacksolveAdjoint` now computes gradients with the current parameters and buffers of the dynamics. Previously, it used the ones captured at construction, so reassigning a buffer or converting the module with, e.g., `.double()` or `.cuda()` gave wrong gradients.
- `JointBacksolveAdjoint` now raises an error if the instances in a batch do not share `t_start`, `t_end` and `t_eval`, as the documentation requires.
- `BacksolveAdjoint` and `JointBacksolveAdjoint` now compute gradients with respect to `t_start`, `t_end` and `t_eval`. They raise an error if a tensor in `args` requires gradients.
- The backward passes of `BacksolveAdjoint` and `JointBacksolveAdjoint` now also preserve the dtypes of `t` and `y`.
- `IntegralController` and `PIDController` now also apply `dt_min` and `dt_max` to the initial step.
- `JointBacksolveAdjoint` now returns the same shape for `status`, `ts` and `stats` as the other adjoints.

## 1.0.0 - 2024-09-22

After almost a year without any bugs found, we mark this as the first stable release!

## 0.2.0 - 2023-11-10

### Changed

- Removed compatibility with pytorch 1.x

## 0.1.9 - 2023-08-29

### Added

- Allow passing `dt_0` for the forward and backward pass in the backsolve adjoint

## 0.1.8 - 2023-03-25

### Changed

- Allow installation with pytorch 2.0 and recommend `torch.compile` in the readme

## 0.1.7 - 2023-03-24

### Fixed

- Ensure that `t0` and `direction` have compatible dtypes in `addcmul`

## 0.1.6 - 2023-02-15

### Fixed

- Replaced the `Status` enum with integer constants in internal code to avoid JIT
  compilation issues on some PyTorch versions

## 0.1.5 - 2023-02-01

### Fixed

- Keep dtype of `y` stable in mixed dtype solving when selecting the initial step size

## 0.1.4 - 2023-01-17

### Changed

- Clamp the timestep to the integration range even for instances that are already finished
  and in the selection of the initial step size

## 0.1.3 - 2023-01-16

### Changed

- The time steps are now clamped so that the dynamics are never evaluated outside of the
  integration range

## 0.1.2 - 2022-11-29

### Changed

- Make torchode compatible with python 3.8

## 0.1.1 - 2022-11-16

### Added

- `dt_max` option for step size controllers to set a maximum step size
