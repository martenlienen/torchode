# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
