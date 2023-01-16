# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
