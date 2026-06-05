# Changelog

All notable changes to this project should be documented in this file

## [0.1.31] - 2026-04-12

### Changed
- Minor fixes

## [0.1.2] - 2026-04-12

### Changed
- Updated the configuration
- Reworked the duration module: calculation now supports experiment-level aggregations instead of a hard binding to campaign-level
- `build_campaign_level` replaced with the more generic `build_experiment_level`, which can aggregate at the level of any given `unit_id_col` (`user_id`, `campaign_id`, etc.)
- Updated duration planning functions to work in a more unified scheme via an aggregated dataframe
- Updated the default metrics configuration for duration planning to the new aggregated dataframe context
- Updated comments and docstrings: campaign-level wording replaced with a more general aggregated / experiment-level description

### Added
- Added new experiment-level metrics for duration planning: `life_days`, `active_days`, `entities_cnt`
- Added new helper functions for duration planning and MDE at the aggregated level
- Added the ability to run duration planning at the level of any experiment unit via `unit_id_col`

### Removed
- Removed an unused import in the duration module
- And a few other minor issues

## [0.1.1] - 2026-04-12

### Changed
- Updated the configuration

## [0.1.0] - 2026-04-12

### Added
- Initial version
