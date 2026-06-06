# Changelog

All notable changes to this project should be documented in this file

## [0.1.40] - 2026-06-06

### Changed

- Reworked `build_ab_chat_message`: metric output logic is now closer to `build_abc_chat_message`
- Added the ability to control key metrics in `build_ab_chat_message` via the `key_metrics` parameter
- Added the ability to limit the number of key metrics via `max_metrics_per_pair`
- Added the ability to separately limit the number of additional notable metrics via `max_colored_extra_per_pair`
- Kept backward compatibility with the legacy `max_metrics` parameter: when passed, it overrides `max_metrics_per_pair`
- Updated metric sorting: key metrics are printed first in the given order, followed by additional significant or notable metrics
- Improved AB bot message formatting: an empty `metric_type` is no longer rendered as empty parentheses
- Improved handling of `p_value`, `rel_delta`, `NaN`, `None`, `inf` and numpy/pandas scalar types
- Reworked `build_dashboard_url_ab`: `extra_params` now supports `list` and `tuple` values, like the ABC version

### Added

- Added the `alpha` parameter to control the statistical significance threshold in `build_ab_chat_message`
- Added the `_safe_rel_delta_pct` helper for safe relative-delta-percent computation
- Added the `_icon_for_ab_row` helper for the extended icon logic in AB messages
- Added a fallback to the legacy sorting when no key metrics and no notable extra metrics are found
- Added protection against missing optional columns in `df_result`

### Fixed

- Fixed a crash risk in `build_ab_chat_message` when the dataframe lacks `p_value`, `rel_delta`, `metric_name` or `metric_type`
- Fixed formatting of very small p-values: values below `0.0001` are now rendered as `&lt;0,0001`
- Fixed the risk of overly long chat messages by limiting key and extra metrics separately
- Fixed a potential issue with incorrect metric rendering on empty or incomplete data

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
