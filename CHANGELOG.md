# Changelog

All notable changes to this project should be documented in this file

## [0.3.0] - 2026-11-12

### Added

- `stat_tests`: a library of two-sample tests returning a single `StatTestResult` — Welch, Student, z-test, Yuen (trimmed means), bootstrap (mean / median / quantile), permutation, Mann–Whitney, Brunner–Munzel, Kolmogorov–Smirnov, Cramér–von Mises, Mood; proportions — z-test, chi-square (with and without Yates), G-test, Fisher, Boschloo, Barnard; ratio metrics — delta method, linearization, unit-level bootstrap; paired samples — paired t, Wilcoxon, sign test, McNemar; TOST and non-inferiority; Poisson rate test
- `multi_group`: omnibus tests (ANOVA, Welch ANOVA, Kruskal–Wallis, chi-square homogeneity, Levene / Brown–Forsythe, Friedman) and post-hoc procedures — Dunn (tie-corrected, with pre-planned pairs), Tukey HSD, Games–Howell, Dunnett
- `multiple_testing.adjust_pvalues`: Bonferroni, Sidak, Holm, Holm–Sidak, Hochberg, Hommel, Benjamini–Hochberg, Benjamini–Yekutieli
- `effect_sizes`: Cohen's d, Hedges' g, Glass's delta, Cliff's delta, Vargha–Delaney A, Cohen's h, relative-lift confidence intervals (delta method and Fieller)
- `variance_reduction`: CUPED (`cuped_adjust`, `add_cuped_column`), pooled-threshold winsorization, post-stratification
- `diagnostics`: SRM (`srm_check`, `srm_check_df`), unit overlap and duplicate checks, per-group descriptive statistics, normality report, test recommendation (`recommend_test`), A/A and A/B simulation (`simulate_aa`)
- `sequential`: group sequential boundaries (Lan–DeMets: O'Brien–Fleming, Pocock) and mSPRT with always-valid p-values and confidence intervals
- `bayesian`: Beta-Binomial for proportions and a normal approximation for means — probability to win, expected loss
- `power`: sample size, MDE and power for means, proportions and ratio metrics (unequal groups, one-sided alternative, correction for the number of comparisons)
- A/B/C/D experiments: `run_abcd_test` compares all six pairs (`include_all_pairs=False` keeps control pairs only), with `style_table_abcd`, `make_synthetic_abcd_dataset`, `export_abcd_results_to_excel`, `build_abcd_chat_message`
- `run_multigroup_test` for any number and naming of groups; `run_omnibus_test` for a per-metric omnibus test
- Optional metrics config columns in `run_ab_test` / `run_abc_test` / `run_abcd_test`: `test` (a test instead of the default), `covariate` (CUPED), `winsorize` (upper quantile)
- `extended=True` appends `EXTENDED_COLUMNS`: test name, statistic, effect size, relative-lift CI, CUPED delta and variance reduction, winsorization threshold
- `run_ab_test(metrics_adjust_method=...)`: multiple-testing correction across metrics (`p_value_adj`, `result_adj`)
- `run_ab_test` now validates the metrics config and input columns up front and raises a descriptive `ValueError` for missing columns (previously an opaque pandas `KeyError`), matching the existing ABC behavior
- Tests: every test checked against scipy / statsmodels references, group sequential boundaries against published tables, mSPRT type I error under continuous peeking, pipeline options, config validation, duration planning, reporting
- Example notebook: A/B/C/D section and a section on test selection, CUPED, diagnostics, sequential testing, Bayesian summaries and sample size

### Fixed

- `additive` metrics: `abs_delta` and `rel_delta` were computed as the difference of raw sums, so with unequal group sizes they reflected the size difference rather than the effect (the sign could disagree with the difference in means the p-value is based on, giving a wrong `result`). The experiment sum is now scaled to the control size (`n_control / n_experiment`); `value_base` and `value_exp` remain raw sums. Values are unchanged for equal groups
- ABC / ABCD: a zero delta with a significant p-value (for example a rank test with equal medians) was labelled `negative`; it is now `neutral`, as in AB
- Excel export (AB and ABC / ABCD): `power_now` (a fraction between 0 and 1) was written with the `0.0"%"` format and showed 0.8 as "0.8%"; it now uses a percent format (80.0%)
- `run_ab_test`: metric type and direction are lower-cased, as in ABC

- `add_cuped_column` failed on pandas 3 with "assignment destination is read-only" (copy-on-write returns read-only arrays)
- Declared the dependencies of styled notebook tables: new `style` extra (`jinja2`, `matplotlib`), also included in `dev`; previously `style_table_*` failed with an `ImportError` on a clean install
- CI lint with the current ruff: narrowed blind `except Exception` clauses, removed redundant casts, sorted imports

### Changed

- Without the new columns and parameters the results of `run_ab_test` and `run_abc_test` are identical to 0.1.50, except for the fixes above
- The pairwise engine in `abc.py` is generalized to any number of groups; `run_abc_test` keeps its behavior and signature
- `build_abc_chat_message`: the pair order is derived from the data, a `test_label` parameter sets the message title
- Excel export creates a sheet per pair found in the result
- `pvalue_adjust_method` accepts short aliases (`bh`, `by`, `hochberg`)
- Synthetic datasets gained an `amount_pre` column (pre-experiment spend) for CUPED demos
- Replaced the deprecated `Styler.applymap` with `Styler.map` in `style_table_ab` (pandas requirement raised to `>=2.1`); removes a `FutureWarning` on modern pandas
- Removed unused imports; cleaned up `.gitignore` and package metadata (keywords, classifiers, issues URL)

## [0.1.50] - 2026-08-18

### Fixed

- Fixed the confidence interval of ratio metrics: the delta method is now used with the variance and denominator mean of each group
- For share metrics the Wald interval was replaced with the Newcombe-Wilson interval, which works correctly at 0% and 100% shares
- For median metrics a reproducible BCa bootstrap confidence interval was added, with a percentile fallback for degenerate distributions
- In AB the p-value is no longer rounded before the statistical significance decision

### Changed

- In ABC the confidence intervals became simultaneous: a Bonferroni correction over the number of compared pairs is applied
- Added the `ci_bootstrap_resamples` and `ci_random_state` parameters to AB and ABC for tuning the median bootstrap interval
- The `ci_lower` and `ci_upper` columns are now explicitly placed after `power_now` in the AB Excel report

### Added

- Added tests for the confidence interval formulas, all metric types, ABC correction and Excel export

## [0.1.41] - 2026-06-27

### Added

- Added confidence intervals (`ci_lower`, `ci_upper`) for the difference between groups in AB and ABC tests
- CI is computed for all metric types except `median` (NaN)
- Calculation method: Welch's t-interval for additive/average/ratio metrics, Wald z-interval for share metrics
- Added the `_welch_df()` helper for Welch-Satterthwaite degrees of freedom
- `ci_lower`, `ci_upper` columns added to AB and ABC Excel exports with `#,##0.0000` formatting
- `ci_lower`, `ci_upper` columns added to styled DataFrame tables

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
