# Changelog

Все заметные изменения в проекте рекомендуется фиксировать в этом файле

## [0.1.3] - 2026-04-12

### Changed
- Мелкие исправления


## [0.1.2] - 2026-04-12

### Changed
- Обновлена конфигурация
- Переработан модуль duration: расчёт теперь поддерживает experiment-level агрегации вместо жёсткой привязки к campaign-level
- `build_campaign_level` заменён на более универсальный `build_experiment_level`, который умеет строить агрегат на уровне любого переданного `unit_id_col` (`user_id`, `campaign_id`, и др.)
- Обновлены planning-функции duration, чтобы они могли работать в более унифицированной схеме через агрегированный dataframe
- Обновлена конфигурация дефолтных метрик для duration planning под новый aggregated dataframe context
- Актуализированы комментарии и docstring'и: вместо campaign-level теперь используется более общее описание aggregated / experiment-level логики

### Added
- Добавлены новые experiment-level метрики для duration planning:`life_days`, `active_days`, `entities_cnt`
- Добавлены новые вспомогательные функции для работы с duration planning и MDE на агрегированном уровне
- Добавлена возможность считать duration planning на уровне любой экспериментальной единицы через `unit_id_col`

### Removed
- Удалён неиспользуемый import в duration-модуле.
- И еще всякие мелкие касяки


## [0.1.1] - 2026-04-12

### Changed
- Обновлена конфигурация


## [0.1.0] - 2026-04-12

### Added
- базовая версия 
