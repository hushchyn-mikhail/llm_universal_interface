# Phi-4-mini

Выгрузка от 7 октября 2026: **56 из 56 результатов** — 8 zero-shot,
8 few-shot, 32 отдельных обучения и 8 multitask. Добавлены Blood с 50%
пропусков и все результаты multitask.

- [table.md](../reports/phi/20261007/table.md) — ROC-AUC по датасетам и режимам.
- [results.csv](../reports/phi/20261007/results.csv) — все пять метрик.
- [results.json](../reports/phi/20261007/results.json) — метрики, диагностика и ссылки на артефакты.
- [environment.json](../reports/phi/20261007/environment.json) — версии пакетов.

В CSV используйте строки `status=completed`. `*_estimate` — метрика на всём
тесте; `*_bootstrap_mean` и `*_bootstrap_std` — среднее и стандартное отклонение
по 1000 bootstrap-выборкам. Это один seed обучения (42), а не несколько обучений.
`*_n_bootstrap_valid` — число выборок, на которых метрика определена.

Для бинарных задач F1, precision и recall считаются для второй метки в `labels`;
для Car — macro, ROC-AUC — macro one-vs-rest. Метрики проверены по сохранённым
предсказаниям. Предсказания и веса остаются на кластере.

Первые 47 результатов получены на V100 32 GB в FP16: batch 4 × 4 = 16.
В исходном прогоне eval batch равен 4, для few-shot и повторного прогона
1 октября — 1. Few-shot использует 64 примера; multitask — восемь датасетов без Jungle.

Blood 50% и multitask завершены в FP32 с batch 4 × 4 и eval batch 4
(задачи `4377395_0` и `4377395_1`). При обучении padding справа;
NaN/Inf в loss, градиентах или весах останавливают запуск до сохранения checkpoint.
Файлы на кластере: `/home/tmizhitskii/phi_retry_20261005/`.
Статусы и метрики — `runs/<run_name>/status.json` и `results.json`,
логи — `logs/campaign-4377395_0.out/.err` и `logs/campaign-4377395_1.out/.err`.

Для повторного запуска: `campaign_retry_20261005.json`,
`phi/configs/retry_20261005/` и `hpc/retry_fp32.sbatch` (две задачи).
Из репозитория они пишут в `runs/retry_20261005/`.
Нужны Python 3.10, `requirements-phi.txt`, данные в `assets/datasets/`
и модель в `assets/models/Phi-4-mini-instruct/`.
Команды выполняются из `timofey/`:

```bash
python hpc/run_job.py 0 --manifest campaign_retry_20261005.json --dry-run
# Уберите --dry-run для запуска; индекс 1 — multitask.
python tools/collect_results.py --manifest campaign_retry_20261005.json --output-dir exports/retry_20261005
```

`experiment.py` — отдельные задачи, `multitask.py` — совместное обучение,
`scoring.py` — расчёт вероятностей классов. Предыдущие выгрузки сохранены
в `reports/phi/20260930/` и `reports/phi/20261005/`.
