# Phi-4-mini

Результаты выгружены 30 сентября 2026. Готовы 43 из 56 комбинаций:
8 zero-shot, 8 few-shot и 27 отдельных обучений. Нет результатов для
Blood с 50% пропусков, Car с 0% и 50%, California с 0% и 90%, а также
для всех восьми датасетов multitask. Эти задачи отправлены на повторный
запуск 1 октября; их результаты в эту выгрузку не входят.

- [table.md](../reports/phi/20260930/table.md) — ROC-AUC по датасетам и режимам.
- [results.csv](../reports/phi/20260930/results.csv) — ROC-AUC, F1, accuracy, precision и recall.
- [results.json](../reports/phi/20260930/results.json) — те же результаты, диагностика и ссылки на артефакты запусков.
- [environment.json](../reports/phi/20260930/environment.json) — версии пакетов на кластере.

В CSV берите строки с `status=completed`. `*_estimate` — метрика на всей
тестовой выборке, `*_bootstrap_mean` и `*_bootstrap_std` — среднее и стандартное
отклонение по 1000 bootstrap-выборкам из test. `*_n_bootstrap_valid` — число
выборок, на которых метрика определена. Это один seed обучения (42);
bootstrap не измеряет разброс между повторными обучениями.
Для бинарных задач F1, precision и recall считаются для второй метки в `labels`,
для Car — macro; ROC-AUC Car — macro one-vs-rest. Пустая ячейка означает,
что результата нет. Значения метрик проверены по сохранённым предсказаниям;
сами предсказания и веса в Git не включены.

Все few-shot используют 64 примера. Multitask включает восемь датасетов без
Jungle. Вместо A100 использовалась V100 32 GB с FP16: microbatch 4 и накопление
градиентов за 4 шага, эффективный batch 16. В исходном прогоне eval batch равен
4 для zero-shot и fine-tuning, 1 для few-shot; для шести повторных задач
он уменьшен до 1 после ошибок с нечисловыми
оценками классов. Обучение в повторном прогоне начинается с базовой модели.

Код: `experiment.py` — отдельные задачи, `multitask.py` — совместное обучение,
`scoring.py` — оценка полной последовательности токенов метки. Старые конфиги
в корне `configs/`, `legacy/` и ноутбуки сохранены как исторические версии.
Для новых запусков используйте `configs/article/` или `configs/retry_20261001/`.

Команды ниже выполняются из `timofey/`, в Python 3.10 с зависимостями из
`requirements-phi.txt`. Данные должны лежать в `assets/datasets/`, модель —
в `assets/models/Phi-4-mini-instruct/`. Скрипты подготовки —
`data/prepare_datasets.py` и `data/prepare_models.py`.

```bash
# Проверить команду одной задачи, не запуская модель
python hpc/run_job.py 0 --manifest campaign_20260927.json --dry-run
# Запустить эту задачу: та же команда без --dry-run

# Собрать доступные результаты исходного прогона
python tools/collect_results.py --manifest campaign_20260927.json --output-dir exports/article
```

Slurm-скрипты `hpc/campaign.sbatch` и `hpc/retry.sbatch` рассчитаны на 49 и
6 задач соответственно; перед `sbatch` создайте `logs/` и проверьте account,
partition и доступные GPU. Логи Slurm — `logs/campaign-JOBID_INDEX.out/.err`.
В `runs/<run_name>/` сохраняются `status.json`, `results.json`, `manifest.json`
и предсказания `batches/*.npz`. Повторные задачи пишут в `runs/retry_20261001/`.
Collector отдельно помечает завершённые, упавшие и отсутствующие результаты.
