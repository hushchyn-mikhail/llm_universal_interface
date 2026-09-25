# llm_universal_interface
LLM как универсальный интерфейс

Эксперименты Gemma 3 4B и Phi-4-mini:

- [Gemma: ноутбуки по датасетам и режимам](timofey/gemma/notebooks/).
- [Phi: запуск экспериментов](timofey/phi/experiment.py) и [конфиги](timofey/phi/configs/).
- [Phi: совместное обучение с исправленным scorer](timofey/phi/notebooks/multitask_corrected.ipynb).
- [Подготовка данных и моделей](timofey/data/) и [зависимости](timofey/requirements.txt).

Исторический код сохранён в `timofey/phi/legacy/` и
`timofey/phi/notebooks/multitask.ipynb`. В нём метки Heart `0` и `1`
оценивались по одинаковому токену пробела, что давало постоянные вероятности
0.5. Исправленные варианты оценивают полные метки и отклоняют переполнение
контекста. Новые значения ROC-AUC требуют повторного запуска.

Пример запуска из `timofey/` после установки зависимостей:

```bash
python data/prepare_datasets.py --skip-existing
python data/prepare_models.py phi
python phi/experiment.py --config phi/configs/phi4_heart_zero_shot.json
```

Для доступа к Gemma используйте `hf auth login` или переменную `HF_TOKEN`.
Данные, веса и токены не включаются в Git.
