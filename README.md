# llm_universal_interface
LLM как универсальный интерфейс

Эксперименты Gemma 3 4B и Phi-4-mini:

- [Gemma: ноутбуки по датасетам и режимам](timofey/gemma/notebooks/).
- [Phi: результаты, код и запуск](timofey/phi/README.md).
- [Phi: таблица ROC-AUC](timofey/reports/phi/20260930/table.md) и [все метрики в CSV](timofey/reports/phi/20260930/results.csv).
- [Подготовка данных и моделей](timofey/data/) и [зависимости](timofey/requirements.txt).

Для Phi опубликованы 43 результата повторного прогона, выгрузка от 30 сентября 2026.
Состав результатов и смысл столбцов описаны в инструкции выше.

Для доступа к Gemma используйте `hf auth login` или переменную `HF_TOKEN`.
Данные, веса и токены не включаются в Git.
