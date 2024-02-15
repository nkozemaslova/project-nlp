## Цель проекта
Предсказание оценки банка клиентом по его текстовому отзыву о работе банка. 

## Перечень команд для запуска проекта 
1. Склонировать репозиторий: `git clone https://github.com/nkozemaslova/nlp_project.git`.
2. Создать и активировать виртуальное окружение: `python3 -m venv .venv`, `source .venv/bin/activate`.
3. Установить poetry и pre-commit: `pip install poetry`, `pip install pre-commit`.
4. `poetry install`
5. `pre-commit install`
6. `dvc pull`
7. Запустить файл train.py: `python train.py`
8. Запустить файл infer.py: `python infer.py`

Результат работы проекта - обученная модель model.bin и файл с предсказаниями predictions.csv. 

## Подробное описание проекта 
- Тренировочные данные представлены в формате csv, содержат наименование банка, текст отзыва, оценку клиента от 1 до 5, временную метку. 
- Датасет был расширен посредством генерации дополнительных временных признаков.
- Преобработка текстов отзывов представлена в файле preprocessing.py
- Использованная модель - CatBoostClassifier с параметрами: iterations=100, learning_rate=0.009, eval_metric='MultiClass'.

