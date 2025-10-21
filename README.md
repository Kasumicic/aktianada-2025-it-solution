# Решение задачи классификации транзакций для Актионады 2025

**Автор:** Гурин Вячеслав Алексеевич

Этот репозиторий содержит полное решение задачи №3 «Бухгалтерия (задача на классификацию)» в рамках IT-Актионады 2025 на основе датасета [USA Banking Transactions Dataset](https://www.kaggle.com/datasets/pradeepkumar2424/usa-banking-transactions-dataset-2023-2024).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10ijxK6Iu2RKzAhjNkVs9ioO5HwXs5IZn)

---

### Структура проекта:

*   **`Aktianada_2025_IT_Solution_Gurin.ipynb`**: Основной файл проекта. Это Jupyter Notebook, который содержит пошаговое исследование, код, выводы и визуализации. Он предназначен для демонстрации и объяснения хода работы. GitHub отрисовывает его как статичную веб-страницу, но вы также можете открыть его интерактивно в Google Colab, нажав на значок выше.

*   **`train_model.py`**: Чистый Python-скрипт для автоматического обучения и оценки финальной модели. Он предназначен для воспроизведения результата и демонстрирует готовность решения к внедрению в продакшн.

### Как воспроизвести результат:

1.  Скачайте датасет `Daily Transactions.csv` с [**Kaggle**](https://www.kaggle.com/datasets/pradeepkumar2424/usa-banking-transactions-dataset-2023-2024).
2.  Поместите скрипт `train_model.py` и файл с данными `Daily Transactions.csv` в одну папку.
3.  Выполните в вашем терминале команду:
    ```bash
    python train_model.py
    ```
