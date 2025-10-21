# -*- coding: utf-8 -*-
"""
train_model.py

Скрипт для обучения и оценки модели классификации банковских транзакций.
Решение для Актионады 2025 по направлению "IT".

Автор: Гурин Вячеслав Алексеевич

Для запуска:
1. Поместите этот скрипт в одну папку с файлом 'Daily Transactions.csv'.
2. Выполните в терминале команду: python train_model.py
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def create_features(df_input):
    """
    Функция для создания полного набора признаков из исходного DataFrame.
    """
    print("-> Шаг 2: Создание признаков...")
    df = df_input.copy()

    # 2.1. Базовые признаки и NLP
    df['Note_cleaned'] = df['Note'].fillna('')
    features_base = pd.get_dummies(df[['Amount', 'Mode', 'Income/Expense']])
    vectorizer = TfidfVectorizer(min_df=3, max_features=100)
    features_text_sparse = vectorizer.fit_transform(df['Note_cleaned'])
    features_text_df = pd.DataFrame(features_text_sparse.toarray(), columns=vectorizer.get_feature_names_out())
    features_advanced = pd.concat([features_base, features_text_df], axis=1)

    # 2.2. Признаки из дат
    df['datetime'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    for col in ['day_of_week', 'day_of_month', 'month', 'hour']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    features_date = df[['day_of_week', 'day_of_month', 'month', 'hour', 'is_weekend']]

    # 2.3. Ручные ("умные") признаки
    mode_stats = df.groupby('Mode')['Amount'].agg(['mean', 'max']).rename(columns={'mean': 'mode_mean', 'max': 'mode_max'})
    df_merged = df.merge(mode_stats, on='Mode', how='left')
    ie_stats = df.groupby('Income/Expense')['Amount'].agg(['mean', 'max']).rename(columns={'mean': 'ie_mean', 'max': 'ie_max'})
    df_merged = df_merged.merge(ie_stats, on='Income/Expense', how='left')
    df_merged['amount_vs_mode_mean'] = (df_merged['Amount'] / df_merged['mode_mean']).fillna(0)
    df_merged['amount_vs_mode_max'] = (df_merged['Amount'] / df_merged['mode_max']).fillna(0)
    df_merged['amount_vs_ie_mean'] = (df_merged['Amount'] / df_merged['ie_mean']).fillna(0)
    df_merged['amount_vs_ie_max'] = (df_merged['Amount'] / df_merged['ie_max']).fillna(0)
    features_manual = df_merged[['amount_vs_mode_mean', 'amount_vs_mode_max', 'amount_vs_ie_mean', 'amount_vs_ie_max']]

    # 2.4. Сборка финального набора признаков
    features_god_tier = pd.concat([features_advanced, features_date, features_manual], axis=1)
    print(f"Финальный набор признаков создан. Размер: {features_god_tier.shape}")
    
    return features_god_tier

def main():
    """
    Основная функция, запускающая весь пайплайн.
    """
    # Шаг 1: Загрузка данных
    print("-> Шаг 1: Загрузка данных...")
    try:
        df = pd.read_csv('Daily Transactions.csv')
        print(f"Данные успешно загружены. Размер: {df.shape}")
    except FileNotFoundError:
        print("Ошибка: Файл 'Daily Transactions.csv' не найден. Убедитесь, что он находится в той же папке, что и скрипт.")
        return

    # Шаг 2: Создание признаков
    features = create_features(df)

    # Шаг 3: Решение усложненной задачи (Category + Subcategory)
    print("\n-> Шаг 3: Решение усложненной задачи (Category + Subcategory)...")
    
    df['Subcategory_filled'] = df['Subcategory'].fillna('Unknown')
    df['Full_Category'] = df['Category'] + '_' + df['Subcategory_filled']
    
    X = features
    y = df['Full_Category']
    
    print(f"Количество уникальных классов: {y.nunique()}")

    # Очистка редких классов
    counts = y.value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        is_not_rare = ~y.isin(rare)
        X_clean = X[is_not_rare]
        y_clean = y[is_not_rare]
    else:
        X_clean = X
        y_clean = y

    # Обучение модели
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
    model.fit(X_train, y_train)

    # Оценка
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average='macro')

    print("\n--- Результаты ---")
    print(f"Итоговый Macro F1-score для усложненной задачи: {f1:.4f}\n")
    print("Детальный отчет по классификации:")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()train_model.py
