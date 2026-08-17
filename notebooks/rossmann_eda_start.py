"""
Rossmann Store Sales — стартовий скрипт для завантаження та первинного EDA
=============================================================================

Перед запуском:
1. Завантажте дані з Kaggle: https://www.kaggle.com/c/rossmann-store-sales/data
   (потрібен безкоштовний акаунт Kaggle; натисніть "Download All")
2. Розпакуйте архів у папку поруч зі скриптом, наприклад: ./data/
   Має бути 3 файли: train.csv, test.csv, store.csv
3. Встановіть бібліотеки (якщо ще не встановлені):
   pip install pandas numpy matplotlib seaborn

Запуск: python rossmann_eda_start.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. Налаштування шляхів до файлів — змініть, якщо дані лежать в іншому місці
# ---------------------------------------------------------------------------
DATA_DIR = "data"  # тека з train.csv, test.csv, store.csv

train_path = f"{DATA_DIR}/train.csv"
store_path = f"{DATA_DIR}/store.csv"

# ---------------------------------------------------------------------------
# 2. Завантаження даних
# ---------------------------------------------------------------------------
print("Завантажуємо дані...")

train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
store = pd.read_csv(store_path)

print(f"train.csv: {train.shape[0]:,} рядків, {train.shape[1]} колонок")
print(f"store.csv: {store.shape[0]:,} рядків, {store.shape[1]} колонок")

# ---------------------------------------------------------------------------
# 3. Об'єднання таблиць — щоб продажі мали контекст по кожному магазину
# ---------------------------------------------------------------------------
df = train.merge(store, on="Store", how="left")
print(f"\nПісля об'єднання: {df.shape[0]:,} рядків, {df.shape[1]} колонок")

# ---------------------------------------------------------------------------
# 4. Первинний огляд даних
# ---------------------------------------------------------------------------
print("\n--- Перші 5 рядків ---")
print(df.head())

print("\n--- Типи даних та пропуски ---")
print(df.info())

print("\n--- Кількість пропущених значень по колонках ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Базова статистика по числових колонках ---")
print(df.describe())

# ---------------------------------------------------------------------------
# 5. Прибираємо дні, коли магазин був закритий (продажі = 0, це не помилка)
# ---------------------------------------------------------------------------
open_df = df[df["Open"] == 1].copy()
print(f"\nЗакритих днів: {(df['Open'] == 0).sum():,}")
print(f"Відкритих днів (для аналізу): {open_df.shape[0]:,}")

# ---------------------------------------------------------------------------
# 6. Приклади базового аналізу мережі магазинів
# ---------------------------------------------------------------------------

# 6.1 Середні продажі за типом магазину
print("\n--- Середні продажі за типом магазину (StoreType) ---")
print(open_df.groupby("StoreType")["Sales"].mean().sort_values(ascending=False))

# 6.2 Середні продажі за типом асортименту
print("\n--- Середні продажі за типом асортименту (Assortment) ---")
print(open_df.groupby("Assortment")["Sales"].mean().sort_values(ascending=False))

# 6.3 Вплив промо-акцій на продажі
print("\n--- Вплив промо (Promo) на середні продажі ---")
print(open_df.groupby("Promo")["Sales"].mean())

# 6.4 Топ-10 магазинів за сумарними продажами
print("\n--- Топ-10 магазинів за загальними продажами ---")
top_stores = open_df.groupby("Store")["Sales"].sum().sort_values(ascending=False).head(10)
print(top_stores)

# ---------------------------------------------------------------------------
# 7. Візуалізації
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 Розподіл продажів
axes[0, 0].hist(open_df["Sales"], bins=50, color="steelblue")
axes[0, 0].set_title("Розподіл щоденних продажів")
axes[0, 0].set_xlabel("Продажі")
axes[0, 0].set_ylabel("Кількість днів")

# 7.2 Продажі за типом магазину
sns.boxplot(data=open_df, x="StoreType", y="Sales", ax=axes[0, 1])
axes[0, 1].set_title("Продажі за типом магазину")

# 7.3 Динаміка середніх продажів по місяцях
monthly = open_df.copy()
monthly["Month"] = monthly["Date"].dt.to_period("M")
monthly_avg = monthly.groupby("Month")["Sales"].mean()
axes[1, 0].plot(monthly_avg.index.astype(str), monthly_avg.values, marker="o")
axes[1, 0].set_title("Середні продажі по місяцях (вся мережа)")
axes[1, 0].tick_params(axis="x", rotation=90)

# 7.4 Продажі за днем тижня
dow_avg = open_df.groupby("DayOfWeek")["Sales"].mean()
axes[1, 1].bar(dow_avg.index, dow_avg.values, color="coral")
axes[1, 1].set_title("Середні продажі за днем тижня (1=Пн ... 7=Нд)")
axes[1, 1].set_xlabel("День тижня")

plt.tight_layout()
plt.savefig("rossmann_eda_overview.png", dpi=150)
print("\nГрафіки збережено у файл: rossmann_eda_overview.png")

print("\nГотово! Це базовий старт — далі можна поглиблюватись у:")
print("- вплив відстані до конкурентів (CompetitionDistance)")
print("- ефект тривалих промо-кампаній (Promo2)")
print("- сезонність навколо свят (StateHoliday, SchoolHoliday)")
print("- прогнозування продажів (наприклад, Prophet, XGBoost)")
