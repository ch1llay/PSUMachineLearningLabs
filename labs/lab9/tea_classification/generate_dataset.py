"""
Генератор синтетического датасета для классификации китайского чая.

Создаёт tea_dataset.csv с 1000 примерами (250 на каждый из 4 классов).

Классы:
    - Тегуань инь (улун, лёгкий)
    - Шу пуэр (ферментированный, тёмный)
    - Да Хун Пао (улун, средний)
    - Шэн пуэр (сырой, сильный)

Признаки:
    - caffeine  (мг/г, ~3-4)
    - theanine  (мг/г, ~5-15)
    - tannins   (%, ~10-20)
    - catechins (мг/г, ~40-150)
    - color     (шкала 1-10, от светло-зелёного до тёмно-красного)

Запуск:
    python labs/lab9/tea_classification/generate_dataset.py

Датасет сохраняется рядом со скриптом в tea_dataset.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PER_CLASS = 250
FEATURES = ["caffeine", "theanine", "tannins", "catechins", "color"]

# (mean, std) по каждому признаку для каждого класса
CLASS_PARAMS = {
    "Тегуань инь": {
        "caffeine":  (3.0, 0.30),
        "theanine":  (12.0, 1.8),
        "tannins":   (10.0, 2.0),
        "catechins": (120.0, 15.0),
        "color":     (3.0, 1.0),
    },
    "Шу пуэр": {
        "caffeine":  (3.5, 0.30),
        "theanine":  (6.0, 1.8),
        "tannins":   (20.0, 2.0),
        "catechins": (40.0, 15.0),
        "color":     (9.0, 1.0),
    },
    "Да Хун Пао": {
        "caffeine":  (3.8, 0.30),
        "theanine":  (9.0, 1.8),
        "tannins":   (14.0, 2.0),
        "catechins": (80.0, 15.0),
        "color":     (6.0, 1.0),
    },
    "Шэн пуэр": {
        "caffeine":  (4.2, 0.30),
        "theanine":  (11.0, 1.8),
        "tannins":   (19.0, 2.0),
        "catechins": (140.0, 15.0),
        "color":     (4.0, 1.0),
    },
}


def generate_class_samples(n, params, rng):
    """Сгенерировать n примеров одного класса по заданным (mean, std) каждого признака."""
    samples = {}
    for feature in FEATURES:
        mean, std = params[feature]
        samples[feature] = rng.normal(mean, std, size=n)
    return pd.DataFrame(samples)


def main():
    rng = np.random.default_rng(SEED)

    # Собираем данные по классам
    parts = []
    for label, params in CLASS_PARAMS.items():
        df_class = generate_class_samples(N_PER_CLASS, params, rng)
        df_class["label"] = label
        parts.append(df_class)

    df = pd.concat(parts, ignore_index=True)

    # Обрезка граничных значений (не округляем — признаки вещественные)
    for f in ["caffeine", "theanine", "tannins", "catechins"]:
        df[f] = df[f].clip(lower=0)
    df["color"] = df["color"].clip(lower=1, upper=10)

    # Перемешиваем строки
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Сохраняем рядом со скриптом
    out_path = Path(__file__).parent / "tea_dataset.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Отчёт
    print(f"Датасет сохранён: {out_path}")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
    print()
    print("Баланс классов:")
    print(df["label"].value_counts())
    print()
    print("Первые 5 строк:")
    print(df.head().to_string(index=False))
    print()
    print("Статистики признаков:")
    print(df[FEATURES].describe().round(2).to_string())


if __name__ == "__main__":
    main()
