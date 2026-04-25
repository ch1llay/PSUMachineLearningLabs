"""Синтетические датасеты для лабораторной 'Интуиция ML'.

Два датасета:
- BMI: рост, вес -> худой/толстый (бинарная классификация)
- Квартиры: площадь, расстояние от центра -> цена (регрессия)

Оба датасета 2-мерные по признакам, чтобы можно было всё визуализировать.
"""

import numpy as np
import pandas as pd


def make_bmi_dataset(n: int = 200, label_noise: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """Датасет 'худой/толстый' по росту и весу.

    Признаки:
        height: рост в см, U(155, 200)
        weight: вес в кг, скоррелирован с ростом + шум

    Метка:
        is_fat: 1 если BMI > 25, иначе 0. С небольшим шумом в метках,
                чтобы не было идеальной разделимости.

    Параметры:
        n: количество примеров
        label_noise: доля случайно перевёрнутых меток (создаёт перекрытие классов)
        random_state: для воспроизводимости
    """
    rng = np.random.default_rng(random_state)

    height = rng.uniform(155, 200, size=n)
    # вес слабо скоррелирован с ростом + большой шум, чтобы BMI варьировался
    weight = 70 + 0.7 * (height - 170) + rng.normal(0, 12, size=n)
    weight = np.clip(weight, 45, 130)

    bmi = weight / (height / 100) ** 2
    is_fat = (bmi > 25).astype(int)

    # шум: случайно переворачиваем часть меток
    n_flip = int(label_noise * n)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    is_fat[flip_idx] = 1 - is_fat[flip_idx]

    return pd.DataFrame({"height": height, "weight": weight, "is_fat": is_fat})


def make_apartment_dataset(n: int = 200, noise_scale: float = 5000, random_state: int = 42) -> pd.DataFrame:
    """Датасет цен на квартиры.

    Признаки:
        area: площадь в м², U(25, 150)
        dist: расстояние от центра в км, U(0, 30)

    Истинная зависимость:
        price = 50_000 + 1500*area - 800*dist + N(0, noise_scale)

    Истинные коэффициенты (1500, -800, 50000) известны и используются
    для проверки, что обучение действительно их находит.

    Параметры:
        n: количество примеров
        noise_scale: std шума в цене (в той же валюте что и цена)
        random_state: для воспроизводимости
    """
    rng = np.random.default_rng(random_state)

    area = rng.uniform(25, 150, size=n)
    dist = rng.uniform(0, 30, size=n)
    noise = rng.normal(0, noise_scale, size=n)

    price = 50_000 + 1500 * area - 800 * dist + noise

    return pd.DataFrame({"area": area, "dist": dist, "price": price})


# Истинные коэффициенты модели цен — для проверки в notebook
APARTMENT_TRUE_COEFS = {"w_area": 1500.0, "w_dist": -800.0, "intercept": 50_000.0}
