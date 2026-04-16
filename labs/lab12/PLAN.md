# План: Лабораторная работа 12 — Ансамбли: бустинг

## 1. Требования

1. Повторить эксперимент лабораторной работы (бустинг)
2. Провести классификацию на наборе данных
3. Оценить качество: confusion matrix + ROC-анализ

## 2. Чек-лист покрытия

| # | Требование | Реализация |
|---|-----------|------------|
| 1 | AdaBoost | AdaBoostClassifier на GTZAN |
| 2 | Gradient Boosting | GradientBoostingClassifier на GTZAN |
| 3 | HistGradientBoosting | HistGradientBoostingClassifier (быстрый аналог XGBoost/LightGBM) |
| 4 | Confusion matrix | Для лучшей модели |
| 5 | ROC-анализ | Micro-average ROC для всех моделей |
| 6 | learning_rate анализ | Влияние lr на ошибку |
| 7 | Feature importance | Из GradientBoosting |
| 8 | Кросс-валидация | Сравнение всех бустинг-методов |

## 3. Структура ноутбука

1. Заголовок + цель
2. Теория: AdaBoost, градиентный бустинг, learning rate
3. Загрузка GTZAN, предобработка
4. AdaBoost, GradientBoosting, HistGradientBoosting — обучение
5. Confusion matrix лучшей модели
6. ROC-кривые — сравнение
7. Влияние learning_rate
8. Feature importance
9. Кросс-валидация
10. Выводы

## 4. Визуализации

1. Barplot — сравнение accuracy трёх методов бустинга
2. Confusion matrix (heatmap)
3. ROC-кривые
4. Learning rate vs accuracy
5. Feature importance (barplot)
6. Кросс-валидация (boxplot)

## 5. Наши преимущества

- 3 метода бустинга включая HistGradientBoosting (аналог XGBoost)
- Анализ learning_rate
- Мультикласс (10 жанров)
- Кросс-валидация
