# План: Лабораторная работа 13 — Ансамбли: стекинг

## 1. Требования

1. Повторить эксперимент (стекинг)
2. Классификация на датасете
3. Confusion matrix + ROC-анализ
4. Посмотреть коэффициенты LogReg финальной модели — какой классификатор важнее
5. Попробовать самостоятельно разные комбинации

## 2. Структура ноутбука

1. Теория: стекинг, VotingClassifier (hard/soft), StackingClassifier
2. Загрузка GTZAN, предобработка
3. Базовые модели (DT, KNN, LR, RF, GradientBoosting)
4. VotingClassifier (hard + soft)
5. StackingClassifier (мета-модель: LogisticRegression)
6. Confusion matrix + ROC
7. Анализ коэффициентов мета-модели
8. Кросс-валидация
9. Выводы

## 3. Визуализации

1. Barplot — все модели (базовые + voting + stacking)
2. Confusion matrix лучшего ансамбля
3. ROC-кривые
4. Barplot — коэффициенты мета-модели
5. Кросс-валидация (boxplot)
