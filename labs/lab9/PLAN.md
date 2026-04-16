# План: Лабораторная работа 9 — Линейная и логистическая регрессия

## 1. Требования (из задания)

1. Загрузить выборку с Kaggle (используем GTZAN)
2. Провести анализ полученной выборки
3. Обучить линейную регрессию и оценить качество: confusion matrix + ROC-анализ (ROC-кривая, AUC)
4. Обучить логистическую регрессию и оценить качество: confusion matrix + ROC-анализ (ROC-кривая, AUC)

## 2. Чек-лист покрытия

| # | Требование | Реализация |
|---|-----------|------------|
| 1 | Загрузка | pd.read_csv GTZAN, info(), describe() |
| 2 | Анализ выборки | Распределение классов, корреляции, описательная статистика |
| 3 | Линейная регрессия | LinearRegression: бинарная задача (жанр vs остальные), порог 0.5, confusion matrix, ROC-AUC |
| 4 | Логистическая регрессия (бинарная) | LogisticRegression: бинарная задача, confusion matrix, ROC-кривая, AUC |
| 5 | Логистическая регрессия (мультикласс) | LogisticRegression(OvR): все 10 жанров, multi-class confusion matrix, OvR ROC-кривые |
| 6 | Предобработка | StandardScaler для числовых признаков, Pipeline + ColumnTransformer |
| 7 | Кросс-валидация | StratifiedKFold, сравнение моделей |
| 8 | Анализ коэффициентов | Веса линейных моделей — какие признаки важнее |

## 3. Структура ноутбука

1. Заголовок: "Лабораторная работа 9 — Линейная и логистическая регрессия"
2. Цель работы
3. **Теоретическое введение**: линейная регрессия (МНК, градиентный спуск), логистическая регрессия (сигмоида, log-loss), ROC-анализ
4. Загрузка и обзор данных GTZAN
5. Предобработка: StandardScaler, train/test split (stratified)
6. **Бинарная классификация** (classical vs rest):
   - LinearRegression как классификатор (порог 0.5)
   - LogisticRegression
   - Confusion matrix для обеих моделей
   - ROC-кривые с AUC
7. **Мультиклассовая классификация** (все 10 жанров):
   - LogisticRegression(OvR) с multi_class
   - Confusion matrix (heatmap 10×10)
   - OvR ROC-кривые по классам + micro/macro AUC
8. **Кросс-валидация**: сравнение accuracy/F1 для Linear vs Logistic
9. **Анализ коэффициентов**: top-признаки логистической регрессии
10. Выводы

## 4. Теоретические темы

- Линейная регрессия: модель y = Xw + b, метод наименьших квадратов
- Градиентный спуск, функция потерь (MSE)
- Логистическая регрессия: сигмоидная функция, log-loss
- Разница: регрессия vs классификация
- ROC-кривая: TPR vs FPR, интерпретация AUC
- Confusion matrix: TP, TN, FP, FN, precision, recall
- OvR (One-vs-Rest) для мультиклассовой задачи

## 5. Визуализации (минимум 3)

1. Barplot — распределение классов в GTZAN
2. Confusion matrix (heatmap) — LinearRegression бинарная
3. Confusion matrix (heatmap) — LogisticRegression бинарная
4. ROC-кривые — сравнение Linear vs Logistic (бинарная задача)
5. Confusion matrix (heatmap 10×10) — мультиклассовая LogisticRegression
6. OvR ROC-кривые — по каждому жанру + micro/macro AUC
7. Barplot — кросс-валидация accuracy/F1 для моделей
8. Barplot — top-20 коэффициентов логистической регрессии

## 6. Наши преимущества

| Аспект | Федя/Олег | Мы |
|--------|-----------|-----|
| Задача | Только бинарная | Бинарная + мультикласс (10 жанров) |
| ROC | Одна кривая | OvR ROC-кривые по всем классам + micro/macro AUC |
| Предобработка | Базовая | Pipeline + ColumnTransformer |
| Валидация | Один split | StratifiedKFold кросс-валидация |
| Интерпретация | Нет | Анализ коэффициентов (feature importance) |
| Визуализации | 2-3 | 8 визуализаций |
| Метрики | accuracy + AUC | accuracy, precision, recall, F1, AUC |
