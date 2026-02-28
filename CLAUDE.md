# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

Act as a Senior Python ML developer/researcher helping complete university machine learning lab assignments.

## Project Overview

PSU Machine Learning Labs — university course with 16 lab assignments. All work is in Jupyter notebooks using Python 3.14 (`.venv/`). No build system, requirements.txt, or test suite.

Core libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly.

## Running

```bash
source .venv/Scripts/activate   # Windows Git Bash
jupyter notebook lab1/lab1.ipynb
```

## Repository Layout

- `lab1/`, `lab2/`, ... — Lab notebook directories (one per lab, primary working area)
- `datasets/gtzan/features_30_sec.csv` — GTZAN music genre dataset (primary dataset for labs)
- `SPO_PGU/` — Teacher's repository with utility functions and reference datasets (not required to use)

### Assignments (tasks to implement)

Located in `машинное обучение/машинное обучение/`:

| # | Topic | Folder |
|---|-------|--------|
| 1 | Табличные данные (loading) | `1. Загрузка данных/` |
| 2 | Сэмплинг | `2. Подготовка данных/` |
| 3 | Описательная статистика | `2. Подготовка данных/` |
| 4.1-4.3 | Визуализация данных (Matplotlib, Plotly, табличные) | `2. Подготовка данных/` |
| 5 | Очистка данных (числовые) | `2. Подготовка данных/` |
| 6 | Отбор признаков | `2. Подготовка данных/` |
| 7 | Кодирование и масштабирование | `2. Подготовка данных/` |
| 9 | Линейная и логистическая регрессия | `3. Обучение/` |
| 10 | Деревья решений и случайный лес | `3. Обучение/` |
| 11 | Ансамбли — Беггинг | `3. Обучение/` |
| 12 | Ансамбли — Бустинг | `3. Обучение/` |
| 13 | Ансамбли — Стекинг | `3. Обучение/` |
| 14 | Персептрон | `3. Обучение/` |
| 15 | Кластеризация | `3. Обучение/` |
| 16 | Ассоциативные правила | `3. Обучение/` |
| — | Многоклассовая классификация | `3. Обучение/` |
| — | Оценка качества | `3. Обучение/` |

Note: Lab 8 does not exist in the course assignments.

### Examples from classmates

- **Федя (Ulchenko):** `examples/Лабы Примеры Федя/` — files named `ML_Lab{N}_Ulchenko_24VVIm1.ipynb` (Labs 1-16, Lab 8 lost)
- **Олег:** `examples/лабы примеры Олег/Машинное обучение/LW{N}/` — each lab in its own directory with data files (Labs 1-7, 9-16)

## Workflow for Each Lab

1. Read the assignment notebook from `машинное обучение/машинное обучение/` (matching lab number)
2. Study Fedya's example: `examples/Лабы Примеры Федя/ML_Lab{N}_Ulchenko_24VVIm1.ipynb`
3. Study Oleg's example: `examples/лабы примеры Олег/Машинное обучение/LW{N}/LW{N}.ipynb`
4. Create a detailed implementation plan — aim to do **better** than the examples (clearer explanations, better visualizations, more thorough analysis)
5. Implement in the corresponding lab directory (e.g. `lab2/lab2.ipynb`, `lab3/lab3.ipynb`, etc.)

## Dataset

Primary dataset: GTZAN Genre Collection (`datasets/gtzan/features_30_sec.csv`)
- 1000 samples, 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), 100 per genre
- 60 columns: filename, length, 57 numeric features, label
- Features: chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, harmony, perceptr (each mean+var), tempo, MFCCs 1-20 (each mean+var)
- Label column: `label` (genre name as string)

## Conventions

- Notebook cells should have markdown headers explaining each step in Russian
- Russian text in labels, comments, and markdown cells is expected
- Use standard scikit-learn workflow: load CSV → DataFrame → train_test_split → fit/predict → visualize
- Black formatter configured in IDE
