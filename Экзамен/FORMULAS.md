# Формулы к экзамену по МО — шпаргалка

> Компаньон к `ANSWERS.md`. Формат: только формулы + 1–2 строки контекста. Для повторения в последний день.
> Нотация: $X \in \mathbb{R}^{N \times d}$ — данные, $N$ — объекты, $d$ — признаки, $y$ — метки, $w, \theta$ — параметры.
> Горбаченковская нотация в скобках: $\ell$ вместо $N$, $n$ вместо $d$, $F$ вместо $X$, $X^\ell$ — выборка, $Q$ — эмпирический риск.

---

## 0. Базовая статистика

Среднее: $\bar x = \frac{1}{N}\sum_{i=1}^N x_i$

Дисперсия (выборочная, несмещ.): $s^2 = \frac{1}{N-1}\sum (x_i - \bar x)^2$

Ковариация: $\text{Cov}(X, Y) = \frac{1}{N-1}\sum (x_i - \bar x)(y_i - \bar y)$

Корреляция Пирсона: $r = \frac{\text{Cov}(X,Y)}{s_X s_Y} \in [-1, 1]$

z-score: $z_i = \frac{x_i - \bar x}{s}$

Межквартильный размах: $\text{IQR} = Q_3 - Q_1$. Выбросы по правилу Тьюки: $x < Q_1 - 1.5\,\text{IQR}$ или $x > Q_3 + 1.5\,\text{IQR}$.

---

## 1. Подготовка данных

### Масштабирование

Min-Max: $x' = \frac{x - \min}{\max - \min} \in [0, 1]$

Standard (z-score): $x' = \frac{x - \mu}{\sigma}$

Robust: $x' = \frac{x - \text{median}}{\text{IQR}}$

Max-abs: $x' = x / \max(|x|) \in [-1, 1]$

Нормализация L2 по строке: $x' = x / \|x\|_2$

### Кодирование

One-Hot: категория $c \in \{1, \dots, K\}$ → вектор длины $K$ с единицей на позиции $c$.

Target Encoding (регуляр.): $\text{enc}(c) = \frac{n_c \bar y_c + m \bar y}{n_c + m}$, $m$ — сила сглаживания.

### PCA (главные компоненты)

Центрирование: $X_c = X - \bar X$

Ковариация: $\Sigma = \frac{1}{N-1} X_c^\top X_c$

Разложение: $\Sigma = V \Lambda V^\top$, $\Lambda = \text{diag}(\lambda_1 \ge \dots \ge \lambda_d)$.

Проекция: $Z = X_c V_k$, $V_k \in \mathbb{R}^{d \times k}$.

Доля объяснённой дисперсии: $\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}$

### Mutual Information / Entropy

Энтропия: $H(X) = -\sum_x p(x) \log p(x)$

Условная: $H(Y|X) = -\sum_{x,y} p(x,y) \log p(y|x)$

MI: $I(X; Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)$

---

## 2. Регрессия

### Линейная (OLS)

Модель: $\hat y = w^\top x + b$

Функционал: $Q(w) = \frac{1}{N}\sum (y_i - w^\top x_i - b)^2 = \frac{1}{N}\|y - Xw\|^2$

Аналитич. решение (нормальное уравнение): $\hat w = (X^\top X)^{-1} X^\top y$

Через псевдообращение: $\hat w = X^+ y$, $X^+ = (X^\top X)^{-1} X^\top$.

Градиент: $\nabla_w Q = \frac{2}{N} X^\top (Xw - y)$

### Регуляризованная

Ridge (L2): $\min_w \|y - Xw\|^2 + \lambda \|w\|_2^2$, $\hat w = (X^\top X + \lambda I)^{-1} X^\top y$

Lasso (L1): $\min_w \|y - Xw\|^2 + \lambda \|w\|_1$ (нет closed-form; координатный спуск).

Elastic Net: $\min_w \|y - Xw\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$.

### Метрики регрессии

MSE: $\frac{1}{N}\sum (y_i - \hat y_i)^2$

RMSE: $\sqrt{\text{MSE}}$

MAE: $\frac{1}{N}\sum |y_i - \hat y_i|$

MAPE: $\frac{100\%}{N}\sum \left|\frac{y_i - \hat y_i}{y_i}\right|$

$R^2$: $1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$

Adjusted $R^2$: $1 - (1 - R^2)\frac{N-1}{N-d-1}$

### Логистическая регрессия

Сигмоида: $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$

Модель: $p(y=1|x) = \sigma(w^\top x + b)$

Правдоподобие (BCE): $Q(w) = -\frac{1}{N}\sum [y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)]$

Градиент: $\nabla_w Q = \frac{1}{N} X^\top (\hat p - y)$

Softmax (многоклассовая): $p(y=k|x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}$

Cross-entropy (многоклас.): $Q = -\frac{1}{N}\sum_i \sum_k y_{ik} \log \hat p_{ik}$

---

## 3. Классификация: расстояния, байес, kNN

### Расстояния

Евклидово: $d_2(x, y) = \sqrt{\sum (x_j - y_j)^2}$

Манхэттен: $d_1(x, y) = \sum |x_j - y_j|$

Минковский: $d_p(x, y) = \left(\sum |x_j - y_j|^p\right)^{1/p}$

Чебышёва: $d_\infty = \max_j |x_j - y_j|$

Косинусное: $d_{\cos} = 1 - \frac{x^\top y}{\|x\|\|y\|}$

Махаланобис: $d_M(x, y) = \sqrt{(x-y)^\top \Sigma^{-1} (x-y)}$

### Байес

Формула: $P(y|x) = \frac{P(x|y)P(y)}{P(x)}$

Байесовский классификатор: $\hat y = \arg\max_c P(c) P(x|c)$

Naive Bayes: $P(x|c) = \prod_j P(x_j | c)$

Гауссовский NB: $P(x_j|c) = \mathcal{N}(x_j; \mu_{jc}, \sigma_{jc}^2)$

Мультиномиальный NB: $P(x|c) \propto \prod_j p_{jc}^{x_j}$

Лапласово сглаживание: $\hat p_{jc} = \frac{n_{jc} + \alpha}{n_c + \alpha V}$

### kNN

$\hat y(x) = \arg\max_c \sum_{i \in N_k(x)} w_i \mathbb{1}[y_i = c]$

Веса: $w_i = 1$ (uniform) или $w_i = 1/d(x, x_i)$ (distance).

Для регрессии: $\hat y = \frac{\sum w_i y_i}{\sum w_i}$

---

## 4. Деревья решений

### Критерии разбиения

Энтропия: $H(S) = -\sum_c p_c \log_2 p_c$

Information Gain: $\text{IG}(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$

Gini: $\text{Gini}(S) = 1 - \sum_c p_c^2 = \sum_c p_c(1-p_c)$

Classification error: $E(S) = 1 - \max_c p_c$

MSE (для регрессии): $\text{MSE}(S) = \frac{1}{|S|}\sum (y_i - \bar y_S)^2$

### Cost-Complexity Pruning (CART)

$C_\alpha(T) = \sum_{t \in \text{leaves}} Q(t) + \alpha |T|$

$|T|$ — число листьев, $\alpha$ — параметр сложности.

---

## 5. SVM

### Жёсткий зазор

Гиперплоскость: $w^\top x + b = 0$

Условие: $y_i(w^\top x_i + b) \ge 1$

Задача: $\min_{w, b} \frac{1}{2}\|w\|^2$ при $y_i(w^\top x_i + b) \ge 1$

Ширина зазора: $\frac{2}{\|w\|}$

### Мягкий зазор

$\min \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$ при $y_i(w^\top x_i + b) \ge 1 - \xi_i, \xi_i \ge 0$

Hinge loss: $\mathcal{L} = \max(0, 1 - y_i(w^\top x_i + b))$

### Двойственная задача

$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$

$0 \le \alpha_i \le C$, $\sum \alpha_i y_i = 0$

Решение: $w = \sum_i \alpha_i y_i x_i$; $\hat y(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$

### Ядра

Линейное: $K(x, y) = x^\top y$

Полиномиальное: $K(x, y) = (x^\top y + c)^d$

RBF (Гауссово): $K(x, y) = \exp(-\gamma \|x - y\|^2)$

Сигмоидное: $K(x, y) = \tanh(\gamma x^\top y + c)$

---

## 6. Кластеризация

### K-means

Функционал: $Q = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$

Обновление центра: $\mu_k = \frac{1}{|C_k|}\sum_{x \in C_k} x$

### Метрики качества

Silhouette: $s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1]$, $a_i$ — ср. расстояние внутри кластера, $b_i$ — до ближайшего чужого.

Davies-Bouldin: $\text{DB} = \frac{1}{K}\sum_i \max_{j \ne i} \frac{s_i + s_j}{d(\mu_i, \mu_j)}$

Inertia (within-cluster sum of squares): $\text{WCSS} = \sum_k \sum_{x \in C_k} \|x - \mu_k\|^2$

### Иерархическая

Single link: $d(A, B) = \min_{a \in A, b \in B} d(a, b)$

Complete link: $d(A, B) = \max d(a, b)$

Average link: $d(A, B) = \frac{1}{|A||B|}\sum d(a, b)$

Ward: минимизация роста WCSS при слиянии.

### DBSCAN

Параметры: $\varepsilon$ (радиус), minPts (минимум соседей).

Core-point: $|N_\varepsilon(x)| \ge \text{minPts}$.

---

## 7. Метрики классификации

Матрица ошибок: TP, TN, FP, FN.

Accuracy = $\frac{TP + TN}{TP+TN+FP+FN}$

Precision = $\frac{TP}{TP + FP}$

Recall (TPR, Sensitivity) = $\frac{TP}{TP + FN}$

Specificity (TNR) = $\frac{TN}{TN + FP}$, FPR = $1 - $ TNR.

F1 = $\frac{2 \cdot P \cdot R}{P + R}$

F-beta = $(1+\beta^2)\frac{P \cdot R}{\beta^2 P + R}$

Balanced accuracy = $\frac{1}{2}(\text{TPR} + \text{TNR})$

MCC = $\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

ROC-AUC: площадь под кривой TPR(FPR).

Log-loss = $-\frac{1}{N}\sum [y_i \log p_i + (1-y_i)\log(1-p_i)]$

Brier = $\frac{1}{N}\sum (p_i - y_i)^2$

---

## 8. Кросс-валидация

### Схемы

k-Fold: $k$ частей, каждая по очереди — тест; итого $k$ моделей.

LOOCV: $k = N$.

Stratified k-Fold: сохраняет распределение классов.

### Оценка

$\text{CV}_k = \frac{1}{k}\sum_{j=1}^k Q_j^{\text{test}}$

Nested CV: внешний (оценка обобщения) + внутренний (подбор гиперпараметров).

---

## 9. Ансамбли

### Voting

Hard: $\hat y = \arg\max_c \sum_m \mathbb{1}[a_m(x) = c]$

Soft: $\hat y = \arg\max_c \sum_m w_m p_m(c|x)$

### Bagging / Random Forest

$a_{\text{bag}}(x) = \frac{1}{M}\sum_m a_m(x)$ (регрессия), $\arg\max$ голосов (классиф.).

Вероятность попасть в бутстрэп: $1 - (1-1/N)^N \to 1 - 1/e \approx 0.632$.

OOB size: $\approx 0.368 N$.

Разложение variance: $\text{Var}(\bar a) = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$.

### AdaBoost (бинарный, $y \in \{-1, +1\}$)

Инициализация: $w_i^{(1)} = 1/N$

Взвешенная ошибка: $\varepsilon_m = \sum_i w_i^{(m)} \mathbb{1}[a_m(x_i) \ne y_i]$

Вес модели: $\alpha_m = \frac{1}{2}\ln\frac{1-\varepsilon_m}{\varepsilon_m}$

Обновление весов: $w_i^{(m+1)} \propto w_i^{(m)} \exp(-\alpha_m y_i a_m(x_i))$

Итог: $a(x) = \text{sign}\left(\sum_m \alpha_m a_m(x)\right)$

Эквивалентная потеря: $\sum_i \exp(-y_i F(x_i))$

### Gradient Boosting

$F_0(x) = \arg\min_c \sum \mathcal{L}(y_i, c)$

Псевдоостатки: $r_i^{(m)} = -\frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)}\bigg|_{F = F_{m-1}}$

Для MSE: $r_i = y_i - F_{m-1}(x_i)$

Для logloss: $r_i = y_i - \sigma(F_{m-1}(x_i))$

Обновление: $F_m = F_{m-1} + \nu h_m$, $\nu$ — learning rate.

XGBoost objective: $\mathcal{L} + \sum_m \Omega(h_m)$, $\Omega(h) = \gamma T + \frac{\lambda}{2}\|w\|^2$.

---

## 10. Нейронные сети

### Нейрон

Pre-activation: $z = w^\top x + b = \sum_j w_j x_j + b$

Активация: $y = \varphi(z)$

Слой: $z = Wx + b$, $y = \varphi(z)$, $W \in \mathbb{R}^{H \times d}$

Параметров в слое: $H(d + 1)$

### Активации

Sigmoid: $\sigma(z) = \frac{1}{1+e^{-z}}$, $\sigma' = \sigma(1-\sigma)$

Tanh: $\tanh z = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, $\tanh' = 1 - \tanh^2 z$

ReLU: $\max(0, z)$; $\varphi' = \mathbb{1}[z > 0]$

Leaky ReLU: $\max(\alpha z, z)$, $\alpha = 0.01$

ELU: $z$ если $z \ge 0$, иначе $\alpha(e^z - 1)$

GELU: $z \cdot \Phi(z)$

Softmax: $\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$

Softplus: $\ln(1 + e^z)$

### Инициализация

LeCun: $\text{Var}(W) = 1/d_{\text{in}}$

Xavier/Glorot: $\text{Var}(W) = \frac{2}{d_{\text{in}} + d_{\text{out}}}$; равномерно $\left[\pm\sqrt{6/(d_{\text{in}}+d_{\text{out}})}\right]$

He (Kaiming): $\text{Var}(W) = \frac{2}{d_{\text{in}}}$ (для ReLU)

### Forward / Backward

Forward: $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$; $h^{(l)} = \varphi(z^{(l)})$

Loss → dL:

Softmax + CE: $\delta^{(L)} = \hat y - y_{\text{onehot}}$

Sigmoid + BCE: $\delta^{(L)} = \hat y - y$

MSE + линейная: $\delta^{(L)} = \hat y - y$

Backward (рекурсия):
$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^\top
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
$$
$$
\delta^{(l-1)} = (W^{(l)})^\top \delta^{(l)} \odot \varphi'(z^{(l-1)})
$$

Обновление: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$

### Dropout

Тренировка: $\tilde h = h \odot m / (1-p)$, $m \sim \text{Bernoulli}(1-p)$.

Инференс: без маски.

### Batch Normalization

$\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$, $y = \gamma \hat x + \beta$

$\mu_B, \sigma_B^2$ — среднее и дисперсия по батчу.

---

## 11. Оптимизаторы

### SGD / Momentum / Nesterov

SGD: $\theta_{t+1} = \theta_t - \eta g_t$

Momentum: $v_t = \mu v_{t-1} + g_t$; $\theta_{t+1} = \theta_t - \eta v_t$, $\mu \approx 0.9$

Nesterov: $v_t = \mu v_{t-1} + \nabla Q(\theta_t - \eta \mu v_{t-1})$

### AdaGrad / RMSProp

AdaGrad: $G_t = G_{t-1} + g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \varepsilon}} g_t$

RMSProp: $G_t = \rho G_{t-1} + (1-\rho) g_t^2$, $\rho \approx 0.9$

### Adam / AdamW

$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$

$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

Bias correction: $\hat m_t = \frac{m_t}{1-\beta_1^t}$, $\hat v_t = \frac{v_t}{1-\beta_2^t}$

Adam: $\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}$

AdamW (decoupled weight decay): $\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} + \lambda \theta_t\right)$

Дефолты: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}, \eta = 10^{-3}$.

### LR Schedulers

Step decay: $\eta_t = \eta_0 \gamma^{\lfloor t/k \rfloor}$

Exp: $\eta_t = \eta_0 \gamma^t$

Cosine annealing: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$

---

## 12. Полезные тождества

### Log-sum-exp trick (численная устойчивость)

$\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}$, $m = \max_i x_i$

### Sigmoid и softmax

$\sigma(z) = \frac{e^z}{1 + e^z}$, $1 - \sigma(z) = \sigma(-z)$

Производная: $\sigma'(z) = \sigma(z)\sigma(-z)$

Softmax + CE (для одного примера): $\frac{\partial \mathcal{L}}{\partial z_k} = p_k - y_k$

### Chain rule

$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$

Для векторов: $\nabla_x L = J_y^\top \nabla_y L$

### Производная нормы

$\nabla_w \|w\|^2 = 2w$

$\nabla_w \|y - Xw\|^2 = 2X^\top (Xw - y)$

### Bias–Variance decomposition

$\mathbb{E}[(y - \hat y)^2] = \underbrace{(\mathbb{E}\hat y - y)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat y - \mathbb{E}\hat y)^2]}_{\text{Variance}} + \sigma^2$

---

## 13. Быстрая справка по гиперпараметрам

| Алгоритм | Ключевые гиперпараметры | Типичные значения |
|----------|--------------------------|-------------------|
| Ridge    | $\alpha$ (L2)            | $10^{-3}$–$10^2$ (log-scale CV) |
| Lasso    | $\alpha$ (L1)            | $10^{-4}$–$10^1$   |
| LogReg   | $C = 1/\lambda$          | $10^{-2}$–$10^2$   |
| kNN      | $k$, metric              | $\sqrt{N}$, Euclid |
| DT       | max_depth, min_samples_leaf | 3–10, 1–20       |
| RF       | n_estimators, max_features | 100–1000, $\sqrt d$ |
| GB       | n_estimators, lr, max_depth | 100–1000, 0.01–0.1, 3–8 |
| SVM      | C, $\gamma$ (RBF)        | $10^{-2}$–$10^2$, $1/d$ |
| K-means  | K                        | elbow, silhouette  |
| DBSCAN   | $\varepsilon$, minPts     | k-distance plot, $2d$ |
| MLP (Adam)| lr, hidden, dropout    | $10^{-3}$, 64–512, 0.1–0.5 |
| CNN (SGD) | lr, momentum, weight decay | 0.1→cosine, 0.9, $5\cdot 10^{-4}$ |

---

## 14. Мнемоники и соответствия

**Горбаченко ↔ стандарт:**

| Горбаченко | Стандарт |
|------------|----------|
| $\ell$     | $N$      |
| $n$        | $d$      |
| $F$        | $X$ (design matrix) |
| $X^\ell$   | $\mathcal{D}$ |
| $a(x)$     | $h(x; \theta)$ или $\hat y(x)$ |
| $Q(a, X^\ell)$ | $\hat R(\theta)$ (empirical risk) |

**Универсальный чек-лист обучения модели:**
1. EDA → пропуски, выбросы, распределения, корреляции.
2. Split (train/val/test), стратификация.
3. Pipeline: impute → encode → scale → feature_selection → model.
4. Baseline (простая модель).
5. Cross-validation для гиперпараметров.
6. Оценка на holdout test.
7. Интерпретация (permutation importance, SHAP).

**Пороги тревоги:**

- $R^2 < 0$ → модель хуже `mean`.
- Accuracy ≈ доля мажоритарного класса → ничего не выучила.
- Train loss $\to 0$, val loss растёт → переобучение.
- Loss = NaN → lr слишком большой или деление на 0.
- Gradient norm $\to 0$ или $\to \infty$ → vanishing/exploding.
