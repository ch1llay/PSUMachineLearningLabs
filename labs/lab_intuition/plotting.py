"""Переиспользуемые функции визуализации для лабораторной 'Интуиция ML'.

Сюда вынесена логика, которая повторяется во всех 3 ноутбуках:
- 3D-поверхность функции ошибки в Plotly (можно крутить мышкой)
- Контурная карта функции ошибки в matplotlib + траектория GD
- Decision boundary для классификаторов в 2D
- 3D-поверхность предсказаний для регрессоров
- Кривая loss по эпохам
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap


def _grid_loss(loss_fn: Callable, w1_range: np.ndarray, w2_range: np.ndarray) -> np.ndarray:
    """Посчитать loss(w1, w2) на сетке. Возвращает 2D-массив shape (len(w2), len(w1))."""
    W1, W2 = np.meshgrid(w1_range, w2_range)
    L = np.empty_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            L[i, j] = loss_fn(W1[i, j], W2[i, j])
    return L


def plot_loss_surface_3d(
    loss_fn: Callable,
    w1_range: np.ndarray,
    w2_range: np.ndarray,
    trajectory: Optional[np.ndarray] = None,
    title: str = "Поверхность функции ошибки",
    w1_name: str = "w1",
    w2_name: str = "w2",
) -> go.Figure:
    """3D-поверхность loss(w1, w2) в Plotly. Можно крутить мышкой.

    Параметры:
        loss_fn: функция (w1, w2) -> loss
        w1_range, w2_range: 1D-массивы значений по осям
        trajectory: опционально, массив shape (n_steps, 2) с точками GD-траектории —
                    будет нарисована поверх поверхности
    """
    L = _grid_loss(loss_fn, w1_range, w2_range)

    fig = go.Figure(
        data=[
            go.Surface(
                x=w1_range,
                y=w2_range,
                z=L,
                colorscale="Viridis",
                opacity=0.85,
                showscale=True,
                colorbar=dict(title="loss"),
            )
        ]
    )

    if trajectory is not None:
        traj_loss = np.array([loss_fn(w1, w2) for w1, w2 in trajectory])
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=traj_loss,
                mode="lines+markers",
                line=dict(color="red", width=4),
                marker=dict(size=3, color="red"),
                name="траектория GD",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=w1_name, yaxis_title=w2_name, zaxis_title="loss"),
        width=800,
        height=600,
    )
    return fig


def plot_loss_contour(
    loss_fn: Callable,
    w1_range: np.ndarray,
    w2_range: np.ndarray,
    trajectory: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: str = "Контурная карта функции ошибки",
    w1_name: str = "w1",
    w2_name: str = "w2",
    n_levels: int = 25,
) -> Axes:
    """Контурная карта loss(w1, w2) с опциональной траекторией GD сверху.

    Это центральная визуализация для интуиции: видно эллипсы уровня
    (вытянутые при разных масштабах признаков, круглые при одинаковых)
    и саму ломаную траекторию спуска в минимум.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    L = _grid_loss(loss_fn, w1_range, w2_range)
    W1, W2 = np.meshgrid(w1_range, w2_range)

    cs = ax.contourf(W1, W2, L, levels=n_levels, cmap="viridis", alpha=0.7)
    ax.contour(W1, W2, L, levels=n_levels, colors="white", linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="loss")

    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "r.-", markersize=4, linewidth=1.2,
                label="траектория GD")
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color="white", s=80,
                   edgecolor="black", zorder=5, label="старт")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="red", s=80, marker="*",
                   edgecolor="black", zorder=5, label="финал")
        ax.legend(loc="best")

    ax.set_xlabel(w1_name)
    ax.set_ylabel(w2_name)
    ax.set_title(title)
    return ax


def plot_decision_boundary(
    predict_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "",
    feature_names: tuple = ("x1", "x2"),
    h: float = None,
) -> Axes:
    """Decision boundary для бинарного классификатора в 2D.

    Параметры:
        predict_fn: функция (X_2d) -> метки классов 0/1
        X: shape (n, 2) — данные в исходных координатах
        y: shape (n,) — истинные метки 0/1
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    if h is None:
        h = max((x_max - x_min) / 200, (y_max - y_min) / 200)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)

    cmap_bg = ListedColormap(["#a8d8ff", "#ffb3a8"])
    cmap_pts = ListedColormap(["#1f77b4", "#d62728"])

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts, edgecolor="k", s=30)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    return ax


def plot_regression_surface(
    predict_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple = ("x1", "x2"),
    target_name: str = "y",
    title: str = "",
    n_grid: int = 30,
) -> go.Figure:
    """3D-поверхность предсказаний регрессора над плоскостью (x1, x2)."""
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), n_grid)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[X1.ravel(), X2.ravel()]
    Z = predict_fn(grid).reshape(X1.shape)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x1_range, y=x2_range, z=Z, colorscale="Viridis",
                             opacity=0.7, showscale=False, name="предсказания"))
    fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode="markers",
                               marker=dict(size=3, color=y, colorscale="Reds",
                                           line=dict(color="black", width=0.3)),
                               name="данные"))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=feature_names[0], yaxis_title=feature_names[1],
                   zaxis_title=target_name),
        width=800, height=600,
    )
    return fig


def plot_loss_curve(history: dict, ax: Optional[Axes] = None,
                    title: str = "Loss по эпохам", log_y: bool = False) -> Axes:
    """График кривой обучения. history должен содержать ключ 'loss' со списком значений."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    losses = history["loss"]
    ax.plot(np.arange(len(losses)), losses, linewidth=1.5)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_weight_evolution(history: dict, ax: Optional[Axes] = None,
                          title: str = "Эволюция весов по эпохам",
                          weight_names: Optional[list] = None) -> Axes:
    """График значений весов w1, w2, ..., b по эпохам.

    history должен содержать ключи 'w' (список массивов формы (n_features,)) и 'b' (список чисел).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    W = np.array(history["w"])
    b = np.array(history["b"])
    n_features = W.shape[1]

    if weight_names is None:
        weight_names = [f"w{i+1}" for i in range(n_features)]

    epochs = np.arange(len(b))
    for i in range(n_features):
        ax.plot(epochs, W[:, i], label=weight_names[i], linewidth=1.5)
    ax.plot(epochs, b, label="b", linewidth=1.5, linestyle="--")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("значение")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
