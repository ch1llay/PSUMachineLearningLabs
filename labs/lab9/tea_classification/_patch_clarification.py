"""Patch notebook 01: add clarification about formula ★★ dimensions and (i) batch notation.
Uses raw strings to avoid all escape issues."""
import json

path = 'labs/lab9/tea_classification/01_numpy_sklearn.ipynb'
nb = json.load(open(path, 'r', encoding='utf-8'))

new_insert = r"""

**Важно про размерности (★★) — это формула для ОДНОГО примера $x$:**

- $x$ — **вектор** длины $D$ (5 признаков одного конкретного чая): $x = (x_1, x_2, x_3, x_4, x_5)$
- $w_k$ — **вектор** длины $D$ (веса класса $k$ для всех признаков): $w_k = (w_{k,1}, \dots, w_{k,5})$
- $(p_k - y_k)$ — **скаляр** (одно число для этого примера и этого класса)
- $(p_k - y_k) \cdot x$ — скаляр умноженный на вектор = **вектор** длины $D$

В покомпонентной форме это:

$$\frac{\partial L}{\partial w_{k,j}} = (p_k - y_k) \cdot x_j, \quad j = 1, \dots, D$$

То есть производная по **$j$-му весу класса $k$** равна «ошибке класса $k$» умноженной на **$j$-й признак** того же примера. Чем больше признак — тем сильнее корректируется соответствующий вес.

"""

batch_clarification = r"""### 3.5 Для батча из $n$ примеров — что такое индекс $(i)$

У нас в train 800 примеров. **Верхний индекс $(i)$ в круглых скобках** — это номер примера в батче, $i = 1, \dots, n$:

- $x^{(i)}$ — вектор признаков **$i$-го примера** (5 чисел для $i$-го чая)
- $p_k^{(i)}$ — вероятность класса $k$, которую модель предсказала **для $i$-го примера**
- $y_k^{(i)}$ — one-hot метка для $i$-го примера (0 или 1 в позиции $k$)
- $L^{(i)}$ — loss **для $i$-го примера**

Верхний индекс в скобках — стандартная ML-нотация, чтобы не путать с нижним индексом координаты. Например, $x^{(5)}_3$ = третий признак пятого примера в батче.

"""

# Найти нужную ячейку и заменить её содержимое (переписать полностью с чистой версией)
target_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'markdown':
        continue
    src = ''.join(cell['source'])
    if 'star\\star\\star)$$' in src and '3.5 Для батча' in src:
        target_cell_idx = i
        break

assert target_cell_idx is not None, "Не нашли теоретическую ячейку"

# Получить текущий source и разрезать по "### 3.5"
current = ''.join(nb['cells'][target_cell_idx]['source'])

# Чистим предыдущие попытки патча (если есть)
markers_to_strip = ['**Важно про размерности', 'Для батча из $n$ примеров — что такое индекс']
for marker in markers_to_strip:
    while marker in current:
        idx = current.find(marker)
        # Стираем от маркера до следующего ###
        next_section = current.find('###', idx + len(marker))
        if next_section > 0:
            current = current[:idx] + current[next_section:]
        else:
            current = current[:idx]
        break  # один проход

# Теперь разрезаем по "### 3.5 Для батча"
split_marker = '### 3.5 Для батча из $n$ примеров'
idx = current.find(split_marker)
assert idx > 0, f"Не нашли маркер '{split_marker}' в очищенном тексте"

before = current[:idx]
# Всё что после заголовка "3.5" — только содержимое без самого заголовка (мы его переписываем)
after_section = current[idx + len(split_marker):]

new_src = before.rstrip() + new_insert + batch_clarification + after_section.lstrip()

lines = new_src.split('\n')
nb['cells'][target_cell_idx]['source'] = [ln + '\n' for ln in lines[:-1]] + [lines[-1]]

json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'Обновлена ячейка {target_cell_idx}')

# Валидация
nb2 = json.load(open(path, 'r', encoding='utf-8'))
for i, c in enumerate(nb2['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'<cell{i}>', 'exec')

# Проверить что патч действительно есть и LaTeX корректный
patched = ''.join(nb2['cells'][target_cell_idx]['source'])
assert 'ОДНОГО примера' in patched
assert r'\frac{\partial L}{\partial w_{k,j}}' in patched, "LaTeX frac/partial битый!"
assert '$x^{(i)}$' in patched
print(f'OK: {len(nb2["cells"])} ячеек, LaTeX корректен, патч на месте')
