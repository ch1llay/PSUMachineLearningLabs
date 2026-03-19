# Диагностика: проверка готовности к автономному выполнению лаб

Выполни все проверки по порядку. После каждой пиши результат: ✅ OK или ❌ ПРОБЛЕМА + что именно не так.

## 1. Проверка структуры репозитория

```bash
# Корневые файлы
test -f CLAUDE.md && echo "✅ CLAUDE.md" || echo "❌ CLAUDE.md не найден"
test -f README.md && echo "✅ README.md" || echo "❌ README.md не найден"
test -f .claude/AUTORUN_PROMPT.md && echo "✅ .claude/AUTORUN_PROMPT.md" || echo "❌ .claude/AUTORUN_PROMPT.md не найден"

# Claude конфиги
test -f .claude/settings.json && echo "✅ .claude/settings.json" || echo "❌ .claude/settings.json не найден"
test -f .claude/progress.json && echo "✅ .claude/progress.json" || echo "❌ .claude/progress.json не найден"
test -f .claude/hooks/stop.sh && echo "✅ .claude/hooks/stop.sh" || echo "❌ .claude/hooks/stop.sh не найден"
```

## 2. Проверка прав на хук

```bash
test -x .claude/hooks/stop.sh && echo "✅ stop.sh исполняемый" || echo "❌ stop.sh не исполняемый — выполни: chmod +x .claude/hooks/stop.sh"
```

## 3. Проверка содержимого progress.json

```bash
cat .claude/progress.json
```
Убедись что JSON валидный и содержит поля: `current_lab`, `completed_labs`, `status`.
```bash
python3 -c "import json; d=json.load(open('.claude/progress.json')); print('✅ JSON валидный:', d)" || echo "❌ JSON невалидный"
```

## 4. Проверка settings.json

```bash
cat .claude/settings.json
python3 -c "import json; d=json.load(open('.claude/settings.json')); hooks=d.get('hooks',{}); stop=hooks.get('Stop',[]); print('✅ Stop hook зарегистрирован:', stop) if stop else print('❌ Stop hook не найден в settings.json')"
```

## 5. Проверка что jq установлен

```bash
jq --version && echo "✅ jq доступен" || echo "❌ jq не установлен — выполни: sudo apt install jq"
```

## 6. Проверка виртуального окружения

```bash
test -d /home/ilya/venvs/psu-ml && echo "✅ venv существует" || echo "❌ venv не найден по пути /home/ilya/venvs/psu-ml"
source /home/ilya/venvs/psu-ml/bin/activate && python3 --version && echo "✅ venv активируется"
source /home/ilya/venvs/psu-ml/bin/activate && jupyter --version && echo "✅ jupyter доступен" || echo "❌ jupyter не установлен"
source /home/ilya/venvs/psu-ml/bin/activate && python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn, plotly; print('✅ все библиотеки доступны')" || echo "❌ не все библиотеки установлены"
```

## 7. Проверка датасета

```bash
test -f datasets/gtzan/features_30_sec.csv && echo "✅ датасет найден" || echo "❌ датасет не найден по пути datasets/gtzan/features_30_sec.csv"
source /home/ilya/venvs/psu-ml/bin/activate && python3 -c "
import pandas as pd
df = pd.read_csv('datasets/gtzan/features_30_sec.csv')
print(f'✅ датасет загружается: {df.shape[0]} строк, {df.shape[1]} столбцов')
print(f'   колонки label: {\"label\" in df.columns}')
"
```

## 8. Проверка заданий преподавателя

```bash
ls "машинное обучение/машинное обучение/" && echo "✅ папка с заданиями найдена" || echo "❌ папка с заданиями не найдена"
```

## 9. Проверка примеров одногруппников

```bash
# Федя
ls "examples/Лабы Примеры Федя/" | grep "ML_Lab" | head -5 && echo "✅ примеры Феди найдены" || echo "❌ примеры Феди не найдены"

# Олег
ls "examples/лабы примеры Олег/Машинное обучение/" | head -5 && echo "✅ примеры Олега найдены" || echo "❌ примеры Олега не найдены"
```

## 10. Проверка AUTORUN_PROMPT.md на корректность markdown

```bash
# Количество блоков кода должно быть чётным
COUNT=$(grep -c '```' .claude/AUTORUN_PROMPT.md)
if [ $((COUNT % 2)) -eq 0 ]; then
  echo "✅ код-блоки закрыты корректно (найдено $COUNT штук)"
else
  echo "❌ нечётное количество ``` ($COUNT) — есть незакрытый блок кода"
fi
```

## 11. Тест хука вручную

Симулируй вызов хука как будто лаба завершена:
```bash
# Временно выставим status=lab_done
python3 -c "
import json
d = json.load(open('.claude/progress.json'))
d['status'] = 'lab_done'
d['current_lab'] = 4
json.dump(d, open('.claude/progress.json', 'w'), indent=2)
print('Установлен тестовый статус lab_done для лабы 4')
"

# Вызываем хук вручную с тестовым JSON
echo '{"stop_hook_active": false}' | bash .claude/hooks/stop.sh
echo "Код возврата хука: $?"

# Проверяем что прогресс обновился
cat .claude/progress.json
# Ожидаем: current_lab=5, status=in_progress
```

## 12. Восстанови progress.json в исходное состояние

```bash
python3 -c "
import json
d = json.load(open('.claude/progress.json'))
d['status'] = 'in_progress'
d['current_lab'] = 4
d['completed_labs'] = [1, 2, 3]
json.dump(d, open('.claude/progress.json', 'w'), indent=2)
print('✅ progress.json восстановлен:', d)
"
```

## Итог

После всех проверок выведи сводку:
- Сколько пунктов ✅ OK
- Список всех ❌ ПРОБЛЕМ с конкретными командами для исправления
- Вердикт: ГОТОВО К ЗАПУСКУ / НУЖНО ИСПРАВИТЬ