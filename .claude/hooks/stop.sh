#!/bin/bash

# Читаем JSON из stdin
INPUT=$(cat)

# Если хук уже запускался в этом цикле — позволяем остановиться
HOOK_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
if [ "$HOOK_ACTIVE" = "true" ]; then
  exit 0
fi

PROGRESS_FILE="$(dirname "$0")/../progress.json"

# Если файла прогресса нет — создаём
if [ ! -f "$PROGRESS_FILE" ]; then
  echo '{"current_lab": 4, "completed_labs": [], "status": "in_progress"}' > "$PROGRESS_FILE"
  exit 0
fi

STATUS=$(jq -r '.status' "$PROGRESS_FILE")
CURRENT=$(jq -r '.current_lab' "$PROGRESS_FILE")

# Если лаба завершена и ещё не дошли до 16 — продолжаем
if [ "$STATUS" = "lab_done" ] && [ "$CURRENT" -le 16 ]; then
  NEXT=$((CURRENT + 1))

  # Обновляем прогресс: следующая лаба, статус in_progress
  jq --argjson next "$NEXT" \
     --argjson cur "$CURRENT" \
     '.current_lab = $next | .status = "in_progress" | .completed_labs += [$cur]' \
     "$PROGRESS_FILE" > /tmp/progress_tmp.json && cp /tmp/progress_tmp.json "$PROGRESS_FILE" && rm /tmp/progress_tmp.json

  # Говорим Claude продолжать и даём следующую инструкцию
  echo "{\"continue\": true, \"reason\": \"Лаба $CURRENT завершена. Начинай выполнение лабы $NEXT по тому же процессу из AUTORUN_PROMPT.md. Прочитай задание, изучи примеры, составь план, реализуй, запусти nbconvert, сделай коммит, обнови README и progress.json (status=lab_done).\"}"
  exit 0
fi

# Все лабы сделаны — останавливаемся
if [ "$CURRENT" -gt 16 ]; then
  echo '{"continue": false}'
  exit 0
fi

# Иначе — нормальная остановка
exit 0