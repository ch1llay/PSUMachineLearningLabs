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
  echo '{"current_lab": 4, "completed_labs": [], "status": "in_progress", "retry_count": 0, "last_error": null}' > "$PROGRESS_FILE"
  exit 0
fi

STATUS=$(jq -r '.status' "$PROGRESS_FILE")
CURRENT=$(jq -r '.current_lab' "$PROGRESS_FILE")
MODE=$(jq -r '.mode // "execute"' "$PROGRESS_FILE")

# ========== РЕЖИМ РЕВЬЮ ==========
if [ "$MODE" = "review" ]; then
  if [ "$STATUS" = "lab_review_done" ] && [ "$CURRENT" -le 16 ]; then
    NEXT=$((CURRENT + 1))

    # Лабы 8 нет в курсе — пропускаем
    if [ "$NEXT" -eq 8 ]; then
      NEXT=9
    fi

    # Обновляем прогресс: следующая лаба, статус review_in_progress
    jq --argjson next "$NEXT" --argjson cur "$CURRENT" \
       '.current_lab = $next | .status = "review_in_progress" | .retry_count = 0 | .last_error = null' \
       "$PROGRESS_FILE" > /tmp/progress_tmp.json && cp /tmp/progress_tmp.json "$PROGRESS_FILE" && rm /tmp/progress_tmp.json

    # Если следующая лаба > 16 — все лабы проверены
    if [ "$NEXT" -gt 16 ]; then
      echo '{"continue": true, "reason": "Все лабы проверены! Сгенерируй итоговый .claude/REVIEW_REPORT.md, закоммить все REVIEW.md и установи status=all_reviewed в progress.json."}'
      exit 0
    fi

    echo "{\"continue\": true, \"reason\": \"Ревью лабы $CURRENT завершено. Начинай ревью лабы $NEXT по процессу из .claude/REVIEW_PROMPT.md.\"}"
    exit 0
  fi

  if [ "$STATUS" = "all_reviewed" ]; then
    echo '{"continue": false, "reason": "Ревью всех лабораторных работ завершено!"}'
    exit 0
  fi

  # Обычная остановка в режиме ревью
  exit 0
fi

# ========== РЕЖИМ ВЫПОЛНЕНИЯ ==========
# Если лаба завершена и ещё не дошли до 16 — продолжаем
if [ "$STATUS" = "lab_done" ] && [ "$CURRENT" -le 16 ]; then
  NEXT=$((CURRENT + 1))

  # Лабы 8 нет в курсе — пропускаем
  if [ "$NEXT" -eq 8 ]; then
    NEXT=9
  fi

  # Обновляем прогресс: следующая лаба, статус in_progress, сброс retry
  jq --argjson next "$NEXT" \
     --argjson cur "$CURRENT" \
     '.current_lab = $next | .status = "in_progress" | .completed_labs += [$cur] | .retry_count = 0 | .last_error = null' \
     "$PROGRESS_FILE" > /tmp/progress_tmp.json && cp /tmp/progress_tmp.json "$PROGRESS_FILE" && rm /tmp/progress_tmp.json

  # Если следующая лаба > 16 — все лабы завершены
  if [ "$NEXT" -gt 16 ]; then
    echo '{"continue": false, "reason": "Все лабораторные работы (4-16) завершены!"}'
    exit 0
  fi

  # Говорим Claude продолжать и даём следующую инструкцию
  echo "{\"continue\": true, \"reason\": \"Лаба $CURRENT завершена. Начинай выполнение лабы $NEXT по процессу из .claude/AUTORUN_PROMPT.md. Прочитай задание, изучи примеры, составь план с теорией, проверь план, реализуй с образовательным контентом, пройди валидацию качества, сделай коммит, обнови README и progress.json (status=lab_done).\"}"
  exit 0
fi

# Все лабы сделаны — останавливаемся
if [ "$CURRENT" -gt 16 ]; then
  echo '{"continue": false}'
  exit 0
fi

# Иначе — нормальная остановка
exit 0
