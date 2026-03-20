# init-lab-perform

Инфраструктура для автономного выполнения и проверки университетских лабораторных работ через Claude Code.

## Структура

```
init-lab-perform/
  commands/
    init-lab-repo.md    # Скилл инициализации репозитория (/init-lab-repo)
    diagnostics.md      # Скилл диагностики окружения (/diagnostics)
  templates/
    *.tmpl              # Шаблоны генерируемых файлов (9 штук)
```

## Установка

Скопировать папку `init-lab-perform/` в корень целевого репозитория:

```bash
cp -r init-lab-perform/ /path/to/target-repo/
```

Скопировать скиллы в `.claude/commands/` целевого репозитория (или глобально в `~/.claude/commands/`):

```bash
# Локально для одного репо
mkdir -p /path/to/target-repo/.claude/commands
cp init-lab-perform/commands/*.md /path/to/target-repo/.claude/commands/

# Или глобально
cp init-lab-perform/commands/*.md ~/.claude/commands/
```

## Использование

1. Перейти в целевой репозиторий
2. Запустить `/init-lab-repo` — скилл просканирует репо, задаст вопросы и сгенерирует:
   - `CLAUDE.md` — инструкции проекта
   - `.claude/AUTORUN_PROMPT.md` — промт автономного выполнения
   - `.claude/REVIEW_PROMPT.md` — промт автономной проверки
   - `.claude/DIAGNOSTICS_PROMPT.md` — промт диагностики
   - `.claude/progress.json` — state machine прогресса
   - `.claude/settings.json` — разрешения и хуки
   - `.claude/hooks/stop.sh` — хук автопродвижения
   - `.mcp.json` — MCP серверы
   - `README.md` — прогресс-таблица
3. Запустить `/diagnostics` для проверки окружения
4. Вставить текст из `.claude/AUTORUN_PROMPT.md` для запуска автономного выполнения

## Шаблоны

Шаблоны используют плейсхолдеры `{{PLACEHOLDER}}` и условные секции `{{#IF CONDITION}}...{{/IF}}`. Скилл `/init-lab-repo` заполняет их на основе ответов пользователя.

Поддерживаемые вариации:
- С примерами одногруппников / без примеров
- С общим датасетом / без датасета
- Произвольная нумерация лаб (с пропусками)
- Любые Python-библиотеки и venv
