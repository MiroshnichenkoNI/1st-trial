# DOVER + Qwen

Для каждого видео: техническая и эстетическая оценки DOVER и описание от Qwen3-VL-2B-Instruct.

## Установка

Linux x86_64, Python 3.12, NVIDIA-драйвер; целевая GPU — Tesla V100 32 GB.

Системные пакеты для Ubuntu 24.04:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential
```

На Ubuntu 22.04 Python 3.12 нужно установить отдельно вместе с `venv` и заголовками.

```bash
bash setup.sh
```

Скрипт создаёт одно окружение `.venv`, устанавливает зависимости с PyPI, скачивает код DOVER и веса обеих моделей. Версии сохраняются в `out/installed.txt`.

## Обработка

```bash
./run.sh --input /path/to/videos --output out/run1
```

Выбираются `*.mp4` из входной папки. `--videos 1.mp4 2.mp4` задаёт список и порядок. Выходная папка должна быть новой или пустой.

Один проход прогревает модели, второй записывает результаты:

- `videos.jsonl` — `job_id`, `video_id`, две оценки DOVER, описание, число токенов, причина завершения;
- `warmup.jsonl` — результаты прогрева;
- `run.json` — настройки и время рабочего прохода.

Пайплайн, текущие методы параллелизма и другие возможные подходы описаны в [REPORT.md](REPORT.md).

