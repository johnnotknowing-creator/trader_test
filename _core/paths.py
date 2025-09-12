# _core/paths.py
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Union

# 1) Определяем корень проекта
# приоритет: переменная окружения (можно задать в devcontainer.json), иначе — по расположению файла
_ENV_KEY = "TRADER_TEST_ROOT"
if _ENV_KEY in os.environ:
    PROJECT_ROOT = Path(os.environ[_ENV_KEY]).resolve()
else:
    # .../trader_test/_core/paths.py -> два уровня вверх
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2) Стандартные каталоги (кросс-платформенно: Windows E:\..., Linux /workspaces/...)
DATA_DIR    = PROJECT_ROOT / "1_data"
RESULTS_DIR = PROJECT_ROOT / "2_results"
TOOLS_DIR   = PROJECT_ROOT / "_tools"
CORE_DIR    = PROJECT_ROOT / "_core"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Гарантируем, что проект и _core в sys.path (на случай запуска скриптов из подпапок)
for p in (str(PROJECT_ROOT), str(CORE_DIR), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.append(p)

# 3) Базовые утилиты путей
def rel(*parts: Union[str, Path]) -> Path:
    "Путь внутри проекта от корня"
    return PROJECT_ROOT.joinpath(*parts)

def ensure_dirs() -> None:
    "Создаёт стандартные папки результатов, если их нет"
    for p in [
        RESULTS_DIR,
        RESULTS_DIR / "features",
        RESULTS_DIR / "featuresets",
        RESULTS_DIR / "models",
        RESULTS_DIR / "reports",
        RESULTS_DIR / "notebooks",
    ]:
        p.mkdir(parents=True, exist_ok=True)

# 4) .env (не обязателен). Делаем ленивую загрузку — без жёсткой зависимости от python-dotenv
def load_dotenv_if_exists(filename: str = ".env") -> None:
    env_path = PROJECT_ROOT / filename
    if env_path.exists():
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(env_path)  # не падаем, если пакета нет
        except Exception:
            pass  # тихо продолжаем, переменные просто не подгрузятся
