from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _core/data_loader.py
import os
import pandas as pd
from typing import Optional, List

# Папка с данными рядом с проектом
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "1_data"))

DATE_CANDIDATES: List[str] = ["datetime", "date", "TRADEDATE", "DATE", "Datetime"]

def _resolve_path(ticker_or_path: str) -> Optional[str]:
    """Определяем путь к файлу: либо это уже путь, либо ищем по шаблонам в 1_data/."""
    # Если это существующий путь — использовать его
    if os.path.exists(ticker_or_path):
        return ticker_or_path

    # Иначе воспринимаем как тикер
    ticker = str(ticker_or_path).upper().strip()
    candidates = [
        f"{ticker}.csv",
        f"{ticker}_D1.csv",
        f"{ticker}_D1_MOEX.csv",
    ]
    # спец-файлы
    if ticker == "IMOEX":
        candidates.insert(0, "IMOEX.csv")
    if ticker == "CBR_KEY_RATE":
        candidates.insert(0, "CBR_KEY_RATE.csv")

    for fname in candidates:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            return p

    return None


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    for c in DATE_CANDIDATES:
        if c in cols:
            return c
    # часто index уже датовый — если так, вернём None и не будем переиндексировать
    return None


def load_data(ticker_or_path: str) -> Optional[pd.DataFrame]:
    """
    Универсальная загрузка:
    - принимает тикер (например, 'GAZP') ИЛИ полный путь к CSV,
    - поддерживает форматы: {TICKER}.csv, {TICKER}_D1.csv, {TICKER}_D1_MOEX.csv,
      а также IMOEX.csv и CBR_KEY_RATE.csv,
    - автоопределяет разделитель и колонку даты,
    - нормализует имя даты к 'datetime' и ставит её индексом.
    """
    file_path = _resolve_path(ticker_or_path)
    if file_path is None:
        print(f"[load_data] ⚠️ Файл для '{ticker_or_path}' не найден в {DATA_DIR}")
        return None

    try:
        # sep=None + engine='python' — пусть pandas сам определит разделитель (',' или ';')
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        print(f"[load_data] ❌ Ошибка чтения {file_path}: {e}")
        return None

    if df is None or df.empty:
        print(f"[load_data] ⚠️ Пустой файл: {file_path}")
        return None

    # Определяем колонку даты
    date_col = _detect_date_column(df)

    # Если нашли колонку даты — парсим и ставим как индекс
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            # нормализуем в имя 'datetime'
            if date_col != "datetime":
                df = df.rename(columns={date_col: "datetime"})
            df = df.set_index("datetime").sort_index()
        except Exception as e:
            print(f"[load_data] ⚠️ Не удалось распарсить дату в колонке '{date_col}': {e}")
            # оставим как есть (без индекса)
    else:
        # Если дата-колонка не найдена, попробуем, нет ли уже DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"[load_data] ⚠️ В {file_path} не найдена колонка даты среди {DATE_CANDIDATES}")

    return df


# Быстрый тест
if __name__ == "__main__":
    print("DATA_DIR:", DATA_DIR)
    sample = os.path.join(DATA_DIR, "AFLT_D1_MOEX.csv")
    d = load_data(sample)  # по пути
    print("AFLT by path:", None if d is None else d.shape)

    d2 = load_data("AFLT")  # по тикеру
    print("AFLT by ticker:", None if d2 is None else d2.shape)
