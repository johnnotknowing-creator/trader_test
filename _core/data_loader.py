# _core/data_loader.py
import os
import pandas as pd
from typing import Optional, List
from pathlib import Path

from _core.paths import DATA_DIR

DATE_CANDIDATES: List[str] = ["datetime", "date", "TRADEDATE", "DATE", "Datetime"]

def _resolve_path(ticker_or_path: str) -> Optional[str]:
    if os.path.exists(ticker_or_path):
        return ticker_or_path
    path_in_data = DATA_DIR / ticker_or_path
    if path_in_data.exists():
        return str(path_in_data)
    ticker = str(ticker_or_path).upper().strip()
    candidates = [f"{ticker}.csv", f"{ticker}_D1.csv", f"{ticker}_D1_MOEX.csv"]
    for fname in candidates:
        p = DATA_DIR / fname
        if p.exists():
            return str(p)
    return None

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    # Эта функция остается без изменений
    cols = {c.lower() for c in df.columns}
    for candidate in DATE_CANDIDATES:
        if candidate.lower() in cols:
            for original_col in df.columns:
                if original_col.lower() == candidate.lower():
                    return original_col
    return None

def load_data(ticker_or_path: str) -> Optional[pd.DataFrame]:
    file_path = _resolve_path(ticker_or_path)
    if not file_path:
        return None
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        print(f"[load_data] ❌ Ошибка чтения {file_path}: {e}")
        return None

    if df is None or df.empty:
        return None
        
    # Убираем дубликаты колонок (как в BRENT.csv), оставляя первое вхождение
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    # Приводим все имена колонок к нижнему регистру для унификации
    df.columns = [str(col).lower() for col in df.columns]

    date_col = _detect_date_column(df)
    if date_col:
        try:
            # --- 👇👇👇 ГЛАВНОЕ ИЗМЕНЕНИЕ 👇👇👇 ---
            # 1. Преобразуем дату в единый формат и убираем время (.dt.normalize())
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
            # 2. Переименовываем колонку в 'datetime'
            df = df.rename(columns={date_col: "datetime"})
            # 3. Удаляем строки, где дата не распозналась
            df.dropna(subset=['datetime'], inplace=True)
            # 4. Сортируем и сбрасываем индекс, но НЕ устанавливаем datetime как индекс
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            # --- 👆👆👆 КОНЕЦ ИЗМЕНЕНИЯ 👆👆👆 ---
        except Exception as e:
            print(f"[load_data] ❌ Ошибка обработки даты в {file_path}: {e}")
            return None
            
    return df