# trader_test/_tools/clean_timeseries.py
# Очистка дневных OHLCV по всем бумагам и формирование "плотных" сегментов без
# искусственного склеивания длинных календарных разрывов.
# Также опционально: заполнение дневного ряда ключевой ставки ЦБР по выходным (forward-fill).

from __future__ import annotations

import argparse
from typing import Optional, List, Tuple

from _core.paths import PROJECT_ROOT, RESULTS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *  # numpy as np, pandas as pd, tqdm, os, time (если чего-то нет в libs — импортируй локально)
from glob import glob

load_dotenv_if_exists()
ensure_dirs()

REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]


# ---------- чтение/нормализация ----------
def read_ohlcv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # имена → нижний регистр
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # привести имя времени к "datetime"
    if "datetime" not in df.columns:
        for alt in ("date", "timestamp", "time", "tradedate"):
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"})
                break

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    # tz-naive для csv
    df["datetime"] = df["datetime"].dt.tz_convert(None)
    return df[REQUIRED_COLS].copy()


def drop_blank_ohlcv_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляем строки, где все цены NaN и volume==0/NaN."""
    prices_nan = df[["open", "high", "low", "close"]].isna().all(axis=1)
    zero_vol = df["volume"].fillna(0).eq(0)
    return df.loc[~(prices_nan & zero_vol)].copy()


# ---------- разбиение на сегменты по большим календарным разрывам ----------
def split_into_segments(
    df: pd.DataFrame,
    *,
    gap_ok_days: int = 3,
    gap_buffer_days: int = 1,
    min_seg_len: int = 5,
) -> pd.DataFrame:
    """
    gap_ok_days: максимум допустимый календарный зазор между двумя подряд датами,
                 иначе считаем "большой разрыв".
                 Пример: пятница → понедельник = 3 дня, это ок.
    gap_buffer_days: сколько дней по краям сегмента дополнительно убрать около "больших разрывов".
                     Это как раз "не брать дни рядом с пустыми местами".
    min_seg_len: минимальная длина сегмента после обрезки; короткие сегменты выкидываем.
    """
    if df.empty:
        return df

    # нормализованные даты без TZ
    dates = pd.to_datetime(df["datetime"]).dt.normalize()
    diffs = dates.diff().dt.days.fillna(0).astype(int)

    # индексы, где большой разрыв (строка i идёт ПОСЛЕ разрыва i-1→i)
    big_gap_idx = df.index[diffs > gap_ok_days].to_list()

    # границы сегментов [start, end] по индексам df
    bounds: List[Tuple[int, int]] = []
    start_idx = df.index[0]
    for i in big_gap_idx:
        end_idx = df.index[df.index.get_loc(i) - 1]  # предыдущий индекс
        bounds.append((start_idx, end_idx))
        start_idx = i
    bounds.append((start_idx, df.index[-1]))

    kept_parts: List[pd.DataFrame] = []
    for k, (s, e) in enumerate(bounds):
        part = df.loc[s:e]

        # обрезаем края около больших разрывов
        trim_left = gap_buffer_days if k > 0 else 0
        trim_right = gap_buffer_days if k < len(bounds) - 1 else 0

        if trim_left > 0 and len(part) > trim_left:
            part = part.iloc[trim_left:]
        if trim_right > 0 and len(part) > trim_right:
            part = part.iloc[:-trim_right]

        if len(part) >= min_seg_len:
            kept_parts.append(part)

    if not kept_parts:
        return df.iloc[0:0].copy()

    out = pd.concat(kept_parts, ignore_index=True)
    return out


# ---------- основная очистка одного файла ----------
def clean_file(
    path: str,
    *,
    gap_ok_days: int,
    gap_buffer_days: int,
    min_seg_len: int,
) -> tuple[int, int, int]:
    """
    Возвращает (n_before, n_after_blank, n_after_segments)
    """
    df = read_ohlcv(path)
    if df is None:
        return (0, 0, 0)
    n_before = len(df)

    df = drop_blank_ohlcv_rows(df)
    n_after_blank = len(df)

    df = split_into_segments(
        df,
        gap_ok_days=gap_ok_days,
        gap_buffer_days=gap_buffer_days,
        min_seg_len=min_seg_len,
    )
    n_after_segments = len(df)

    if n_after_segments > 0:
        # гарантируем порядок и форматтер даты
        df = df.sort_values("datetime").reset_index(drop=True)
        df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    else:
        # если всё вычистили — не перезаписываем исходник пустым
        pass

    return (n_before, n_after_blank, n_after_segments)


# ---------- заполнение CBR key rate ----------
def fill_cbr_daily(
    cbr_path: str,
    out_path: Optional[str] = None,
    *,
    date_col_candidates: List[str] = ["date", "datetime"],
    value_col_candidates: List[str] = ["value", "rate", "key_rate", "cbr_key_rate"],
) -> Optional[str]:
    """
    Читает файл со ставкой ЦБР (дневной/еженедельный), приводит к дневному ряду
    (resample 'D') и заполняет пропуски forward-fill'ом (ставка действует до изменения).
    """
    if not os.path.exists(cbr_path):
        print(f"[CBR] not found: {cbr_path}")
        return None

    try:
        df = pd.read_csv(cbr_path)
    except Exception as e:
        print(f"[CBR] read failed: {e}")
        return None

    df = df.rename(columns={c: c.lower() for c in df.columns})
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        print("[CBR] no date column recognized")
        return None

    val_col = None
    for c in value_col_candidates:
        if c in df.columns:
            val_col = c
            break
    if val_col is None:
        # если один столбец числовой — попробуем его
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            val_col = numeric_cols[0]
        else:
            print("[CBR] no numeric value column recognized")
            return None

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "cbr_key_rate"})

    # приводим к дневному и ffill — это и заполнит выходные «ставкой недели»
    daily = (df.set_index("date")
               .resample("D")
               .ffill()
               .reset_index())
    daily["date"] = daily["date"].dt.tz_convert(None)

    out = out_path or os.path.join(RESULTS_DIR, "cbr_key_rate_daily.csv")
    daily.to_csv(out, index=False, date_format="%Y-%m-%d")
    return out


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Clean OHLCV time series and (optionally) fill CBR key rate.")
    ap.add_argument("--data-dir", default="1_data", help="Папка с CSV (относительно корня проекта).")
    ap.add_argument("--pattern", default="*_D1_MOEX.csv", help="Глоб-шаблон файлов бумаг.")
    ap.add_argument("--gap-ok", type=int, default=3, help="Максимально допустимый календарный зазор (дней) без разрыва.")
    ap.add_argument("--gap-buffer", type=int, default=2, help="Сколько дней срезать по краям около больших разрывов.")
    ap.add_argument("--min-seg-len", type=int, default=5, help="Минимальная длина сегмента после обрезки.")
    ap.add_argument("--inplace", action="store_true", help="Сохранять поверх исходных файлов (по умолчанию — да).")

    # CBR
    ap.add_argument("--fill-cbr", action="store_true", help="Заполнить дневной ряд ключевой ставки ЦБР (ffill).")
    ap.add_argument("--cbr-path", default=None, help="Путь к исходному CSV со ставкой ЦБР.")
    ap.add_argument("--cbr-out", default=None, help="Куда сохранить дневной ряд ставки ЦБР.")

    args = ap.parse_args()

    data_dir = (PROJECT_ROOT / args.data_dir)
    files = sorted(glob(str(data_dir / args.pattern)))

    report_rows = []
    for path in tqdm(files, desc="Cleaning OHLCV"):
        n_before, n_after_blank, n_after_segments = clean_file(
            path,
            gap_ok_days=args.gap_ok,
            gap_buffer_days=args.gap_buffer,
            min_seg_len=args.min_seg_len,
        )
        report_rows.append({
            "file": os.path.basename(path),
            "rows_before": n_before,
            "rows_after_drop_blank": n_after_blank,
            "rows_after_segments": n_after_segments,
            "dropped_blank": n_before - n_after_blank,
            "dropped_segments": n_after_blank - n_after_segments,
            "total_dropped": n_before - n_after_segments,
        })

    # Отчёт
    rep = pd.DataFrame(report_rows)
    rep_path = RESULTS_DIR / "clean_report.csv"
    if not rep.empty:
        rep.to_csv(rep_path, index=False)
        print(f"\n[OK] Report saved to {rep_path}")
    else:
        print("\n[WARN] No files matched the pattern — report is empty")

    # CBR
    if args.fill_cbr:
        if not args.cbr_path:
            print("[CBR] --cbr-path is required when --fill-cbr is set.")
        else:
            out = fill_cbr_daily(args.cbr_path, args.cbr_out)
            if out:
                print(f"[CBR] Daily key rate saved to: {out}")


if __name__ == "__main__":
    main()
