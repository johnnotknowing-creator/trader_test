# _tools/universal_downloader.py
# Загрузка дневной истории акций с MOEX ISS и сохранение в CSV.
# Сохраняет файлы в формате: <SYMBOL>_D1_MOEX.csv
# Без TensorFlow. Полная пагинация через history.cursor (JSON), без пустых CSV
# и без «пустых» свечей (все OHLC NaN и volume==0 удаляются).

from __future__ import annotations

# Базовые пути/инициализация проекта
from _core.paths import PROJECT_ROOT, ensure_dirs, load_dotenv_if_exists

# Централизованные импорты и утилиты (numpy/pandas/requests/tqdm/…, multiprocessing, partial, relativedelta)
from _core.libs import *

load_dotenv_if_exists()
ensure_dirs()


# ====== Настройки загрузчика ======
class DownloaderConfig:
    def __init__(
        self,
        years: int = 20,
        data_dir: str = "1_data",
        engine: str = "stock",
        market: str = "shares",
        board: str = "TQBR",
        sleep_sec: float = 0.0,
        processes: Optional[int] = None,
    ) -> None:
        self.years = years
        self.data_dir = data_dir
        self.engine = engine
        self.market = market
        self.board = board
        self.sleep_sec = sleep_sec
        self.processes = processes


# ====== Пути/имена файлов (ВАШ ФОРМАТ) ======
def _data_dir_path(cfg: DownloaderConfig) -> str:
    p = PROJECT_ROOT / cfg.data_dir
    os.makedirs(p, exist_ok=True)
    return str(p)


def _csv_path_for_symbol(symbol: str, cfg: DownloaderConfig) -> str:
    # ВАШ формат: ABIO_D1_MOEX.csv
    filename = f"{symbol}_D1_MOEX.csv"
    return os.path.join(_data_dir_path(cfg), filename)


# ====== HTTP-helpers: берём из libs, иначе локальные фоллбеки ======
def _get_session_and_getter():
    """
    Возвращает (session, http_get_text) — либо из _core.libs,
    либо локальные устойчивые фоллбеки (чтобы скрипт не падал).
    """
    if "make_http_session" in globals() and callable(globals().get("make_http_session")) \
       and "http_get_text" in globals() and callable(globals().get("http_get_text")):
        return make_http_session(), http_get_text

    # ---- Локальные фоллбеки (если не добавили в libs) ----
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    def _make_http_session_fallback() -> requests.Session:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0", "Connection": "close"})
        retry = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def _http_get_text_fallback(session: requests.Session, url: str, *, params=None,
                                timeout: float = 30.0, max_attempts: int = 5, sleep_base: float = 0.5) -> str:
        exc = None
        for i in range(max_attempts):
            try:
                resp = session.get(url, params=params, timeout=timeout)
                if resp.status_code == 200:
                    _ = resp.content  # читаем тело, чтобы поймать SSL/Chunked ошибки здесь
                    return resp.text
            except (requests.exceptions.SSLError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError) as e:
                exc = e
            time.sleep(sleep_base * (2 ** i))
            try:
                session.close()
            except Exception:
                pass
            session = _make_http_session_fallback()
        raise RuntimeError(f"HTTP GET failed after retries for {url}: {exc}")

    return _make_http_session_fallback(), _http_get_text_fallback


# ====== Чтение/нормализация существующего CSV ======
def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "datetime" not in df.columns:
        for alt in ("date", "timestamp", "time", "TRADEDATE"):
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"})
                break

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        if pd.api.types.is_numeric_dtype(df["datetime"]):
            vals = df["datetime"].astype("float64")
            unit = "ms" if np.nanmedian(vals) > 1e11 else "s"
            dt = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")
        df["datetime"] = dt

    return df


def _last_date_from_existing(df: pd.DataFrame | None) -> date | None:
    if df is None or df.empty or "datetime" not in df.columns:
        return None
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dropna()
    if dt.empty:
        return None
    return dt.max().date()


# ====== Удаление «пустых» свечей ======
def _drop_blank_ohlcv_rows(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    mask_prices_all_nan = df[["open", "high", "low", "close"]].isna().all(axis=1)
    mask_zero_vol = df["volume"].fillna(0).eq(0)
    mask_blank = mask_prices_all_nan & mask_zero_vol
    return df.loc[~mask_blank].copy()


# ====== Нормализация формата истории ======
def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "TRADEDATE": "datetime",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "tradedate": "datetime",
    }
    df = df.rename(columns=rename)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    cols = ["datetime", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df[cols]
    df = _drop_blank_ohlcv_rows(df)
    return df


# ====== Загрузка истории (JSON + history.cursor, полная пагинация) ======
def _fetch_moex_history_full(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cfg: DownloaderConfig) -> pd.DataFrame:
    base = (
        f"https://iss.moex.com/iss/history/engines/{cfg.engine}/markets/{cfg.market}"
        f"/boards/{cfg.board}/securities/{symbol}.json"
    )

    session, _get_text = _get_session_and_getter()

    params = {
        "from": start.date().isoformat(),
        "till": end.date().isoformat(),
        "iss.meta": "off",
        "lang": "ru",
        "start": 0,
    }

    frames: List[pd.DataFrame] = []
    total = None
    page_size = 100  # уточним из history.cursor

    while True:
        resp = session.get(base, params=params, timeout=30)
        if resp.status_code != 200:
            ok = False
            for _ in range(3):
                time.sleep(0.8)
                resp = session.get(base, params=params, timeout=30)
                if resp.status_code == 200:
                    ok = True
                    break
            if not ok:
                break

        try:
            js = resp.json()
        except Exception:
            time.sleep(0.5)
            continue

        if "history" not in js or "columns" not in js["history"] or "data" not in js["history"]:
            break

        cols = js["history"]["columns"]
        data = js["history"]["data"]
        df = pd.DataFrame(data, columns=cols)
        df = _normalize_history_df(df)
        if not df.empty:
            frames.append(df)

        if total is None and "history.cursor" in js:
            cur = js["history.cursor"]
            if "columns" in cur and "data" in cur and cur["data"]:
                try:
                    ccols = cur["columns"]
                    crow = cur["data"][0]
                    total_idx = ccols.index("TOTAL") if "TOTAL" in ccols else 0
                    psize_idx = ccols.index("PAGESIZE") if "PAGESIZE" in ccols else 1
                    total = int(crow[total_idx])
                    page_size = int(crow[psize_idx])
                except Exception:
                    pass

        params["start"] = int(params["start"]) + page_size
        if total is not None and int(params["start"]) >= total:
            break

        if cfg.sleep_sec:
            time.sleep(cfg.sleep_sec)

    if not frames:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return out.loc[(out["datetime"] >= start.normalize()) & (out["datetime"] <= end.normalize())]


# ====== Слияние и сохранение ======
def _merge_and_save(save_path: str, df_new: pd.DataFrame) -> None:
    df_new = _drop_blank_ohlcv_rows(df_new)
    existing = _safe_read_csv(save_path)
    if existing is None or existing.empty:
        merged = df_new
    else:
        existing = _normalize_history_df(existing)
        merged = pd.concat([existing, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    merged = _drop_blank_ohlcv_rows(merged)
    if merged.empty:
        return  # не создаём пустой файл

    merged["datetime"] = pd.to_datetime(merged["datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    merged.to_csv(save_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")


# ====== Публичные функции ======
def download_and_save_ticker(
    symbol: str,
    *,
    years: int = 20,
    data_dir: str = "1_data",
    board: str = "TQBR",
    sleep_sec: float = 0.0,
) -> Optional[str]:
    """
    Загружает историю по тикеру и сохраняет/дополняет CSV в формате <SYMBOL>_D1_MOEX.csv.
    Если не удалось получить данные — возвращает None (и не создаёт пустых файлов).
    """
    cfg = DownloaderConfig(years=years, data_dir=data_dir, board=board, sleep_sec=sleep_sec)
    save_path = _csv_path_for_symbol(symbol, cfg)

    # tz-aware UTC timestamps, чтобы не было "tz-naive vs tz-aware"
    end_ts = pd.Timestamp.now(tz="UTC").normalize()

    last_date = _last_date_from_existing(_safe_read_csv(save_path))
    if last_date is not None:
        start_ts = pd.Timestamp(last_date, tz="UTC") + pd.Timedelta(days=1)
        if start_ts > end_ts:
            return save_path
    else:
        start_ts = end_ts - relativedelta(years=cfg.years)

    df_new = _fetch_moex_history_full(symbol, start_ts, end_ts, cfg)
    if not df_new.empty:
        _merge_and_save(save_path, df_new)
        return save_path
    else:
        return None  # ничего не сохраняем


def download_all_stocks(
    tickers: List[str],
    *,
    years: int = 20,
    data_dir: str = "1_data",
    board: str = "TQBR",
    processes: Optional[int] = None,
    sleep_sec: float = 0.0,
) -> None:
    """
    Параллельная загрузка/догрузка по списку тикеров.
    """
    if not tickers:
        print("Список тикеров пуст — пропуск.")
        return

    procs = processes or os.cpu_count() or 1
    task = partial(download_and_save_ticker, years=years, data_dir=data_dir, board=board, sleep_sec=sleep_sec)

    # Для надёжности кросс-сред — spawn
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=procs) as pool:
        for _ in tqdm(pool.imap(task, tickers), total=len(tickers), desc="Загрузка акций"):
            pass
