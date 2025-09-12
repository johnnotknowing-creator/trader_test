# _core/feature_generator.py

import pandas as pd
import numpy as np
import talib
from ta import add_all_ta_features
from ta.utils import dropna

# ---------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---------------- #

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия столбцов к единому формату."""
    rename_dict = {
        "datetime": "datetime",
        "date": "datetime",
        "time": "datetime",

        "open": "open", "Open": "open", "OPEN": "open",
        "high": "high", "High": "high", "HIGH": "high",
        "low": "low", "Low": "low", "LOW": "low",
        "close": "close", "Close": "close", "CLOSE": "close",
        "vol": "volume", "Vol": "volume", "VOL": "volume",
        "volume": "volume", "Volume": "volume", "VOLUME": "volume",
    }
    df = df.rename(columns=lambda x: rename_dict.get(x, x))
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет технические индикаторы (TA-Lib, ta, pandas) и derived-фичи.
    """
    df = df.copy()

    if not {"close", "volume"}.issubset(df.columns):
        print("[add_extra_features] Пропуск: нет 'close' или 'volume'")
        return df

    # --- Momentum
    df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)
    df["stoch_k"], df["stoch_d"] = talib.STOCH(
        df["high"], df["low"], df["close"],
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    df["willr"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod=14)

    # --- Trend
    df["ema_20"] = talib.EMA(df["close"], timeperiod=20)
    df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
    df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(
        df["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )

    # --- Volatility
    df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
        df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # --- Volume-based
    df["obv"] = talib.OBV(df["close"], df["volume"])

    # --- Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)

    # --- Rolling stats
    df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
    df["volume_mean_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()

    # --- Полный пакет ta (50+ индикаторов)
    df = add_all_ta_features(
        df,
        open="open", high="high", low="low", close="close", volume="volume",
        fillna=True
    )

    return df


# ---------------- ОСНОВНАЯ ФУНКЦИЯ ---------------- #

def create_features(df: pd.DataFrame, index_df: pd.DataFrame = None, horizon: int = 1) -> pd.DataFrame:
    """
    Основная функция для генерации фичей.
    df: DataFrame с колонками ['open','high','low','close','volume']
    index_df: индексный ряд (например IMOEX), если доступен
    horizon: горизонт прогнозирования (оставлен для совместимости)
    """
    if df is None or df.empty:
        raise ValueError("create_features получил пустой DataFrame")

    # 1. Нормализация колонок
    features_df = normalize_columns(df.copy())

    # 2. Derived features
    features_df = add_extra_features(features_df)

    # 3. Относительные фичи (если есть индекс)
    if index_df is not None and not index_df.empty:
        index_df = normalize_columns(index_df.copy())
        features_df["index_close"] = index_df["close"]
        features_df["rel_close"] = features_df["close"] / index_df["close"]
        features_df["rel_volume"] = features_df["volume"] / index_df["volume"]

    # 4. Clean
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()

    return features_df


# ---------------- ТЕСТ ---------------- #
if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir)
    file_to_test = os.path.join(project_root, "1_data", "GAZP_D1_MOEX.csv")

    print(f"Тестирую feature_generator на {file_to_test}")
    df = pd.read_csv(file_to_test, parse_dates=["datetime"], index_col="datetime")
    feats = create_features(df)

    print("Размер:", feats.shape)
    print("Признаков:", len(feats.columns))
    print("Первые колонки:", feats.columns[:20].tolist())
