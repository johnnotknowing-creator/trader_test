from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/cross_sectional_features.py
from _core.data_loader import load_data

def build_index(data_dir: str) -> pd.DataFrame:
    idx = load_data(os.path.join(data_dir, "IMOEX.csv"))
    if idx is None or idx.empty:
        raise FileNotFoundError("1_data/IMOEX.csv не найден или пуст")
    idx = idx.reset_index().rename(columns={"datetime": "date"})
    idx = idx.sort_values("date")
    idx["idx_ret"] = np.log(idx["close"] / idx["close"].shift(1))
    # режимы рынка
    idx["idx_rv_20"] = idx["idx_ret"].rolling(20).std()
    idx["idx_trend_up"] = (idx["close"] > idx["close"].rolling(200).mean()).astype(int)
    idx["idx_high_vol"] = (idx["idx_rv_20"] > idx["idx_rv_20"].rolling(252).quantile(0.8)).astype(int)
    return idx[["date","close","idx_ret","idx_rv_20","idx_trend_up","idx_high_vol"]].rename(columns={"close":"index_close"})

def roll_beta(x: pd.Series, y: pd.Series, w: int = 60) -> pd.Series:
    cov = x.rolling(w).cov(y)
    var = y.rolling(w).var()
    return cov / (var + 1e-9)

def compute_cross_sectional(df_all: pd.DataFrame) -> pd.DataFrame:
    # базовые однотикерные расчёты, необходимые для CS-признаков
    df_all["ret1"] = np.log(df_all["close"] / df_all["close"].shift(1))
    df_all["mom_60"] = df_all.groupby("ticker")["close"].transform(lambda s: s / s.shift(60) - 1)
    df_all["hv_20"]  = df_all.groupby("ticker")["ret1"].transform(lambda s: s.rolling(20).std())

    # перцентильные ранги по дате
    df_all["cs_rank_mom_60"] = df_all.groupby("date")["mom_60"].rank(pct=True)
    df_all["cs_rank_vol_20"] = df_all.groupby("date")["hv_20"].rank(pct=True)

    # бета/корреляция к индексу, остаточный момент
    def _beta_grp(g):
        return roll_beta(g["ret1"], g["idx_ret"], w=60)
    def _corr_grp(g):
        return g["ret1"].rolling(60).corr(g["idx_ret"])

    df_all["beta_60"] = df_all.groupby("ticker", group_keys=False).apply(_beta_grp)
    df_all["corr_60"] = df_all.groupby("ticker", group_keys=False).apply(_corr_grp)

    resid = df_all["ret1"] - df_all["beta_60"] * df_all["idx_ret"]
    df_all["resid_mom_20"] = resid.groupby(df_all["ticker"]).transform(lambda s: s.rolling(20).sum())

    return df_all

def main(args):
    data_dir = os.path.join(PROJECT_ROOT, "1_data")
    out_dir = os.path.join(data_dir, "_features_cs")
    os.makedirs(out_dir, exist_ok=True)

    # индекс
    idx = build_index(data_dir)

    # список тикеров
    uni_path = os.path.join(PROJECT_ROOT, args.universe_file)
    tickers = pd.read_csv(uni_path)["ticker"].dropna().unique().tolist()

    # собираем большой датафрейм (date,ticker,close,volume, ...)
    frames = []
    for t in tqdm(tickers, desc="Чтение тикеров"):
        path = os.path.join(data_dir, f"{t}_D1_MOEX.csv")
        df = load_data(path)
        if df is None or df.empty or len(df) < args.min_len:
            continue
        df = df.reset_index().rename(columns={"datetime": "date"})
        df["ticker"] = t
        frames.append(df[["date","ticker","open","high","low","close","volume"]])

    if not frames:
        print("⚠️ Нет данных для расчёта.")
        return

    all_df = pd.concat(frames, ignore_index=True).sort_values(["date","ticker"])
    all_df = all_df.merge(idx, on="date", how="left")  # index_close/idx_ret/режимы

    # считаем кросс-секционные признаки
    all_df = compute_cross_sectional(all_df)

    # сохраняем по тикерам, чтобы потом просто джоинить при подготовке признаков
    cols_out = [
        "date","ticker",
        "cs_rank_mom_60","cs_rank_vol_20",
        "beta_60","corr_60","resid_mom_20",
        "index_close","idx_rv_20","idx_trend_up","idx_high_vol"
    ]
    all_df = all_df[cols_out]

    for t, g in all_df.groupby("ticker"):
        g.to_csv(os.path.join(out_dir, f"{t}_cs.csv"), index=False)

    print(f"✅ Готово. Сохранено в {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Кросс-секционные признаки (ранги, бета, режимы рынка).")
    p.add_argument("--universe_file", type=str, default="universe.csv")
    p.add_argument("--min_len", type=int, default=500)
    args = p.parse_args()
    main(args)
