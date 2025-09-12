from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/feature_selector_perm.py
from _core.data_loader import load_data
from _core.feature_generator import create_features
from sklearn.model_selection import TimeSeriesSplit

CONFIG = {
    "HORIZON": 20,
    "TOP_N_FEATURES": 25,
    "PERMUTATION_REPEATS": 10,   # количество повторов PI на валидации
    "N_SPLITS": 5,               # число временных фолдов
    "PURGE_GAP_DAYS": 20,        # защита от утечек между train и val
    "CORR_THRESHOLD": 0.90,      # порог взаимной корреляции признаков
    "MIN_LEN_TICKER": 500,       # минимум строк на тикер
    "BOOTSTRAPS": 1,             # лёгкая устойчивость: сколько прогонов с субсемплингом времени
    "BOOTSTRAP_FRACTION": 1.0    # доля данных во времени (<=1.0). 1.0 = выключено
}

def _build_index_df(data_dir):
    idx_raw = load_data(os.path.join(data_dir, "IMOEX.csv"))
    if idx_raw is None or idx_raw.empty:
        raise FileNotFoundError("Не найден или пуст 1_data/IMOEX.csv")
    idx = idx_raw.copy()
    for p in [10, 20, 50]:
        idx[f'index_roc_{p}'] = ta.roc(idx['close'], length=p)
    idx.reset_index(inplace=True)
    return idx

def _collect_training_frame(universe_path, data_dir, index_df, horizon, min_len=500):
    try:
        tickers_df = pd.read_csv(universe_path)
        tickers = tickers_df['ticker'].dropna().unique()
    except FileNotFoundError:
        print(f"❌ Файл со списком тикеров не найден: {universe_path}")
        return None

    all_train = []
    for ticker in tqdm(tickers, desc="Сбор данных для селектора"):
        df = load_data(os.path.join(data_dir, f"{ticker}_D1_MOEX.csv"))
        if df is None or len(df) < min_len:
            continue
        feats = create_features(df, index_df=index_df, horizon=horizon)
        if feats.empty:
            continue
        feats['datetime'] = pd.to_datetime(feats['datetime'])
        # берём «трейн без последнего года»
        cutoff = feats['datetime'].max() - relativedelta(years=1)
        train_df = feats[feats['datetime'] < cutoff].copy()
        if len(train_df) == 0:
            continue
        all_train.append(train_df)
    if not all_train:
        return None
    combined = pd.concat(all_train, ignore_index=True)
    return combined

def _purge_train_indices(tr_idx, va_idx, gap):
    """Убираем последние 'gap' наблюдений из train, чтобы разнести во времени от начала val."""
    if gap <= 0:
        return tr_idx
    max_train = tr_idx.max()
    min_val = va_idx.min()
    # если между max_train и min_val меньше gap, обрежем train
    cutoff = min_val - gap
    return tr_idx[tr_idx <= cutoff]

def _permutation_importance_oos(X, y, n_splits=5, gap=20, repeats=10, scoring='f1_weighted',
                                model_kwargs=None, n_jobs=-1, boots=1, bootstrap_frac=1.0, random_state=42):
    """
    Возвращает DataFrame с усреднёнными по OOS-фолдам важностями признаков.
    Можно включить лёгкий бутстрап по времени: берём центральный кусок ряда.
    """
    rng = np.random.default_rng(random_state)
    feats = X.columns.tolist()
    agg_list = []

    for b in range(boots):
        Xb, yb = X, y
        if bootstrap_frac < 1.0:
            # time-subset: берём центральное окно длиной frac
            n = len(X)
            win = int(max(100, np.floor(n * bootstrap_frac)))
            start = int(np.floor((n - win) / 2))
            Xb = X.iloc[start:start+win]
            yb = y.iloc[start:start+win]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_imps = []

        for fold, (tr_idx_raw, va_idx) in enumerate(tscv.split(Xb)):
            tr_idx = _purge_train_indices(tr_idx_raw, va_idx, gap)
            if len(tr_idx) == 0:
                continue

            X_tr, y_tr = Xb.iloc[tr_idx], yb.iloc[tr_idx]
            X_va, y_va = Xb.iloc[va_idx], yb.iloc[va_idx]

            # LightGBM на train
            params = {"random_state": (random_state + b + fold), "n_jobs": -1}
            if model_kwargs:
                params.update(model_kwargs)
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=0)])

            # Permutation Importance на ВАЛИДАЦИИ (OOS)
            result = permutation_importance(
                model, X_va, y_va,
                n_repeats=repeats, random_state=(random_state + 123*b + fold),
                scoring=scoring, n_jobs=n_jobs
            )
            fold_imps.append(pd.DataFrame({
                "feature": feats,
                f"imp_fold{fold}": result.importances_mean
            }))

        if not fold_imps:
            continue

        imp_df = fold_imps[0]
        for extra in fold_imps[1:]:
            imp_df = imp_df.merge(extra, on="feature", how="inner")
        imp_df["imp_mean"] = imp_df.filter(like="imp_fold").mean(axis=1)
        imp_df["boot_id"] = b
        agg_list.append(imp_df[["feature", "imp_mean"]])

    if not agg_list:
        raise RuntimeError("Не удалось посчитать permutation importance ни на одном фолде/бутстрапе.")

    out = pd.concat(agg_list, ignore_index=True)
    out = out.groupby("feature", as_index=False)["imp_mean"].mean().sort_values("imp_mean", ascending=False)
    return out

def _correlation_filter(feature_order, corr_df, threshold=0.90):
    selected = []
    for f in feature_order:
        if f not in corr_df.columns:
            selected.append(f)
            continue
        ok = True
        for s in selected:
            if s in corr_df.columns and abs(corr_df.loc[f, s]) >= threshold:
                ok = False
                break
        if ok:
            selected.append(f)
    return selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Отбор признаков методом OOS Permutation Importance (time-aware).")
    # Совместимость интерфейса с feature_selector.py
    parser.add_argument('--model_name', type=str, required=True, help='Уникальное имя для сохранения списка признаков')
    parser.add_argument('--universe_file', type=str, required=True, help='Путь к файлу со списком тикеров')

    # Доп. параметры (опционально)
    parser.add_argument('--top_n', type=int, default=CONFIG["TOP_N_FEATURES"], help='Сколько признаков сохранить')
    parser.add_argument('--scoring', type=str, default='f1_weighted', help='Метрика для permutation_importance')
    parser.add_argument('--n_splits', type=int, default=CONFIG["N_SPLITS"], help='Число временных фолдов')
    parser.add_argument('--purge_gap', type=int, default=CONFIG["PURGE_GAP_DAYS"], help='Разрыв (дней) между train и val')
    parser.add_argument('--repeats', type=int, default=CONFIG["PERMUTATION_REPEATS"], help='Повторы permutation importance')
    parser.add_argument('--corr_threshold', type=float, default=CONFIG["CORR_THRESHOLD"], help='Порог |ρ| для корр-фильтра')
    parser.add_argument('--min_len_ticker', type=int, default=CONFIG["MIN_LEN_TICKER"], help='Мин. длина истории тикера')
    parser.add_argument('--boots', type=int, default=CONFIG["BOOTSTRAPS"], help='Бутстрапы по времени (устойчивость)')
    parser.add_argument('--bootstrap_frac', type=float, default=CONFIG["BOOTSTRAP_FRACTION"], help='Доля времени в бутстрапе (<=1.0)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='n_jobs для permutation_importance')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "1_data")
    results_dir = os.path.join(PROJECT_ROOT, "2_results")
    os.makedirs(results_dir, exist_ok=True)

    # 0) Индекс и сбор объединённого train-фрейма
    print("--- Этап 0: Подготовка данных по индексу IMOEX ---")
    index_df = _build_index_df(data_dir)

    print("\n--- Этап 1: Сбор трейновых данных для селектора ---")
    universe_path = os.path.join(PROJECT_ROOT, args.universe_file)
    combined_df = _collect_training_frame(universe_path, data_dir, index_df,
                                          horizon=CONFIG["HORIZON"], min_len=args.min_len_ticker)
    if combined_df is None or combined_df.empty:
        print("❌ Не удалось собрать данные для анализа."); exit(1)

    # Формируем X / y
    feature_columns = [c for c in combined_df.columns if c not in ['datetime', 'target']]
    X = combined_df[feature_columns]
    y = combined_df['target']

    # 2) OOS Permutation Importance по фолдам времени (+ лёгкий бутстрап при желании)
    print("\n--- Этап 2: Расчёт OOS Permutation Importance ---")
    perm_df = _permutation_importance_oos(
        X, y,
        n_splits=args.n_splits,
        gap=args.purge_gap,
        repeats=args.repeats,
        scoring=args.scoring,
        model_kwargs={"n_jobs": -1},
        n_jobs=args.n_jobs,
        boots=args.boots,
        bootstrap_frac=args.bootstrap_frac,
        random_state=42
    )

    # 3) Корреляционный фильтр (по объединённым данным)
    print("\n--- Этап 3: Корреляционный фильтр ---")
    # чтобы не падать на редких фичах, оставим те, что действительно в X
    ordered_feats = [f for f in perm_df['feature'].tolist() if f in X.columns]
    corr = X[ordered_feats].corr().abs()
    filtered = _correlation_filter(ordered_feats, corr, threshold=args.corr_threshold)

    # 4) Топ-N и сохранения
    top_features = filtered[:args.top_n]
    imp_report = perm_df.copy()
    imp_report['selected'] = imp_report['feature'].isin(top_features).astype(int)

    out_json = os.path.join(results_dir, f"selected_features_{args.model_name}.json")
    with open(out_json, 'w') as f:
        json.dump(top_features, f, indent=4)
    print(f"\n✅ Список из {len(top_features)} признаков сохранён в: {out_json}")

    out_csv = os.path.join(results_dir, f"perm_importance_{args.model_name}.csv")
    imp_report.to_csv(out_csv, index=False)
    print(f"📄 Полный отчёт по важностям сохранён: {out_csv}")

    print("\n--- Топ признаков ---")
    print(pd.DataFrame({"feature": top_features}).to_string(index=False))
