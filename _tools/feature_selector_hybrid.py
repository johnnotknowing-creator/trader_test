from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/feature_selector_hybrid.py
from _core.data_loader import load_data
from _core.feature_generator import create_features

def load_selected_features(model_name):
    path = os.path.join(PROJECT_ROOT, "2_results", f"selected_features_{model_name}.json")
    with open(path, "r") as f:
        feats = json.load(f)
    return feats

def borda_rank(list_a, list_b, penalty=None):
    """
    Присваиваем каждой фиче суммарный ранг: rankA + rankB.
    Отсутствующие в списке получают штраф = penalty (по умолчанию max(lenA,lenB)+5).
    """
    lenA, lenB = len(list_a), len(list_b)
    if penalty is None:
        penalty = max(lenA, lenB) + 5

    ranks = {}
    for i, f in enumerate(list_a):
        ranks.setdefault(f, 0)
        ranks[f] += (i + 1)
    for i, f in enumerate(list_b):
        ranks.setdefault(f, 0)
        ranks[f] += (i + 1)
    # Накажем отсутствующих
    union = set(list_a) | set(list_b)
    for f in union:
        if f not in list_a:
            ranks[f] += penalty
        if f not in list_b:
            ranks[f] += penalty
    # В отсортированный список (меньше — лучше)
    ranked = sorted(ranks.items(), key=lambda x: x[1])
    return ranked  # [(feature, score), ...]

def build_index_df(data_dir):
    index_raw_df = load_data(os.path.join(data_dir, "IMOEX.csv"))
    idx = index_raw_df.copy()
    for p in [10, 20, 50]:
        idx[f'index_roc_{p}'] = ta.roc(idx['close'], length=p)
    idx.reset_index(inplace=True)
    return idx

def sample_universe(universe_path, k=60):
    tickers = pd.read_csv(universe_path)['ticker'].dropna().unique().tolist()
    if len(tickers) > k:
        rng = np.random.default_rng(42)
        tickers = sorted(rng.choice(tickers, size=k, replace=False).tolist())
    return tickers

def build_feature_frame(tickers, data_dir, index_df, horizon=20, min_len=500):
    frames = []
    for t in tqdm(tickers, desc="Сбор данных для корр-матрицы"):
        df = load_data(os.path.join(data_dir, f"{t}_D1_MOEX.csv"))
        if df is None or len(df) < min_len:
            continue
        feats = create_features(df, index_df=index_df, horizon=horizon)
        if feats.empty:
            continue
        # Возьмём «старую» часть, чтобы не тащить всё (ускорение)
        frames.append(feats.iloc[:-252])
    if not frames:
        return None
    big = pd.concat(frames, ignore_index=True)
    return big

def correlation_filter(features_ordered, corr_df, threshold=0.9):
    """Идём по отсортированному списку; если фича сильно коррелирует с уже выбранной — пропускаем."""
    selected = []
    for f in features_ordered:
        if f not in corr_df.columns:
            # если вдруг фичи нет в данных для корреляции — всё равно берём
            selected.append(f)
            continue
        ok = True
        for s in selected:
            if s in corr_df.columns:
                if abs(corr_df.loc[f, s]) >= threshold:
                    ok = False
                    break
        if ok:
            selected.append(f)
    return selected

def main(args):
    data_dir = os.path.join(PROJECT_ROOT, "1_data")
    results_dir = os.path.join(PROJECT_ROOT, "2_results")
    os.makedirs(results_dir, exist_ok=True)

    # 1) Считываем два списка фич
    feats_a = load_selected_features(args.model_name_a)
    feats_b = load_selected_features(args.model_name_b)

    # 2) Агрегируем ранги (Borda)
    ranked = borda_rank(feats_a, feats_b)
    union_order = [f for f, _ in ranked]

    # 3) (Опционально) считаем корреляции на объединённом фрейме
    if not args.no_corr_filter:
        print("\n--- Построение корр.матрицы для фильтрации ---")
        index_df = build_index_df(data_dir)
        tickers = sample_universe(os.path.join(PROJECT_ROOT, args.universe_file), k=args.sample_tickers)
        feat_frame = build_feature_frame(tickers, data_dir, index_df, horizon=20)
        if feat_frame is None:
            print("⚠️ Не удалось построить общий фрейм для корреляций. Пропускаю корр-фильтр.")
            corr_df = None
        else:
            # Берём только фичи из объединённого набора, которые реально присутствуют
            cols = [c for c in union_order if c in feat_frame.columns]
            corr_df = feat_frame[cols].corr()
    else:
        corr_df = None

    # 4) Применяем корр-фильтр
    if corr_df is not None:
        filtered = correlation_filter(union_order, corr_df, threshold=args.corr_threshold)
    else:
        filtered = union_order

    # 5) Обрезаем до TOP_N и сохраняем
    final_feats = filtered[:args.top_n]

    out_json = os.path.join(results_dir, f"selected_features_{args.out_model_name}.json")
    with open(out_json, "w") as f:
        json.dump(final_feats, f, indent=4)
    print(f"\n✅ Гибридный список из {len(final_feats)} фич сохранён: {out_json}")

    # 6) Отчёт по рангам
    rep = pd.DataFrame(ranked, columns=["feature", "borda_rank"])
    rep["in_A"] = rep["feature"].isin(feats_a).astype(int)
    rep["in_B"] = rep["feature"].isin(feats_b).astype(int)
    rep["selected"] = rep["feature"].isin(final_feats).astype(int)
    out_csv = os.path.join(results_dir, f"hybrid_rank_{args.out_model_name}.csv")
    rep.to_csv(out_csv, index=False)
    print(f"📄 Отчёт по агрегированным рангам сохранён: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Гибридный отбор признаков (Borda + корр-фильтр).")
    parser.add_argument('--model_name_a', type=str, required=True, help='Имя набора фич #A (без selected_features_*.json)')
    parser.add_argument('--model_name_b', type=str, required=True, help='Имя набора фич #B')
    parser.add_argument('--out_model_name', type=str, required=True, help='Имя выхода для нового списка фич')
    parser.add_argument('--universe_file', type=str, default='universe.csv', help='Файл со списком тикеров')
    parser.add_argument('--top_n', type=int, default=25, help='Сколько фич оставить')
    parser.add_argument('--corr_threshold', type=float, default=0.9, help='Порог |ρ| для корр-фильтра')
    parser.add_argument('--sample_tickers', type=int, default=60, help='Сколько тикеров брать для оценки корреляций')
    parser.add_argument('--no_corr_filter', action='store_true', help='Отключить корреляционный фильтр')
    args = parser.parse_args()
    main(args)
