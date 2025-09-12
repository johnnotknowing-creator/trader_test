from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/feature_selector.py
from _core.data_loader import load_data
from _core.feature_generator import create_features

CONFIG = {
    "HORIZON": 20,
    "TOP_N_FEATURES": 25,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Отбор признаков с помощью LightGBM.")
    parser.add_argument('--model_name', type=str, required=True, help='Уникальное имя для сохранения списка признаков.')
    parser.add_argument('--universe_file', type=str, required=True, help='Путь к файлу со списком тикеров.')
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "1_data")
    universe_path = os.path.join(PROJECT_ROOT, args.universe_file)
    
    try:
        tickers_df = pd.read_csv(universe_path)
        all_csv_files = [os.path.join(data_dir, f"{ticker}_D1_MOEX.csv") for ticker in tickers_df['ticker'].dropna().unique()]
        print(f"Найдено {len(all_csv_files)} тикеров для обработки.")
    except FileNotFoundError:
        print(f"❌ Файл со списком тикеров не найден: {universe_path}"); exit()

    print("--- Подготовка данных по индексу IMOEX ---")
    index_raw_df = load_data(os.path.join(data_dir, "IMOEX.csv"))
    index_df = index_raw_df.copy()
    for period in [10, 20, 50]:
        index_df[f'index_roc_{period}'] = ta.roc(index_df['close'], length=period)
    index_df.reset_index(inplace=True)

    print("--- Сбор данных для обучения селектора ---")
    all_train_dfs = []
    for file_path in tqdm(all_csv_files, desc="Сбор данных"):
        df = load_data(file_path)
        if df is None or len(df) < 500: continue
        
        df_features = create_features(df, index_df=index_df, horizon=CONFIG["HORIZON"])
        if df_features.empty: continue
            
        df_features['datetime'] = pd.to_datetime(df_features['datetime'])
        train_df = df_features[df_features['datetime'] < (df_features['datetime'].max() - relativedelta(years=1))].copy()
        all_train_dfs.append(train_df)
    
    if not all_train_dfs: 
        print("❌ Не удалось собрать данные для анализа."); exit()
        
    combined_df = pd.concat(all_train_dfs, ignore_index=True)
    print(f"✅ Данные собраны. Общее количество записей: {len(combined_df)}")

    print("--- Обучение LightGBM и отбор признаков ---")
    feature_columns = [col for col in combined_df.columns if col not in ['datetime', 'target']]
    X_train = combined_df[feature_columns]
    y_train = combined_df['target']
    
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    top_features_df = feature_importances.sort_values('importance', ascending=False).head(CONFIG["TOP_N_FEATURES"])
    
    print("\n--- Рейтинг признаков по версии LightGBM ---")
    print(top_features_df.to_string())
    
    top_features_list = top_features_df['feature'].tolist()
    
    results_dir = os.path.join(PROJECT_ROOT, "2_results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"selected_features_{args.model_name}.json")
    
    with open(file_path, 'w') as f:
        json.dump(top_features_list, f, indent=4)
        
    print(f"\n✅ Список из {len(top_features_list)} признаков сохранен в: {file_path}")
