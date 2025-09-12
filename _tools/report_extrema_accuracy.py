from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
# _tools/report_extrema_accuracy.py
from _core.data_loader import load_data
from _core.feature_generator import create_features

def calculate_metrics(predicted_indices, actual_indices, total_actual_events):
    """
    Рассчитывает точность и полноту на основе индексов пиков/впадин.
    """
    hits = 0
    if len(predicted_indices) > 0:
        for p_idx in predicted_indices:
            # Ищем реальное событие в окне +/- 2 дня от предсказанного
            if any(abs(p_idx - a_idx) <= 2 for a_idx in actual_indices):
                hits += 1
    
    precision = hits / len(predicted_indices) if len(predicted_indices) > 0 else 0
    recall = hits / total_actual_events if total_actual_events > 0 else 0
    return precision * 100, recall * 100

def main(args):
    data_dir = os.path.join(PROJECT_ROOT, "1_data")
    results_dir = os.path.join(PROJECT_ROOT, "2_results")
    model_path = os.path.join(results_dir, f"model_{args.model_name}.keras")
    universe_path = os.path.join(PROJECT_ROOT, args.universe_file)
    
    data_proc_dir = os.path.join(results_dir, f"preprocessed_data_{args.model_name}")
    scaler_path = os.path.join(data_proc_dir, "global_scaler.joblib")
    metadata_path = os.path.join(data_proc_dir, "metadata.json")

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(metadata_path, 'r') as f: metadata = json.load(f)
        tickers = pd.read_csv(universe_path)['ticker'].tolist()
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Не найден необходимый файл: {e}"); return

    selected_features = metadata['features']
    lookback_period = metadata['lookback_period']
    
    print("--- Подготовка данных по индексу IMOEX ---")
    index_raw_df = load_data(os.path.join(data_dir, "IMOEX.csv"))
    index_df = index_raw_df.copy()
    for period in [10, 20, 50]:
        index_df[f'index_roc_{period}'] = ta.roc(index_df['close'], length=period)
    index_df.reset_index(inplace=True)

    print("\n--- Расчет расширенных метрик для всех акций ---")
    report_data = []
    for ticker in tqdm(tickers, desc="Анализ акций"):
        df = load_data(os.path.join(data_dir, f"{ticker}_D1_MOEX.csv"))
        if df is None or len(df) < 500: continue
        
        df_features = create_features(df, index_df, horizon=20)
        if df_features.empty or not set(selected_features).issubset(df_features.columns): continue

        df_features.loc[:, selected_features] = scaler.transform(df_features[selected_features])
        test_df = df_features.iloc[-252-lookback_period:]
        
        sequences, targets = [], []
        for i in range(lookback_period, len(test_df)):
            sequences.append(test_df[selected_features].values[i-lookback_period:i])
            targets.append(test_df['target'].values[i])
            
        if not sequences: continue
        X_test, y_test = np.array(sequences), np.array(targets)
        
        probabilities = model.predict(X_test, verbose=0).flatten()
        
        # --- НАЧАЛО НОВОГО БЛОКА РАСЧЕТА МЕТРИК ---
        # 1. Метрики для ПИКОВ (сигналов на покупку)
        predicted_peaks, _ = find_peaks(probabilities, height=0.5, distance=5)
        real_peaks = np.where(y_test == 1)[0]
        peaks_precision, peaks_recall = calculate_metrics(predicted_peaks, real_peaks, len(real_peaks))

        # 2. Метрики для ВПАДИН (сигналов на "непокупку")
        # Ищем пики в инвертированном сигнале
        predicted_lows, _ = find_peaks(1 - probabilities, height=0.5, distance=5)
        real_lows = np.where(y_test == 0)[0]
        lows_precision, lows_recall = calculate_metrics(predicted_lows, real_lows, len(real_lows))
        
        # 3. Средние метрики и F1-Score
        avg_precision = (peaks_precision + lows_precision) / 2
        avg_recall = (peaks_recall + lows_recall) / 2
        
        # F1-считаем по средним, чтобы получить общую оценку
        f1_score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        report_data.append({
            'ticker': ticker,
            'lows_precision': lows_precision,
            'lows_recall': lows_recall,
            'peaks_precision': peaks_precision,
            'peaks_recall': peaks_recall,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f1_score': f1_score
        })
        # --- КОНЕЦ НОВОГО БЛОКА ---
        
    report_df = pd.DataFrame(report_data)
    # Форматируем вывод для наглядности
    for col in report_df.columns:
        if col != 'ticker':
            report_df[col] = report_df[col].map('{:.2f}'.format)

    print("\n--- Итоговые медианные метрики ---")
    # Преобразуем обратно в float для расчета медианы
    report_df_numeric = report_df.drop(columns=['ticker']).astype(float)
    print(report_df_numeric.median())
    
    report_filename = os.path.join(results_dir, f"report_{args.model_name}_on_{os.path.splitext(os.path.basename(args.universe_file))[0]}.csv")
    report_df.to_csv(report_filename, index=False)
    print(f"\n✅ Детальный отчет сохранен в: {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели по экстремумам с детальной разбивкой.")
    parser.add_argument('--model_name', type=str, required=True, help='Имя модели для оценки.')
    parser.add_argument('--universe_file', type=str, required=True, help='Файл со списком тикеров.')
    args = parser.parse_args()
    main(args)
