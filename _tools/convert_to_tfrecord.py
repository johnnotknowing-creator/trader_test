from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
# _tools/convert_to_tfrecord.py
from _core.data_loader import load_data
from _core.feature_generator import create_features

def create_sequences(data, target, lookback_period):
    sequences, targets = [], []
    for i in range(lookback_period, len(data)):
        sequences.append(data[i-lookback_period:i])
        targets.append(target[i])
    return np.array(sequences), np.array(targets)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(sequence, target):
    feature = {
        'sequence': _bytes_feature(tf.io.serialize_tensor(sequence).numpy()),
        'target': _int64_feature(target)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def main(args):
    data_dir = os.path.join(PROJECT_ROOT, "1_data")
    results_dir = os.path.join(PROJECT_ROOT, "2_results")
    universe_path = os.path.join(PROJECT_ROOT, args.universe_file)
    features_path = os.path.join(results_dir, f"selected_features_{args.input_model_name}.json")
    
    try:
        tickers = pd.read_csv(universe_path)['ticker'].tolist()
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Не найден необходимый файл: {e}"); return

    print("--- Этап 0: Подготовка данных по индексу IMOEX ---")
    index_raw_df = load_data(os.path.join(data_dir, "IMOEX.csv"))
    index_df = index_raw_df.copy()
    for period in [10, 20, 50]:
        index_df[f'index_roc_{period}'] = ta.roc(index_df['close'], length=period)
    index_df.reset_index(inplace=True)

    print("--- Этап 1: Сбор данных для обучения скейлера ---")
    all_train_data = []
    for ticker in tqdm(tickers, desc="Сбор данных для скейлера"):
        df = load_data(os.path.join(data_dir, f"{ticker}_D1_MOEX.csv"))
        if df is None or len(df) < 500: continue
        df_features = create_features(df, index_df, horizon=20)
        if df_features.empty: continue
        train_df = df_features.iloc[:-252]
        all_train_data.append(train_df[selected_features])
        
    if not all_train_data: print("❌ Не удалось собрать данные."); return
    combined_train_data = pd.concat(all_train_data)
    
    print("--- Этап 2: Обучение и сохранение скейлера ---")
    scaler = RobustScaler().fit(combined_train_data)
    output_dir = os.path.join(results_dir, f"preprocessed_data_{args.output_model_name}")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "global_scaler.joblib"))
    
    print("--- Этап 3: Создание TFRecord файлов ---")
    num_train, num_test = 0, 0
    with tf.io.TFRecordWriter(os.path.join(output_dir, 'train.tfrecord')) as writer_train, \
         tf.io.TFRecordWriter(os.path.join(output_dir, 'test.tfrecord')) as writer_test:
        for ticker in tqdm(tickers, desc="Создание TFRecord"):
            df = load_data(os.path.join(data_dir, f"{ticker}_D1_MOEX.csv"))
            if df is None or len(df) < 500: continue
            df_features = create_features(df, index_df, horizon=20)
            if not set(selected_features).issubset(df_features.columns): continue

            df_features[selected_features] = scaler.transform(df_features[selected_features])
            train_df, test_df = df_features.iloc[:-252], df_features.iloc[-252:]
            
            X_train, y_train = create_sequences(train_df[selected_features].astype(np.float32).values, train_df['target'].values, 60)
            X_test, y_test = create_sequences(test_df[selected_features].astype(np.float32).values, test_df['target'].values, 60)
            
            for seq, lbl in zip(X_train, y_train): writer_train.write(serialize_example(seq, lbl))
            for seq, lbl in zip(X_test, y_test): writer_test.write(serialize_example(seq, lbl))
            num_train += len(X_train); num_test += len(X_test)
            
    metadata = {
        "input_model_name": args.input_model_name,
        "output_model_name": args.output_model_name,
        "features": selected_features,
        "num_features": len(selected_features),
        "lookback_period": 60,
        "horizon": 20,
        "num_train_examples": num_train,
        "num_test_examples": num_test
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\n✅ Процесс завершен. Данные сохранены в: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка данных и конвертация в TFRecord.")
    parser.add_argument('--input_model_name', type=str, required=True, help='Имя модели, от которой берутся признаки (напр. perm_v1_60_20)')
    parser.add_argument('--output_model_name', type=str, required=True, help='Имя для выходного набора данных (напр. universal_perm_v1)')
    parser.add_argument('--universe_file', type=str, required=True, help='Файл со списком тикеров (напр. universe.csv)')
    args = parser.parse_args()
    main(args)
