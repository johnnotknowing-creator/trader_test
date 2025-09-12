from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
# _tools/train_universal_lstm.py

CONFIG = {"batch_size": 4096*2, "num_runs": 30}

def parse_tfrecord_fn(example_proto, lookback_period, num_features):
    feature_description = {'sequence': tf.io.FixedLenFeature([], tf.string), 'target': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_description)
    sequence = tf.reshape(tf.io.parse_tensor(example['sequence'], out_type=tf.float32), (lookback_period, num_features))
    return sequence, tf.cast(example['target'], tf.float32)

def build_lstm_model(lookback_period, num_features):
    inputs = Input(shape=(lookback_period, num_features))
    lstm_out = LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(inputs)
    attention_out = Attention()([lstm_out, lstm_out])
    lstm_out_2 = LSTM(128, kernel_regularizer=regularizers.l2(0.001))(attention_out)
    dropout_1 = Dropout(0.25)(lstm_out_2)
    dense_1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(dropout_1)
    outputs = Dense(1, activation='sigmoid')(dense_1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение универсальной LSTM модели.")
    parser.add_argument('--model_name', type=str, required=True, help='Уникальное имя модели')
    args = parser.parse_args()

    data_proc_dir = os.path.join(PROJECT_ROOT, "2_results", f"preprocessed_data_{args.model_name}")
    metadata_path = os.path.join(data_proc_dir, 'metadata.json')
    try:
        with open(metadata_path, 'r') as f: metadata = json.load(f)
    except FileNotFoundError:
        print(f"❌ Ошибка: датасет для модели '{args.model_name}' не найден."); exit()

    num_features, lookback_period = metadata['num_features'], metadata['lookback_period']
    num_train, num_test = metadata['num_train_examples'], metadata['num_test_examples']
    
    train_ds = tf.data.TFRecordDataset(os.path.join(data_proc_dir, 'train.tfrecord')).map(lambda x: parse_tfrecord_fn(x, lookback_period, num_features))
    test_ds = tf.data.TFRecordDataset(os.path.join(data_proc_dir, 'test.tfrecord')).map(lambda x: parse_tfrecord_fn(x, lookback_period, num_features))
    
    train_pipeline = train_ds.shuffle(10000).repeat().batch(CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
    test_pipeline = test_ds.batch(CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
    
    best_accuracy, all_accuracies = 0.0, []
    model_path = os.path.join(PROJECT_ROOT, "2_results", f"model_{args.model_name}.keras")

    for i in range(CONFIG['num_runs']):
        print(f"\\n{'='*20} ЗАПУСК ОБУЧЕНИЯ #{i+1}/{CONFIG['num_runs']} ДЛЯ '{args.model_name}' {'='*20}")
        model = build_lstm_model(lookback_period, num_features)
        early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True, verbose=1)
        steps_per_epoch = math.ceil(num_train / CONFIG['batch_size'])
        validation_steps = math.ceil(num_test / CONFIG['batch_size'])
        model.fit(train_pipeline, epochs=100, validation_data=test_pipeline, callbacks=[early_stopping], steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
        
        _, accuracy = model.evaluate(test_pipeline, steps=validation_steps, verbose=0)
        all_accuracies.append(accuracy)
        if accuracy > best_accuracy:
            print(f"✅ Новый лучший результат: {accuracy * 100:.2f}%"); best_accuracy = accuracy
            model.save(model_path); print(f"✅ Модель сохранена в: {model_path}")

    print(f"\\n{'='*25} ИТОГОВАЯ СТАТИСТИКА {'='*25}")
    all_accuracies_percent = [acc * 100 for acc in all_accuracies]
    print(f"Результаты по {CONFIG['num_runs']} запускам: {[f'{acc:.2f}%' for acc in all_accuracies_percent]}")
    print(f"Средняя точность: {np.mean(all_accuracies_percent):.2f}% (std: {np.std(all_accuracies_percent):.2f}%)")
    if os.path.exists(model_path):
        print(f"\\n✅ Финальная лучшая модель с точностью {best_accuracy*100:.2f}% сохранена.")
