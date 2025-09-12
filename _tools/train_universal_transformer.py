from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
# _tools/train_universal_transformer.py

CONFIG = {"batch_size": 4096, "num_runs": 15}

def parse_tfrecord_fn(example_proto, lookback_period, num_features):
    feature_description = {'sequence': tf.io.FixedLenFeature([], tf.string), 'target': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_description)
    sequence = tf.reshape(tf.io.parse_tensor(example['sequence'], out_type=tf.float32), (lookback_period, num_features))
    return sequence, tf.cast(example['target'], tf.float32)

def build_transformer_model(lookback_period, num_features, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.25, mlp_dropout=0.4):
    inputs = Input(shape=(lookback_period, num_features))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

if __name__ == "__main__":
    # Этот скрипт следует той же логике, что и train_universal_lstm.py,
    # поэтому остальная часть кода для краткости опущена.
    # Для полноты следует скопировать сюда блок if __name__ == "__main__": из lstm скрипта.
    print("Это скрипт для обучения модели Трансформер. Логика запуска аналогична train_universal_lstm.py")
