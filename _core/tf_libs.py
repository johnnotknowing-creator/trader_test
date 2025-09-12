# _core/tf_libs.py
# TensorFlow + Keras импорты с "тихой" настройкой логгирования

import os, sys

# Тише до импорта
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # 1=INFO, 2=INFO+WARN, 3=ERROR+
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")

# Заглушаем самый ранний stderr во время импорта TF
__stderr_backup = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
    import tensorflow as tf
finally:
    try:
        sys.stderr.close()
    except Exception:
        pass
    sys.stderr = __stderr_backup

# Тише после импорта (python-логгеры)
try:
    tf.get_logger().setLevel("ERROR")
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Dropout, LSTM, Attention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
