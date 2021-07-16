import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, Input

import load_data

(X_train, y_train), (X_test, y_test) = load_data.load_data()

