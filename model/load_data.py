from tensorflow.keras.datasets import imdb


def load_data():

  top_words = 5000   # Vocab Size

  (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

  return (X_train, y_train), (X_test, y_test)