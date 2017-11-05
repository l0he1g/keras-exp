from keras import optimizers, losses
from keras.layers import Embedding, GlobalAveragePooling1D
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

import conf
from fasttext_exp.baiduQA import BaiduQA


def run(state_size=50):
  print("state_size=%d" % state_size)
  data = BaiduQA(conf.baiduQA_pt)
  train_ys, test_ys, train_xs, test_xs = data.split()
  print("n(train)=%d, n(test)=%d" % (len(train_xs), len(test_xs)))

  model = build_fasttext_model(data.voca_size,
                               data.label_num,
                               state_size)

  model.fit(pad_sequences(train_xs),
            train_ys,
            batch_size=128,
            epochs=20,
            verbose=2,
            validation_data=(pad_sequences(test_xs), test_ys))


def build_fasttext_model(voca_size, label_num, state_size):
  model = Sequential([
    Embedding(voca_size, state_size),
    GlobalAveragePooling1D(),
    Dense(100, activation="relu"),
    Dropout(0.3),
    Dense(label_num, activation="softmax")
  ])
  model.compile(optimizer="adam",
                loss=losses.sparse_categorical_crossentropy,
                metrics=["accuracy"])
  return model


if __name__ == '__main__':
  run()
