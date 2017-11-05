from random import shuffle

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelEncoder

import conf


class BaiduQA:
  def __init__(self, pt):
    self.labels, self.docs = self.load(pt)
    self.doc_num = len(self.docs)
    print("doc_num=%d" % self.doc_num)

    self.label_encoder = LabelEncoder()
    self.ys = self.label_encoder.fit_transform(self.labels)
    self.label_num = len(self.label_encoder.classes_)
    print("label_num=%d" % self.label_num)

    self.tokenizer = Tokenizer(split=" ")
    self.tokenizer.fit_on_texts(self.docs)
    self.xs = self.tokenizer.texts_to_sequences(self.docs)
    self.voca_size = max(self.tokenizer.word_index.values()) + 1
    print("voca_size=%d" % self.voca_size)

  @staticmethod
  def load(pt):
    labels = []
    docs = []
    print("read:" + pt)
    lines = open(pt).readlines()
    shuffle(lines)
    for l in lines:
      label, doc = l.strip().split("\t")[1].split("|")
      labels.append(label)
      docs.append(doc)

    print("n(doc)=%d" % len(labels))
    return labels, docs

  def split(self):
    train_ys, test_ys, train_xs, test_xs = train_test_split(self.ys, self.xs, train_size=0.75)
    return train_ys, test_ys, train_xs, test_xs

  def next_batch(self, batch_size):
    i = 0
    while True:
      if i + batch_size > self.doc_num:
        i == 0
      yield (self.ys[i: i + batch_size], self.xs[i: i + batch_size])


if __name__ == '__main__':
  baiduQA = BaiduQA(conf.baiduQA_pt)
  ys, xs = next(baiduQA.next_padded_batch(20))
  print(ys)
  print(xs)
