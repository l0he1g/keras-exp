from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import accuracy_score

import conf
from scikit.baiduQA import BaiduQA


def run():
  data = BaiduQA(conf.baiduQA_pt)
  train_ys, test_ys, train_xs, test_xs = data.split()
  print("n(train)=%d, n(test)=%d" % (train_xs.shape[0], test_xs.shape[0]))

  lr = LogisticRegression()
  print("begin training")
  lr.fit(train_xs, train_ys)

  train_predicts = lr.predict(train_xs)
  test_predicts = lr.predict(test_xs)

  train_acc = accuracy_score(train_ys, train_predicts)
  test_acc = accuracy_score(test_ys, test_predicts)
  print("train_acc=%f, test_acc=%f" % (train_acc, test_acc))

if __name__ == '__main__':
    run()