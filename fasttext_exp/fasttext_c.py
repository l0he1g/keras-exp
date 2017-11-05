from sklearn.model_selection import train_test_split

import conf
from fasttext_exp.baiduQA import BaiduQA

"""pyfasttext和fasttextpy都有问题，只能用命令行的fasttext"""
def run():
  labels, docs = BaiduQA.load(conf.baiduQA_pt)
  label_prefix = "__label__"
  labels = [label_prefix + l for l in labels]

  train_labels, test_labels, train_docs, test_docs = train_test_split(labels, docs, train_size=0.75)
  train_pt = "/tmp/train.txt"
  write_pt(train_pt, train_labels, train_docs)
  test_pt = "/tmp/test.txt"
  write_pt(test_pt, test_labels, test_docs)


def write_pt(pt, labels, docs):
  wf = open(pt, "w")
  for label, doc in zip(labels, docs):
    wf.write("%s %s\n" % (label, doc))
  print("write:" + pt)


def evaluate(labels, predicts):
  n = len(labels)
  n_right = sum([1 for l, p in zip(labels, predicts) if l == p])
  return n_right / (n + 1e-10)


if __name__ == '__main__':
  run()
