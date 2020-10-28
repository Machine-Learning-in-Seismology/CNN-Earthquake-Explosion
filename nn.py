import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from model import build_nn, get_early_stop
from sklearn.utils import class_weight
import pandas as pd
import time

print("Loading data...")
with open("earthquake.pkl", "rb") as f:
    eq = pickle.load(f, encoding='latin1')
with open("explosions.pkl", "rb") as f:
    ex = pickle.load(f, encoding='latin1')

eq_labels = np.ones(eq.shape[0])
ex_labels = np.zeros(ex.shape[0])

X = np.concatenate((ex, eq))
y = np.concatenate((ex_labels, eq_labels))
X = np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
skf.get_n_splits(X, y)
nn = build_nn()
results = pd.DataFrame(columns=['fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")
fold = 0
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    nn.fit(x=X_train, y=y_train,
           batch_size=500, epochs=1000,
           validation_split=0.25, class_weight=class_weights,
           callbacks=[get_early_stop()])

    score = nn.evaluate(X_test, y_test)
    fnr = score[3] / float(score[3] + score[4])
    fpr = score[1] / float(score[1] + score[2])
    print(f"fold: {fold:d} FNR: {fnr:.2f} FPR: {fpr:.2f} ACC: {score[5]:.2f}")
    results.loc[len(results)] = [fold, fnr, fpr, score[5]]
    results.to_csv("results/results" + timestr + ".csv", sep="\t", encoding='utf-8')
    fold += 1
