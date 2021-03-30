import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from model import build_nn, get_early_stop
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint

DEBUG_THRESHOLD = False


def plot_signal(s, n):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(s[0], color="tab:orange")
    ax2.plot(s[1], color="tab:blue")
    ax3.plot(s[2], color="tab:green")
    fig.tight_layout()
    plt.savefig(f"example_{n:d}.png")
    plt.close()


print("Loading data...")
EXP_DIR = "../Normalized/KOERI_Explosions/"
# EXP_DIR = "Miao_Explosions/"
ERT_DIR = "../Normalized/Earthquakes"
eq = []
ex = []
for i in range(2):
    with open(os.path.join(ERT_DIR, f"n_earthquakes_{i:d}.pkl"), "rb") as f:
        eq_ = pickle.load(f, encoding='latin1')
        eq = eq + eq_
    with open(os.path.join(EXP_DIR, f"n_koeri_explosions{i:d}.pkl"), "rb") as f: # n_miao_
        ex_ = pickle.load(f, encoding='latin1')
        ex = ex + ex_

y_ex = [0] * len(ex)
y_eq = [1] * len(eq)
y = np.array(y_ex + y_eq)
x = ex + eq
X = np.zeros([len(x), 3, 9002])
for i, z in enumerate(x):
    X[i, :, :z.shape[1]] = z
del y_ex, y_eq, ex, eq

if DEBUG_THRESHOLD:
    plot_signal(X[0], 0)
    plot_signal(X[-1], 9000)

    th = np.mean(np.max(X[:9000, 0, :], axis=1))
    y_ = np.zeros([X.shape[0]])
    y_m = np.max(X[:, 0, :], axis=1) > th
    y_[y_m] = 1
    print(np.sum(np.abs(y - y_)) / len(X))

X = np.reshape(X, [X.shape[0], X.shape[2], 3])
#print(X.shape)
nn = build_nn(X.shape[1:])
y, X = shuffle(y, X)

#divider = int(X.shape[0]*0.25)
#y_test, y_train = y[:divider], y[divider:]
#X_test, X_train = X[:divider,:,:], X[divider:,:,:]

#mc = ModelCheckpoint(filepath='miao_model.h5', monitor='val_loss', save_best_only=True)
mc = ModelCheckpoint(filepath='koeri_model.h5', monitor='val_loss', save_best_only=True)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}
nn.fit(x=X, y=y,
       batch_size=100, epochs=200,
       validation_split=0.25, class_weight=class_weights,
       callbacks=[get_early_stop(),mc])
#score = nn.evaluate(X_test, y_test)
#print(score)
#nn.save('my_model.h5')
