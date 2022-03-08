import pickle
import numpy as np
import pandas as pd
import os, glob, random, time
import matplotlib.pyplot as plt
from model_linville import build_nn, get_early_stop
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from datetime import datetime
from obspy import Trace
DEBUG_THRESHOLD = False


EPOCHS = [20]

print("Loading data...")
MASTER_DIR = "../DBs/"

# FFT
EXP_DIR = MASTER_DIR + "Linville/EXP/"
ERT_DIR = MASTER_DIR + "Linville/EQ/"
eq_spec = []
ex_spec = []

for i in range(3):
	with open(os.path.join(EXP_DIR, f"explosions_{i:d}.pkl"), "rb") as f: # n_miao_
		ex_ = pickle.load(f, encoding='latin1')
		ex_spec = ex_spec + ex_
for i in range(len(os.listdir(ERT_DIR))):
	with open(os.path.join(ERT_DIR, f"earthquakes_{i:d}.pkl"), "rb") as f:
		eq_ = pickle.load(f, encoding='latin1')
		eq_spec = eq_spec + eq_



eq_spec_labels = np.ones(len(eq_spec))
ex_spec_labels = np.zeros(len(ex_spec))
X_spec = np.concatenate((ex_spec, eq_spec))
X_spec = X_spec[:,0,:,:]
print(X_spec.shape)
y_spec = np.concatenate((ex_spec_labels, eq_spec_labels))


# Build the model
nn = build_nn(X_spec.shape[1:])

# Shuffle
np.random.seed(42)
shuffled_indices = np.random.permutation(len(y_spec))
X_spec = X_spec[shuffled_indices]
y_spec = y_spec[shuffled_indices]

# Result DB
results = pd.DataFrame(columns=['seed', 'fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")


# Kfold
kf = KFold(n_splits=4)

fold = 0
for train_index, test_index in kf.split(X_spec):
	# FFT
	X_spec_train, X_spec_test = X_spec[train_index], X_spec[test_index]
	y_spec_train, y_spec_test = y_spec[train_index], y_spec[test_index]

	for seed in range(5):
		for epoch in EPOCHS:

			# Give a name to model
			f_name = 'Models/linville_model_' + timestr  + '.h5'

			# Fit the model
			mc = ModelCheckpoint(filepath=f_name, monitor='val_loss', save_best_only=True)
			nn.fit(x=[X_spec_train], y=[y_spec_train],
				   batch_size=100, epochs=epoch,
				   validation_split=0.25,
				   callbacks=[get_early_stop(),mc])
			score = nn.evaluate([X_spec_test], [y_spec_test])
			print("FNR: %.2f FPR: %.2f ACC: %.2f" % (
									score[3] / float(score[3] + score[4]),
									score[1] / float(score[1] + score[2]),
									score[5])
									  )
			fnr = score[3] / float(score[3] + score[4])
			fpr = score[1] / float(score[1] + score[2])
			results.loc[len(results)] = [seed, fold, fnr, fpr, score[5]]
			results.to_csv("results/linville_results" + timestr + ".csv", sep="\t", encoding='utf-8')
	fold += 1
