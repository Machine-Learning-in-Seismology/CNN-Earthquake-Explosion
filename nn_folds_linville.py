import numpy as np
import pandas as pd
import os, glob, random, time, pickle
import matplotlib.pyplot as plt
#from model_linville import build_nn, get_early_stop
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#from keras.utils.vis_utils import plot_model
from datetime import datetime
from obspy import Trace
import warnings
warnings.filterwarnings("ignore")
DEBUG_THRESHOLD = False

models = os.listdir('Linville_Models/')

model_names = []; tmp =[];
for mdl in models:
	model_names.append(mdl[:-3])
	tmp.append(mdl[:-3])

# Emtpy DB to fill
#col_names = model_names
#col_names.append('fold')
#col_names.append('Real')
df = pd.DataFrame(columns=['LSTM_40_1446', 'LSTM_40_1449', 'LSTM_40_1447', 'LSTM_40_1448', 'LSTM_40_1440', 'LSTM_40_1442', 'LSTM_40_1441', 'LSTM_40_1443', 'LSTM_40_1444', 'LSTM_40_1445', 'fold', 'Real'])

print("Loading Models...")
for mdl,name in zip(models,model_names):
	globals()['model%s' % name] = load_model(os.path.join('Linville_Models',mdl))
print("Models Loaded...")

EPOCHS = [20]

print("Loading data...")
MASTER_DIR = "../DBs/"   # "/home/dertuncay/Miao-Eq-Exp-Test/DB/"

# FFT
EXP_DIR = MASTER_DIR + "Linville/EXP/"
ERT_DIR = MASTER_DIR + "Linville/EQ/"
eq_spec = []
ex_spec = []


#for i in range(len(os.listdir(EXP_DIR))):
for i in range(3):
	with open(os.path.join(EXP_DIR, f"explosions_{i:d}.pkl"), "rb") as f: # n_miao_
		ex_ = pickle.load(f, encoding='latin1')
		ex_spec = ex_spec + ex_
for i in range(len(os.listdir(ERT_DIR))):
	with open(os.path.join(ERT_DIR, f"earthquakes_{i:d}.pkl"), "rb") as f:
		eq_ = pickle.load(f, encoding='latin1')
		eq_spec = eq_spec + eq_

print("Data Loaded...")

eq_spec_labels = np.ones(len(eq_spec))
ex_spec_labels = np.zeros(len(ex_spec))
#print(len(eq_spec_labels))
#print(len(ex_spec_labels))
X_spec = np.concatenate((ex_spec, eq_spec))
#print(X_spec.shape)
y_spec = np.concatenate((ex_spec_labels, eq_spec_labels))
# X_spec = np.reshape(X_spec, (X_spec.shape[0],40,48,3))

# Build the model
#nn = build_nn(X_spec.shape[1:])
#y, X = shuffle(y, X)

# Shuffle
np.random.seed(42)
shuffled_indices = np.random.permutation(len(y_spec))
X_spec = X_spec[shuffled_indices]
y_spec = y_spec[shuffled_indices]

timestr = time.strftime("%Y%m%d-%H%M%S")

# Kfold
kf = KFold(n_splits=4)

fold = 0
for train_index, test_index in kf.split(X_spec):
	# FFT
	X_spec_train, X_spec_test = X_spec[train_index], X_spec[test_index]
	y_spec_train, y_spec_test = y_spec[train_index], y_spec[test_index]

	for name in model_names:
		globals()['pred%s' % name] = []
	for station, label in zip(X_spec_test,y_spec_test):
		for name in model_names:
			res = globals()['model%s' % name].predict_classes(station)
			globals()['pred%s' % name] = res[0]
#			globals()['pred%s' % name].append(res[0])
#               # Save to csv
		df = df.append({'fold':fold,
		'LSTM_40_1443': predLSTM_40_1443,
		'LSTM_40_1445': predLSTM_40_1445,
		'LSTM_40_1442': predLSTM_40_1442,
		'LSTM_40_1444': predLSTM_40_1444,
		'LSTM_40_1441': predLSTM_40_1441,
		'LSTM_40_1449': predLSTM_40_1449,
		'LSTM_40_1446': predLSTM_40_1446,
		'LSTM_40_1440': predLSTM_40_1440,
		'LSTM_40_1448': predLSTM_40_1448,
		'LSTM_40_1447': predLSTM_40_1447,
		'Real': label}, ignore_index=True)
#		print(df)
#		# Save to csv
#		df = pd.DataFrame({'fold':fold,'modelLSTM_40_1443': predLSTM_40_1443,
#		'modelLSTM_40_1445': predLSTM_40_1445,
#		'modelLSTM_40_1442': predLSTM_40_1442,
#		'modelLSTM_40_1444': predLSTM_40_1444,
#		'modelLSTM_40_1441': predLSTM_40_1441,
#		'modelLSTM_40_1449': predLSTM_40_1449,
#		'modelLSTM_40_1446': predLSTM_40_1446,
#		'modelLSTM_40_1440': predLSTM_40_1440,
#		'modelLSTM_40_1448': predLSTM_40_1448,
#		'modelLSTM_40_1447': predLSTM_40_1447,
#		'Real': label})
#		df.to_csv("results/linville_fold_results_fold" + str(fold) + '_' + timestr + ".csv", sep="\t", encoding='utf-8')
	fold += 1

df.to_csv("results/linville_fold_results_" + timestr + ".csv", sep="\t", encoding='utf-8')
