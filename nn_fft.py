import pickle
import numpy as np
import pandas as pd
import os, glob, random, time, sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from datetime import datetime
from obspy import Trace
import tensorflow as tf
DEBUG_THRESHOLD = False

model_no = sys.argv[1]
if model_no == '0':
	from model_fft import build_nn, get_early_stop
elif model_no == '1':
	from model_fft1 import build_nn, get_early_stop
elif model_no == '2':
	from model_fft2 import build_nn, get_early_stop
elif model_no == '3':
	from model_fft3 import build_nn, get_early_stop
elif model_no == '4':
	from model_fft4 import build_nn, get_early_stop
elif model_no == '5':
	from model_fft5 import build_nn, get_early_stop
elif model_no == '6':
	from model_fft6 import build_nn, get_early_stop
elif model_no == '7':
	from model_fft7 import build_nn, get_early_stop
elif model_no == '8':
	from model_fft8 import build_nn, get_early_stop
elif model_no == '9':
	from model_fft9 import build_nn, get_early_stop

def normal_wf(wf):
	tr = Trace(wf)
	tr.normalize()
	return tr.data[:9000]

def normal_fft(wf):
	tr = Trace(wf)
	tr.normalize()
	return tr.data[:50]

EPOCHS = [200]

print("Loading data...")
MASTER_DIR = "../DBs/"
# Waveform
EXP_DIR = MASTER_DIR + "WF/EXP/"
ERT_DIR = MASTER_DIR + "WF/EQ/"
eq_wf = []
ex_wf = []
for i in range(3):
	with open(os.path.join(EXP_DIR, f"explosions_{i:d}.pkl"), "rb") as f: # n_miao_
		ex_ = pickle.load(f, encoding='latin1')
		ex_wf = ex_wf + ex_
for i in range(len(os.listdir(ERT_DIR))):
	with open(os.path.join(ERT_DIR, f"earthquakes_{i:d}.pkl"), "rb") as f:
		eq_ = pickle.load(f, encoding='latin1')
		eq_wf = eq_wf + eq_


y_ex_wf = [0] * len(ex_wf)
y_eq_wf = [1] * len(eq_wf)
y_wf = np.array(y_ex_wf + y_eq_wf)
x_wf = ex_wf + eq_wf
X_wf = np.zeros([len(x_wf), 9000, 3])
for i, z in enumerate(x_wf):
	if len(z.shape) == 1:
		len_min = np.min([len(z[0]),len(z[1]),len(z[2])])
		z[0] = z[0][:len_min]
		z[1] = z[1][:len_min]
		z[2] = z[2][:len_min]
		z2 = np.array([z[0],z[1],z[2]])
		z = z2
	z1 = np.zeros((3,9000))
	z1[0,:] = normal_wf(z[0,:])
	z1[1,:] = normal_wf(z[1,:])
	z1[2,:] = normal_wf(z[2,:])
	X_wf[i, :, :z1.shape[1]] = z1.T

# FFT
EXP_DIR = MASTER_DIR + "FFT/EXP/"
ERT_DIR = MASTER_DIR + "FFT/EQ/"
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

y_ex_spec = [0] * len(ex_spec)
y_eq_spec = [1] * len(eq_spec)
y_spec = np.array(y_ex_spec + y_eq_spec)
x_spec = ex_spec + eq_spec
X_spec = np.zeros([len(x_spec), 50, 3])
for i, z in enumerate(x_spec):
	z1 = np.zeros((3,50))
	z1[0,:] = normal_fft(z[0,:])
	z1[1,:] = normal_fft(z[1,:])
	z1[2,:] = normal_fft(z[2,:])
	X_spec[i, :, :z1.shape[1]] = z1.T


# Build the model
nn = build_nn(X_wf.shape[1:],X_spec.shape[1:])

# Shuffle
np.random.seed(42)
shuffled_indices = np.random.permutation(len(y_spec))
X_wf = X_wf[shuffled_indices]
X_spec = X_spec[shuffled_indices]
y_wf = y_wf[shuffled_indices]
y_spec = y_spec[shuffled_indices]

# Result DB
results = pd.DataFrame(columns=['seed', 'fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")

# Kfold
kf = KFold(n_splits=4,shuffle=False)

fold = 0
for train_index, test_index in kf.split(X_wf):
	# WF
	X_wf_train, X_wf_test = X_wf[train_index], X_wf[test_index]
	y_wf_train, y_wf_test = y_wf[train_index], y_wf[test_index]
	# FFT
	X_spec_train, X_spec_test = X_spec[train_index], X_spec[test_index]
	y_spec_train, y_spec_test = y_spec[train_index], y_spec[test_index]

	for seed in range(5):
		for epoch in EPOCHS:

			# Give a name to model
			f_name = 'Models/model_' + timestr  + '.h5'
			log_name = "logs/log" + model_no + "_" + timestr + ".csv"
			history_logger=tf.keras.callbacks.CSVLogger(log_name, separator=",", append=True)
			# Fit the model
			mc = ModelCheckpoint(filepath=f_name, monitor='val_loss', save_best_only=True)
			test_score = nn.fit(x=[X_wf_train,X_spec_train], y=[y_wf_train,y_spec_train],
				   batch_size=100, epochs=epoch,
				   validation_split=0.25,
				   callbacks=[get_early_stop(),mc,history_logger])
			
			score = nn.evaluate([X_wf_test, X_spec_test], [y_wf_test, y_spec_test])
			print("FNR: %.2f FPR: %.2f ACC: %.2f" % (
									score[3] / float(score[3] + score[4]),
									score[1] / float(score[1] + score[2]),
									score[5])
									  )
			fnr = score[3] / float(score[3] + score[4])
			fpr = score[1] / float(score[1] + score[2])
			results.loc[len(results)] = [seed, fold, fnr, fpr, score[5]]
			results.to_csv("results/model" + model_no + "results" + timestr + ".csv", sep="\t", encoding='utf-8')
	fold += 1
