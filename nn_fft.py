import pickle
import numpy as np
import pandas as pd
import os, glob, random, time
import matplotlib.pyplot as plt
from model_fft import build_nn, get_early_stop
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from datetime import datetime
from obspy import Trace
DEBUG_THRESHOLD = False

def normal_wf(wf):
	tr = Trace(wf)
	tr.normalize()
	return tr.data[:9000]

def normal_fft(wf):
	tr = Trace(wf)
	tr.normalize()
	return tr.data[:500]

def plot_signal(s, n):
	fig, (ax1, ax2, ax3) = plt.subplots(3)
	ax1.plot(s[0], color="tab:orange")
	ax2.plot(s[1], color="tab:blue")
	ax3.plot(s[2], color="tab:green")
	fig.tight_layout()
	plt.show()
	#plt.savefig(f"example_{n:d}.png")
	#plt.close()

EPOCHS = [20]

print("Loading data...")
MASTER_DIR = "/media/dertuncay/Elements2/Miao/FVG/"   # "/home/dertuncay/Miao-Eq-Exp-Test/DB/"
# Waveform
EXP_DIR = MASTER_DIR + "WF/EXP/"
ERT_DIR = MASTER_DIR + "WF/EQ/"
eq_wf = []
ex_wf = []
#for i in range(len(os.listdir(EXP_DIR))):#
for i in range(1):
	with open(os.path.join(EXP_DIR, f"explosions_{i:d}.pkl"), "rb") as f: # n_miao_
		ex_ = pickle.load(f, encoding='latin1')
		ex_wf = ex_wf + ex_
for i in range(len(os.listdir(ERT_DIR))):
	with open(os.path.join(ERT_DIR, f"earthquakes_{i:d}.pkl"), "rb") as f:
		eq_ = pickle.load(f, encoding='latin1')
		eq_wf = eq_wf + eq_

print(len(ex_wf),len(eq_wf))

y_ex_wf = [0] * len(ex_wf)
y_eq_wf = [1] * len(eq_wf)
y_wf = np.array(y_ex_wf + y_eq_wf)
x_wf = ex_wf + eq_wf
del y_ex_wf, y_eq_wf, ex_wf, eq_wf
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



# print(X_wf.shape)
#X_wf = np.reshape(X_wf, [X_wf.shape[0], X_wf.shape[2], 3])
#X_wf = np.reshape(X_wf, (X_wf.shape[0], X_wf.shape[2], X_wf.shape[1]))
# print(X_wf.shape)

# plt.plot(X_wf[10,:,2],color='r')
# plt.plot(X_wf[-1,:,2],color='b')
# plt.show()


# FFT
EXP_DIR = MASTER_DIR + "FFT/EXP/"
ERT_DIR = MASTER_DIR + "FFT/EQ/"
eq_spec = []
ex_spec = []
#for i in range(5):
#    with open(os.path.join(ERT_DIR, f"earthquakes_{i:d}.pkl"), "rb") as f:
#        eq_ = pickle.load(f, encoding='latin1')
#        eq_spec = eq_spec + eq_
#    with open(os.path.join(EXP_DIR, f"explosions{i:d}.pkl"), "rb") as f: # n_miao_
#        ex_ = pickle.load(f, encoding='latin1')
#        ex_spec = ex_spec + ex_

#for i in range(len(os.listdir(EXP_DIR))):
for i in range(1):
#    with open(os.path.join(ERT_DIR, f"earthquakes_{i:d}.pkl"), "rb") as f:
#        eq_ = pickle.load(f, encoding='latin1')
#        eq_wf = eq_wf + eq_
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
del y_ex_spec, y_eq_spec, ex_spec, eq_spec
X_spec = np.zeros([len(x_spec), 500, 3])
for i, z in enumerate(x_spec):
	z1 = np.zeros((3,500))
	z1[0,:] = normal_fft(z[0,:])
	z1[1,:] = normal_fft(z[1,:])
	z1[2,:] = normal_fft(z[2,:])
   
	# plot_signal(z,3)
	X_spec[i, :, :z1.shape[1]] = z1.T

# plt.plot(X_spec[10,1,:],color='r')
# plt.plot(X_spec[-1,1,:],color='b')
# plt.show()
# X_spec = np.reshape(X_spec, [X_spec.shape[0], X_spec.shape[2], X_spec.shape[1]])
# plt.plot(X_spec[10,:,1],color='r')
# plt.plot(X_spec[-1,:,1],color='b')
# plt.show()
# print(X_spec.shape)

# Build the model
nn = build_nn(X_wf.shape[1:],X_spec.shape[1:])
#y, X = shuffle(y, X)

# Shuffle
shuffled_indices = np.random.permutation(len(y_spec))
X_wf = X_wf[shuffled_indices]
X_spec = X_spec[shuffled_indices]
y_wf = y_wf[shuffled_indices]
y_spec = y_spec[shuffled_indices]

# # Train Test Split
# X_wf_train, X_wf_test, y_wf_train, y_wf_test = train_test_split(X_wf, y_wf, test_size=0.25, random_state=42)
# X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, test_size=0.25, random_state=42)

# Result DB
results = pd.DataFrame(columns=['seed', 'fold', 'fnr', 'fpr', 'acc'])
timestr = time.strftime("%Y%m%d-%H%M%S")


# Kfold
kf = KFold(n_splits=5)

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
			# date = str(datetime.date(datetime.now()))
			# l_dates = str(len(glob.glob('*' + date + '*')))
			# f_name = 'Models/model_' + date + '_' + l_dates  + '.h5'
			f_name = 'Models/model_' + timestr  + '.h5'

			# Fit the model
			mc = ModelCheckpoint(filepath=f_name, monitor='val_loss', save_best_only=True)
			#class_weights = class_weight.compute_class_weight('balanced', np.unique(y_wf), y_wf)
			#class_weights = {i: class_weights[i] for i in range(len(class_weights))}
			nn.fit(x=[X_wf_train,X_spec_train], y=[y_wf_train,y_spec_train],
				   batch_size=100, epochs=epoch,
				   validation_split=0.25,
				   callbacks=[get_early_stop(),mc])
			score = nn.evaluate([X_wf_test, X_spec_test], [y_wf_test, y_spec_test])
			print("FNR: %.2f FPR: %.2f ACC: %.2f" % (
									score[3] / float(score[3] + score[4]),
									score[1] / float(score[1] + score[2]),
									score[5])
									  )
			fnr = score[3] / float(score[3] + score[4])
			fpr = score[1] / float(score[1] + score[2])
			results.loc[len(results)] = [seed, fold, fnr, fpr, score[5]]
			results.to_csv("results/results" + timestr + ".csv", sep="\t", encoding='utf-8')
	fold += 1