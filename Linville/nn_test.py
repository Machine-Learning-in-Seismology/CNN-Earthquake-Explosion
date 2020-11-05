import pickle, time, sys, os, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
sys.path.insert(1, '/home/dertuncay/Earthquake_Detection_ML/CNN-Earthquake-Explosion-main')
from model import build_nn, get_early_stop
from sklearn.utils import class_weight
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats
warnings.filterwarnings("ignore")

# Spectrogram
def spectrogram(tr):
	tr.trim(tr.stats.starttime, tr.stats.starttime + 90,fill_value=0)
	Sxx, freqs, t, im = plt.specgram(tr.data[:-1],Fs = 100, NFFT = 256, noverlap = 32,Fc = 1)#, noverlap = 256*0.12)#
	final = Sxx[:48,:]
	plt.close('all')
	return final, t, freqs[:48]

models = os.listdir('models/')

model_names = []; tmp =[];
for mdl in models:
	model_names.append(mdl[:-3])
	tmp.append(mdl[:-3])

for mdl,name in zip(models,model_names):
	globals()['model%s' % name] = load_model(os.path.join('models',mdl))

res_names = tmp.append('Real')

print("Loading data...")
with open("../../CNN-Earthquake-Explosion-main/earthquake.pkl", "rb") as f:
		eq = pickle.load(f, encoding='latin1')
with open("../../CNN-Earthquake-Explosion-main/explosions.pkl", "rb") as f:
		ex = pickle.load(f, encoding='latin1')

eq_labels = np.zeros(eq.shape[0])
ex_labels = np.ones(ex.shape[0])

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
	# Linville
	for name in model_names:
		globals()['pred%s' % name] = []

	for station, label in zip(X_test,y_test):
		tr1 = Trace()
		tr1.data = station[:,0]
		tr1.stats = Stats()
		tr2 = Trace()
		tr2.data = station[:,1]
		tr2.stats = Stats()
		tr3 = Trace()
		tr3.data = station[:,2]
		tr3.stats = Stats()

		tr1.stats.startime = UTCDateTime(2009, 1, 1, 12, 0, 0)
		tr1.stats.npts = 5000
		tr1.stats.sampling_rate = 50.
		tr1.stats.delta = 1/50
		tr1.stats.calib = 1
		tr1.stats.station = 'TEST'
		tr1.stats.channel = 'HHE'
		
		tr2.stats.startime = UTCDateTime(2009, 1, 1, 12, 0, 0)
		tr2.stats.npts = 5000
		tr2.stats.sampling_rate = 50.
		tr2.stats.delta = 1/50
		tr2.stats.calib = 1
		tr2.stats.station = 'TEST'
		tr2.stats.channel = 'HHN'

		tr3.stats.startime = UTCDateTime(2009, 1, 1, 12, 0, 0)
		tr3.stats.npts = 5000
		tr3.stats.sampling_rate = 50.
		tr3.stats.delta = 1/50
		tr3.stats.calib = 1
		tr3.stats.station = 'TEST'
		tr3.stats.channel = 'HHZ'

		sta  = Stream(traces=[tr1, tr2, tr3])
		# Data process
		sr = 100.0
		sta.resample(sr)
		sta.detrend()
		sta.taper(0.01, type='hann')
		sta.filter('highpass',freq=1)#Butterworth, 4 corners
		# Calculate Spectrogram
		specs = np.array([])
		for tr in sta:
			freq, _, _ = spectrogram(tr)
			specs = np.append(specs, freq)
			length = 40
			width  = 48
		input = specs.reshape((width*3, length))
		input = input.transpose()
		input = np.array([input,input,input])
		''' Outputs
		0 = Earthquake
		1 = Blast
		'''
		for name in model_names:
			res = globals()['model%s' % name].predict_classes(input)
			globals()['pred%s' % name].append(res[0])
	# Save to csv

	df = pd.DataFrame({'modelLSTM_40_1443': predLSTM_40_1443,
  'modelLSTM_40_1445': predLSTM_40_1445,
  'modelLSTM_40_1442': predLSTM_40_1442,
  'modelLSTM_40_1444': predLSTM_40_1444,
  'modelLSTM_40_1441': predLSTM_40_1441,
  'modelLSTM_40_1449': predLSTM_40_1449,
  'modelLSTM_40_1446': predLSTM_40_1446,
  'modelLSTM_40_1440': predLSTM_40_1440,
  'modelLSTM_40_1448': predLSTM_40_1448,
  'modelLSTM_40_1447': predLSTM_40_1447,
  'Real': y_test})
	df.to_csv(str(fold)+'.csv',index=False)
	fold += 1