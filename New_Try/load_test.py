import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from model import build_nn, get_early_stop
from sklearn.utils import shuffle
from sklearn.utils import class_weight

model = build_nn((9002,3))
models = ['koeri','miao']

model.load_weights('koeri_model.h5')
# model.load_weights('miao_model.h5')
import os, glob

# Results
cols = ['Label', 'Prediction']
res = pd.DataFrame(columns=cols)

# Load Data
EXP_DIR = "../Normalized/FVG_Explosions/"
ERT_DIR = "../Normalized/Central_Italy_Earthquakes"
eq = []
ex = []
for i in range(3):
    with open(os.path.join(ERT_DIR, f"n_citaly_earthquakes{i:d}.pkl"), "rb") as f:
        eq_ = pickle.load(f, encoding='latin1')
        eq = eq + eq_

for i in range(2):
    with open(os.path.join(EXP_DIR, f"n_fvg_explosions{i:d}.pkl"), "rb") as f:
        ex_ = pickle.load(f, encoding='latin1')
        ex = ex + ex_

# Explosions
for data in ex:
    try:
        data = np.reshape(data, [1, data.shape[-1], 3])
        # print(data.shape)
        # Read Station
        # Explosion: 0 | Earthquake : 1
        prediction = model.predict(data)
        # print('Label: {} | Prediction {}'.format(0,prediction[0][0]))
        res = res.append({'Label': 0, 'Prediction': prediction[0][0]}, ignore_index=True)
    except:
        res = res.append({'Label': 0, 'Prediction': 'Error'}, ignore_index=True)

# Explosions
for data in eq:
    try:
        data = np.reshape(data, [1, data.shape[-1], 3])
        # print(data.shape)
        # Read Station
        # Explosion: 0 | Earthquake : 1
        prediction = model.predict(data)
        # print('Label: {} | Prediction {}'.format(0,prediction[0][0]))
        res = res.append({'Label': 1, 'Prediction': prediction[0][0]}, ignore_index=True)
    except:
        res = res.append({'Label': 1, 'Prediction': 'Error'}, ignore_index=True)

res.to_csv('koeri_model_pred_res.csv',index=False)
