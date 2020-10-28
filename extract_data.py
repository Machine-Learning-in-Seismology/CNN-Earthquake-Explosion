import pickle
from collections import defaultdict

from obspy import read, Stream
from obspy.core import UTCDateTime
import numpy as np
import pandas as pd
import glob, os, re
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy import signal
import numpy as np
import random

DIR = "/mnt/big-data/earthquake/EQ_Exp/"
explosions = ['Explosions_Final']  # ,'Explosion_Early_Final'] , 'Explosions']
ex = {}
for explosion in explosions:
    exp_events = glob.glob(os.path.join(DIR, explosion + '/*'))
    bar = ProgressBar(max_value=len(exp_events))
    for event in bar(exp_events):
        st = read(event, format='MSEED')
        st.filter('bandpass', freqmin=1, freqmax=20)
        st.resample(50.0)
        stations = defaultdict(lambda: dict())
        ex[event] = stations
        for tr in st:
            # Add zeros to beginning and ending of the signal with random length
            if tr.stats.npts * tr.stats.delta < 99:
                t_dif = 100 - tr.stats.npts * tr.stats.delta
                s = round(random.uniform(0, t_dif),2)
                e = round(t_dif - s,2)
                data = np.concatenate([np.zeros(int(s*tr.stats.sampling_rate)), tr.data, np.zeros(int(e*tr.stats.sampling_rate))], axis=0)
            else:
                data = tr.data
            stations[tr.stats.station][tr.stats.channel] = data

data_exp = np.empty((0, 3, 5000), float)
bar = ProgressBar(max_value=len(ex))
for name, stations in bar(ex.items()):
    for key in stations.keys():
        sign = list(stations[key].values())
        for i in range(0, len(sign), 3):
            # data_exp.append(np.array([sign[i], sign[i + 1], sign[i + 2]]))
            new = np.zeros((1, 3, 5000))
            new[0, 0, :min(sign[i].shape[0], 5000)] = sign[i][:min(sign[i].shape[0], 5000)]
            new[0, 1, :min(sign[i + 1].shape[0], 5000)] = sign[i + 1][:min(sign[i + 1].shape[0], 5000)]
            new[0, 2, :min(sign[i + 2].shape[0], 5000)] = sign[i + 2][:min(sign[i + 2].shape[0], 5000)]
            # new = np.array([np.array([sign[i][:2500], sign[i + 1][:2500], sign[i + 2][:2500]])])
            data_exp = np.append(data_exp, new, axis=0)

earthquakes = ['Earthquakes']
eq = {}
for earthquake in earthquakes:
    exp_events = glob.glob(os.path.join(DIR, earthquake + '/*'))
    bar = ProgressBar(max_value=len(exp_events))
    for event in bar(exp_events):
        event_ss = event.replace(earthquake, 'Earthquake_Start_Stop').replace('mseed', 'csv')
        df = pd.read_csv(event_ss)
        st = read(event, format='MSEED')
        st.filter('bandpass', freqmin=1, freqmax=20)
        st.resample(50.0)
        stations = defaultdict(lambda: dict())
        eq[event] = stations
        for tr in st:
            df_station = df[df['Station'] == tr.stats.station]
            if len(df) < 2:
                continue
            df_sta = df_station[df_station["Phase"] == "P"]
            p = UTCDateTime(year=df_sta['Year'].values[0],
                            month=df_sta['Month'].values[0],
                            day=df_sta['Day'].values[0],
                            hour=df_sta['Hour'].values[0],
                            minute=df_sta['Minute'].values[0],
                            second=int(str(df_sta['Sec'].values[0]).split('.')[0]),
                            microsecond=int(str(df_sta['Sec'].values[0]).split('.')[1]))
            tr.trim(p, p + 40, pad=True, fill_value=0)
            # Add zeros to beginning and ending of the signal with random length
            if tr.stats.npts * tr.stats.delta < 99:
                t_dif = 100 - tr.stats.npts * tr.stats.delta
                s = round(random.uniform(0, t_dif),2)
                e = round(t_dif - s,2)
                data = np.concatenate([np.zeros(int(s*tr.stats.sampling_rate)), tr.data, np.zeros(int(e*tr.stats.sampling_rate))], axis=0)
            else:
                data = tr.data
            stations[tr.stats.station][tr.stats.channel] = data

data_eq = np.empty((0, 3, 5000), float)
bar = ProgressBar(max_value=len(eq))
for name, stations in bar(eq.items()):
    for key in stations.keys():
        sign = list(stations[key].values())
        for i in range(0, len(sign), 3):
            new = np.zeros((1, 3, 5000))
            new[0, 0, :min(sign[i].shape[0], 5000)] = sign[i][:min(sign[i].shape[0], 5000)]
            new[0, 1, :min(sign[i + 1].shape[0], 5000)] = sign[i + 1][:min(sign[i + 1].shape[0], 5000)]
            new[0, 2, :min(sign[i + 2].shape[0], 5000)] = sign[i + 2][:min(sign[i + 2].shape[0], 5000)]
            # data_eq = np.append(data_eq, np.array([sign[i], sign[i + 1], sign[i + 2]]), axis=0)
            data_eq = np.append(data_eq, new, axis=0)


with open('explosions.pkl', 'wb') as f:
    pickle.dump(data_exp, f, pickle.HIGHEST_PROTOCOL)
with open('earthquake.pkl', 'wb') as f:
    pickle.dump(data_eq, f, pickle.HIGHEST_PROTOCOL)
