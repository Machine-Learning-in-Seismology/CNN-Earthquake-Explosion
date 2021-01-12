import pickle

import obspy
import h5py
from obspy import UTCDateTime
import numpy as np
import pandas as pd
from obspy.clients.fdsn.client import Client
import matplotlib.pyplot as plt
import os

from progressbar import ProgressBar


def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type'] + 'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type'] + 'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type'] + 'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream


DIR = "/mnt/big-data/earthquake/EQ_Exp/STEAD"
file_name = 'merged.hdf5'
csv_file = 'merged.csv'

# reading the csv file into a dataframe:
df = pd.read_csv(os.path.join(DIR, csv_file))
print('total events in csv file: ' + str(len(df)))
# filterering the dataframe
df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 200)]
print('total events selected: ' + str(len(df)))

eqs = []
# retrieving selected waveforms from the hdf5 file:
dtfl = h5py.File(os.path.join(DIR, file_name), 'r')
bar = ProgressBar(max_value=len(df))
for index, row in bar(df.iterrows()):
    dataset = dtfl.get('data/' + str(row['trace_name']))
    st = make_stream(dataset)
    # waveforms, 3 channels: first row: E channle, second row: N channel, third row: Z channel
    st.filter('bandpass', freqmin=1, freqmax=20)
    st.trim(UTCDateTime(row['trace_start_time']) + row['p_travel_sec'] - 10,
            UTCDateTime(row['trace_start_time']) + row['p_travel_sec'] + 80, pad=True, fill_value=0)
    output = []
    if len(st) == 3:
        for tr in st:
            output.append(tr.data)
        output = np.array(output)
        eqs.append(output)
    else:
        print("Something went wrong: not 3 channels...")

with open(os.path.join(DIR, 'earthquakes.pkl'), 'wb') as f:
    pickle.dump(eqs, f, pickle.HIGHEST_PROTOCOL)
