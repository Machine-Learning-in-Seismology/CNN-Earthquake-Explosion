from collections import defaultdict

from obspy import read, Stream
from obspy.core import UTCDateTime
import numpy as np
import pandas as pd
import glob, os, re
import matplotlib.pyplot as plt

DIR = "/mnt/big-data/earthquake/EQ_Exp/"
explosions = ['Explosion_Early_Final', 'Explosions']
# for explosion in explosions:
#     exp_events = glob.glob(os.path.join(DIR, explosion + '/*'))
#     for event in exp_events:
#         st = read(event, format='MSEED')
#         st.filter('bandpass', freqmin=1, freqmax=20)
#         st.resample(25.0)
#         stations = defaultdict(lambda: dict())
#         for tr in st:
#             stations[tr.stats.station][tr.stats.channel] = tr.data
#             print(len(tr.data))
# max 3 channels per signal
# channels have specific name
# if vertical channel is missing, replace with 0s
# if horizontal channel is missing, replace with other hotizontal
# if only one channel, delete it

earthquakes = ['Earthquakes']
for earthquake in earthquakes:
    exp_events = glob.glob(os.path.join(DIR, earthquake + '/*'))
    for event in exp_events:
        event_ss = event.replace(earthquake, 'Earthquake_Start_Stop').replace('mseed', 'csv')
        df = pd.read_csv(event_ss)
        st = read(event, format='MSEED')
        st.filter('bandpass', freqmin=1, freqmax=20)
        st.resample(25.0)
        stations = defaultdict(lambda: dict())
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
            stations[tr.stats.station][tr.stats.channel] = tr.data
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(tr.times('matplotlib'), tr.data)
            ax.xaxis_date()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Count')
            plt.xticks(rotation=0)
            plt.show()
