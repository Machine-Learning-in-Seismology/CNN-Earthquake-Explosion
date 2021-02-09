import glob
import os
import pickle

from obspy import read, UTCDateTime, Trace
import numpy as np
from progressbar import ProgressBar

DIR = "/mnt/big-data/earthquake/EQ_Exp/KOERI_Bulletin/"
exps = glob.glob(os.path.join(DIR, 'KOERI_Explosion/*'))
explosions = []

bar = ProgressBar(max_value=len(exps))
for exp in bar(exps):
    st = read(exp, format='MSEED')
    st.resample(100)
    st.filter('bandpass', freqmin=1, freqmax=20)
    unique_stas = []
    for tr in st:
        if tr.stats.station not in unique_stas:
            unique_stas.append(tr.stats.station)

    for sta_name in unique_stas:
        sta = st.select(station=sta_name)
        # Merge data with gaps
        # Find Multi channel
        tmp_chan = []
        for tr in sta:
            tmp_chan.append(tr.stats.channel)
        multi_chan = []
        unique_chans = []
        for chan in tmp_chan:
            if tmp_chan.count(chan) > 1:
                multi_chan.append(chan)
        # Remove multichannel
        multi_chan = list(set(multi_chan))
        if len(multi_chan) != 0:
            for dub_chan in multi_chan:
                st_tmp = sta.select(channel=dub_chan)
                for tr in st_tmp:
                    sta.remove(tr)
                st_tmp.merge(method=1, fill_value=0)
                sta += st_tmp

        # Be sure that signls has the same length
        starts = []
        ends = []
        for tr in sta:
            starts.append(tr.stats.starttime)
            ends.append(tr.stats.endtime)
        start = max(starts)
        end = max(starts) + 90
        sta.trim(start, end, pad=True, fill_value=0)

        # Divide multi instrument stations into single instrument
        for chan in unique_chans:
            output = []
            sta_chan = sta.select(channel = chan + '*')
            if len(sta_chan) == 3:
                for tr in sta_chan:
                    output.append(tr.data)
                output = np.array(output)
                explosions.append(output)
            else:
                for tr in sta:
                    print(tr.id)
                print(f"Something went wrong: not 3 channels... {len(sta):d}")
                
explosions = np.array(explosions)
with open(os.path.join(DIR, 'explosions.pkl'), 'wb') as f:
    pickle.dump(explosions, f, pickle.HIGHEST_PROTOCOL)
