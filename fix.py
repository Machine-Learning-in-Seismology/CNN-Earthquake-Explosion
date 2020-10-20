from obspy import read, Stream

files = {"/mnt/big-data/earthquake/EQ_Exp/Explosions_Final/2019_12_10_12_59_21.mseed": "VNDS",
         "/mnt/big-data/earthquake/EQ_Exp/Explosions_Final/2013_12_18_12_59_35.mseed": "MYKA",
         "/mnt/big-data/earthquake/EQ_Exp/Explosions_Final/2018_12_21_13_0_55.mseed": "CEY"}

tobedeleted = {"VNDS": "BHE", "MYKA": "BHE", "CEY": "HHE"}

for file, station in files.items():
    st = read(file, format='MSEED')
    for tr in st:
        if tr.stats.station == station and tr.stats.channel == tobedeleted[tr.stats.station]:
            st.remove(tr)
    st.write(file, format='MSEED')


