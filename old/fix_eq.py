from obspy import read, Stream

files = {"/mnt/big-data/earthquake/EQ_Exp/Earthquakes/190327205918.mseed": ["FDMO", "TB01"],
         "/mnt/big-data/earthquake/EQ_Exp/Earthquakes/190327220350.mseed": ["FDMO", "TB01", "TB02"],
         "/mnt/big-data/earthquake/EQ_Exp/Earthquakes/190205080911.mseed": ["TB01"],
         "/mnt/big-data/earthquake/EQ_Exp/Earthquakes/190328092226.mseed": ["TB01"],
         "/mnt/big-data/earthquake/EQ_Exp/Earthquakes/190327210041.mseed": ["FDMO"]}

tobechanged = ["TB01", "TB02"]
tobeduplicated = ["FDMO"]

for file, station in files.items():
    st = read(file, format='MSEED')
    for s in station:
        st2 = st.select(station=s)
        if s == "FDMO":
            st3 = st2.select(channel="HHN").copy()
            st3[0].stats.channel = "HHE"
            st2 += st3
            st += st3
        l = len(st2)
        index = 0
        for tr in st2:
            if tr.stats.station in station and tr.stats.station in tobechanged and index == l-1:
                tr.stats.channel = "DH3"
                # print(tr.stats.station, tr.stats.channel)
            index += 1
        for tr in st2:
            st.remove(tr)
        st += st2
        st = st.sort()
        st.write(file, format='MSEED')
