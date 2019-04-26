from obspy import read


st = read("165.dat")

print(st.__str__(extended=True))
print(st[0].stats.seg2)

for i in range(len(st)):
    st[i].stats.distance = st[i].stats.seg2["RECEIVER_LOCATION"]

st.plot(type='section', orientation='horizontal')
