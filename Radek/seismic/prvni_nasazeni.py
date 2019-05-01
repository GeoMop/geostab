import sys

from obspy import read
from obspy.signal.trigger import recursive_sta_lta
import matplotlib.pyplot as plt


st = read(sys.argv[1])

print(st.__str__(extended=True))
print(st[0].stats.seg2)

for i in range(len(st)):
    st[i].stats.distance = st[i].stats.seg2["RECEIVER_LOCATION"]

fig = plt.figure()

st.plot(type='section', orientation='horizontal', show=False, fig=fig, scale=4)

# hledani prvniho nasazeni
ax = fig.add_subplot(1, 1, 1)
for i in range(len(st)):
    tr = st[i]
    df = tr.stats.sampling_rate
    cft = recursive_sta_lta(tr.data, int(0.005 * df), int(0.010 * df))
    for j in range(len(cft)):
        if cft[j] > 1.4:
            ax.plot([j / df], [0.004 * i], 'rx')
            break

plt.show()
