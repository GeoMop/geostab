import obspy
from obspy.signal.trigger import recursive_sta_lta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class TraceItem:
    def __init__(self, data, cft, prev=None, next=None):
        self.data = data
        """Trace data"""
        self.cft = cft
        """Characteristic function"""
        self.cft_max = np.amax(cft)
        self.cft_max_i = np.argmax(cft)
        self.prev = prev
        """Previous trace"""
        self.next = next
        """Next trace"""


class SeismicMeasurement:
    def __init__(self):
        self.sampling_rate = 0.0
        self.trace_dict = {}
        """Dict of individual traces"""


def load_measurement(files, nsta=40, nlta=60):
    """
    Loads measurements from files.
    :param files: List of files
    :param nsta: Length of short time average window in samples
    :param nlta: Length of long time average window in samples
    :return: SeismicMeasurement
    """
    sm = SeismicMeasurement()

    first = True
    for file in files:
        st = obspy.read(file)

        if first:
            sm.sampling_rate = st[0].stats.sampling_rate
            first = False

        prev = None
        for tr in st:
            sl = float(tr.stats.seg2["SOURCE_LOCATION"])
            rl = float(tr.stats.seg2["RECEIVER_LOCATION"])
            key = (sl, rl)
            cft = recursive_sta_lta(tr.data, nsta, nlta)
            sm.trace_dict[key] = TraceItem(tr.data, cft, prev, None)
            if prev is not None:
                sm.trace_dict[prev].next = key
            prev = key

    return sm


def create_map(required_first_arrival):
    """
    Creates maps between (source_location, receiver_location) and index in optimize vector.
    :param required_first_arrival: List of (source_location, receiver_location)
    :return: {(source_location, receiver_location) -> xi}, {xi -> [(source_location, receiver_location), ]}
    """
    trace_to_xi = {}
    xi_to_trace = {}

    next_ind = 0
    #for rfa in required_first_arrival:
    for rfa in sorted(required_first_arrival):
        if rfa[1] != rfa[0]:
            if (rfa[1], rfa[0]) in trace_to_xi:
                xi = trace_to_xi[(rfa[1], rfa[0])]
                trace_to_xi[rfa] = xi
                xi_to_trace[xi].append(rfa)
            else:
                trace_to_xi[rfa] = next_ind
                xi_to_trace[next_ind] = [rfa]
                next_ind += 1

    return trace_to_xi, xi_to_trace


def crit_fun(x, seismic_measurement, trace_to_xi, xi_to_trace, diff_weight=20, sequence_weight=20):
    """
    Criterial function.
    :param x: Parameter vector
    :param seismic_measurement:
    :param trace_to_xi:
    :param xi_to_trace:
    :param diff_weight: weight of penalization difference of difference first arrivals
    :param sequence_weight: weight of penalization if first arrivals not in sequence
    :return: criterium
    """
    crit = 0.0
    for i in range(len(x)):
        for trace_key in xi_to_trace[i]:
            # characteristic function
            ti = seismic_measurement.trace_dict[trace_key]
            crit += (ti.cft_max - ti.cft[int(x[i] * seismic_measurement.sampling_rate)]) ** 2

            # find previous and next x
            sl, rl = trace_key
            ti = seismic_measurement.trace_dict[trace_key]
            prev_x = None
            if ti.prev is not None:
                if ti.prev in trace_to_xi:
                    prev_x = x[trace_to_xi[ti.prev]]
                elif ti.prev[0] == ti.prev[1]:
                    prev_x = 0.0
            next_x = None
            if ti.next is not None:
                if ti.next in trace_to_xi:
                    next_x = x[trace_to_xi[ti.next]]
                elif ti.next[0] == ti.next[1]:
                    next_x = 0.0

            # non sequence penalization
            if rl < sl:
                if next_x is not None:
                    if x[i] < next_x:
                        crit += (next_x - x[i]) * sequence_weight
            elif rl < sl:
                if prev_x is not None:
                    if x[i] < prev_x:
                        crit += (prev_x - x[i]) * sequence_weight

            # difference penalization
            if (prev_x is not None) and (next_x is not None):
                crit += (x[i] * 2 - prev_x - next_x) ** 2 * diff_weight

    return crit


def create_bounds(seismic_measurement, xi_to_trace, min_off, min_vel, max_off, max_vel):
    """
    Create bounds intervals for x.
    :param xi_to_trace:
    :param min_off: time minimal offset
    :param min_vel: minimal velocity
    :param max_off: time maximal offset
    :param max_vel: maximal velocity
    :return: bounds
    """
    bounds = []
    for i in range(len(xi_to_trace)):
        trace_key = xi_to_trace[i][0]
        sl, rl = trace_key
        if rl > sl:
            min = min_off + (rl - sl) / min_vel
            max = max_off + (rl - sl) / max_vel
        else:
            min = min_off + (sl - rl) / min_vel
            max = max_off + (sl - rl) / max_vel
        max_time = (len(seismic_measurement.trace_dict[trace_key].data) - 1) / seismic_measurement.sampling_rate
        if max > max_time:
            max = max_time
        bounds.append((min, max))
    return bounds


def create_initial_population(seismic_measurement, xi_to_trace, bounds, popsize=15):
    """
    Creates initial population random with one specimen in maximum of characteristic function.
    :param seismic_measurement:
    :param xi_to_trace:
    :param bounds:
    :param popsize: The population has popsize * len(x) individuals.
    :return:
    """
    m = popsize * len(bounds)
    init = np.zeros((m, len(bounds)))
    for i in range(m - 1):
        for j in range(len(bounds)):
            init[i, j] = (bounds[j][1] - bounds[j][0]) * np.random.random_sample() + bounds[j][0]
    for j in range(len(bounds)):
        sr = seismic_measurement.sampling_rate
        cft = seismic_measurement.trace_dict[xi_to_trace[j][0]].cft
        init[m - 1, j] = np.argmax(cft[int(bounds[j][0] * sr):int(bounds[j][1] * sr)]) / sr + bounds[j][0]
    return init


def create_initial_population2(seismic_measurement, xi_to_trace, bounds, popsize=15):
    """
    Creates initial population random around maximum of characteristic function.
    :param seismic_measurement:
    :param xi_to_trace:
    :param bounds:
    :param popsize: The population has popsize * len(x) individuals.
    :return:
    """
    m = popsize * len(bounds)
    init = np.zeros((m, len(bounds)))
    for i in range(m - 1):
        for j in range(len(bounds)):
            sr = seismic_measurement.sampling_rate
            cft = seismic_measurement.trace_dict[xi_to_trace[j][0]].cft
            init[i, j] = np.argmax(cft[int(bounds[j][0] * sr):int(bounds[j][1] * sr)]) / sr + bounds[j][0] \
                         + 0.001 * np.random.random_sample() + 0.005
    return init


def first_arrival_from_x(required_first_arrival, trace_to_xi, x):
    """
    Converts first arrivals from x to {(source_location, receiver_location) -> first_arrival}.
    :param required_first_arrival:
    :param trace_to_xi:
    :param x:
    :return: {(source_location, receiver_location) -> first_arrival}
    """
    first_arrival = {}
    for rfa in required_first_arrival:
        if rfa in trace_to_xi:
            first_arrival[rfa] = x[trace_to_xi[rfa]]
        else:
            first_arrival[rfa] = 0.0
    return first_arrival


def plt_bounds_from_x_bounds(required_first_arrival, trace_to_xi, x_bounds):
    """
    Converts bounds of x to bounds of first arrivals.
    :param required_first_arrival:
    :param trace_to_xi:
    :param x_bounds: bounds of x
    :return:
    """
    plt_bounds = {}
    for rfa in required_first_arrival:
        if rfa in trace_to_xi:
            plt_bounds[rfa] = x_bounds[trace_to_xi[rfa]]
    return plt_bounds


def plot_results(seismic_measurement, first_arrival, first_arrival2={}, plt_bounds={}, all_traces=True):
    """
    Plot first arrivals.
    :param seismic_measurement:
    :param first_arrival: {(source_location, receiver_location) -> first_arrival}
    :param first_arrival2: {(source_location, receiver_location) -> first_arrival}
    :param plt_bounds: {(source_location, receiver_location) -> (min, max)}
    :param all_traces: if True show all traces
    :return:
    """
    sl_list = sorted({fa[0] for fa in first_arrival})

    # plot shape
    plt_num = len(sl_list)
    if plt_num <= 1:
        cols = 1
    elif plt_num <= 4:
        cols = 2
    elif plt_num <= 9:
        cols = 3
    else:
        cols = 4
    rows = plt_num // cols
    if plt_num % cols > 0:
        rows += 1

    for i in range(plt_num):
        ls = sl_list[i]

        # plot traces by ObsPy
        if all_traces:
            fa_keys = seismic_measurement.trace_dict.keys()
        else:
            fa_keys = first_arrival.keys()
        st = obspy.Stream()
        for fa in fa_keys:
            if fa[0] == ls:
                st.append(obspy.Trace(seismic_measurement.trace_dict[fa].data,
                                      header={"distance": fa[1], "sampling_rate": seismic_measurement.sampling_rate}))
        ax = plt.subplot(rows, cols, i+1)
        st.plot(type='section', orientation='horizontal', show=False, fig=ax.figure, scale=4, norm_method="trace")

        sm = seismic_measurement
        sr = sm.sampling_rate
        for fa, x in first_arrival.items():
            if fa[0] == ls:
                # characteristic function
                # cft = sm.trace_dict[fa].cft
                # ax.plot(np.array(range(len(cft))) / sr, cft / 500 + fa[1] / 1000)

                # bounds
                if fa in plt_bounds:
                    ax.plot([plt_bounds[fa]], [fa[1] / 1000], 'yx')

                # first arrivals 2
                if fa in first_arrival2:
                    ax.plot([first_arrival2[fa]], [fa[1] / 1000], 'gx')

                # max of characteristic function
                if fa in plt_bounds:
                    #ax.plot([sm.trace_dict[fa].cft_max_i / sr ], [fa[1] / 1000], 'b+')
                    ax.plot([(np.argmax(sm.trace_dict[fa].cft[int(plt_bounds[fa][0] * sr):int(plt_bounds[fa][1] * sr)])
                              / sr + plt_bounds[fa][0])], [fa[1] / 1000], 'b+')

                # first arrivals
                ax.plot([x], [fa[1] / 1000], 'rx')

    plt.show()


def plot_cft(seismic_measurement, trace_key):
    """
    Plot characteristic function.
    :param seismic_measurement:
    :param trace_key:
    :return:
    """
    ti = seismic_measurement.trace_dict[trace_key]
    time = np.array(range(len(ti.data))) / seismic_measurement.sampling_rate
    plt.plot(time, ti.data / np.amax(np.absolute(ti.data)))
    plt.plot(time, ti.cft)
    plt.grid(True)
    plt.show()


def load_fa_xls(file_name):
    """
    Loads first arrivals from xls file.
    :param file_name:
    :return: {(source_location, receiver_location) -> first_arrival}
    """
    df = pd.read_excel(file_name, sheet_name=0, usecols="A:H", skiprows=2, header=None, dtype=object)

    def read_table(row_base, col_base):
        fa = {}
        for i in range(1, 8):
            sl = df[col_base + i][row_base]
            for j in range(2, 13):
                rl = df[col_base][row_base + j]
                fa[(sl, rl)] = df[col_base + i][row_base + j] / 1000
        return fa

    fa = read_table(0, 0)
    fa.update(read_table(15, 0))

    return fa
