import obspy
from obspy.signal.trigger import recursive_sta_lta
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import pandas as pd

import math
import json


# first arrivals from measurement
#################################

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
            elif rl > sl:
                if prev_x is not None:
                    if x[i] < prev_x:
                        crit += (prev_x - x[i]) * sequence_weight

            # difference penalization
            if (prev_x is not None) and (next_x is not None):
                crit += (x[i] * 2 - prev_x - next_x) ** 2 * diff_weight

    return crit


def create_bounds(seismic_measurement, xi_to_trace, min_off, max_vel, max_off, min_vel):
    """
    Create bounds intervals for x.
    :param seismic_measurement:
    :param xi_to_trace:
    :param min_off: time minimal offset
    :param max_vel: maximal velocity
    :param max_off: time maximal offset
    :param min_vel: minimal velocity
    :return: bounds
    """
    bounds = []
    for i in range(len(xi_to_trace)):
        trace_key = xi_to_trace[i][0]
        sl, rl = trace_key
        if rl > sl:
            min = min_off + (rl - sl) / max_vel
            max = max_off + (rl - sl) / min_vel
        else:
            min = min_off + (sl - rl) / max_vel
            max = max_off + (sl - rl) / min_vel
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
                    ax.plot([(np.argmax(sm.trace_dict[fa].cft[int(plt_bounds[fa][0] * sr):int(plt_bounds[fa][1] * sr)])
                              / sr + plt_bounds[fa][0])], [fa[1] / 1000], 'b+')
                else:
                    ax.plot([sm.trace_dict[fa].cft_max_i / sr ], [fa[1] / 1000], 'b+')

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
            sl = float(df[col_base + i][row_base])
            for j in range(2, 13):
                rl = float(df[col_base][row_base + j])
                fa[(sl, rl)] = df[col_base + i][row_base + j] / 1000
        return fa

    fa = read_table(0, 0)
    fa.update(read_table(15, 0))

    return fa


def save_first_arrival(first_arrival, file_name):
    """
    Saves first arrivals to file.
    :param first_arrival: {(source_location, receiver_location) -> first_arrival}
    :param file_name:
    :return:
    """
    data = sorted([[k[0], k[1], v] for k, v in first_arrival.items()])
    with open(file_name, 'w') as fd:
        json.dump(data, fd, indent=4, sort_keys=True)


def load_first_arrival(file_name):
    """
    Loads first arrivals from file.
    :param file_name:
    :return: {(source_location, receiver_location) -> first_arrival}
    """
    with open(file_name, "r") as fd:
        data = json.load(fd)
    first_arrival = {(float(d[0]), float(d[1])): float(d[2]) for d in data}
    return first_arrival


# first arrivals from measurement - xdiff version
#################################################

def xdiff_create_map(seismic_measurement, required_first_arrival):
    """
    Creates map from (source_location, receiver_location) to indexes in optimize vector.
    :param seismic_measurement:
    :param required_first_arrival: List of (source_location, receiver_location)
    :return: {(source_location, receiver_location) -> [xi, ...]}
    """
    trace_to_xi = {}
    next_ind = 0

    def xi(trace):
        nonlocal next_ind

        if (trace is None) or (trace[0] == trace[1]):
            return []

        if trace in trace_to_xi:
            return trace_to_xi[trace]

        ti = seismic_measurement.trace_dict[trace]
        if trace[0] < trace[1]:
            base = ti.prev
        else:
            base = ti.next
        ret = xi(base) + [next_ind]
        trace_to_xi[trace] = ret
        next_ind += 1
        return ret

    for rfa in sorted(required_first_arrival):
        xi(rfa)

    return trace_to_xi


def xdiff_crit_fun(x, seismic_measurement, required_first_arrival, trace_to_xi, diff_weight=20, inv_weight=20):
    """
    Criterial function.
    :param x: Parameter vector
    :param seismic_measurement:
    :param required_first_arrival
    :param trace_to_xi:
    :param xi_to_trace:
    :param diff_weight: weight of penalization difference of difference first arrivals
    :param inv_weight: weight of penalization if inverse measurement is not same
    :return: criterium
    """
    crit = 0.0
    fa = xdiff_first_arrival_from_x(required_first_arrival, trace_to_xi, x)
    for trace_key in trace_to_xi:
        # characteristic function
        ti = seismic_measurement.trace_dict[trace_key]
        crit += (ti.cft_max - ti.cft[int(fa[trace_key] * seismic_measurement.sampling_rate)]) ** 2

        # find previous and next x
        ti = seismic_measurement.trace_dict[trace_key]
        prev_x = None
        if ti.prev is not None:
            if ti.prev in fa:
                prev_x = fa[ti.prev]
        next_x = None
        if ti.next is not None:
            if ti.next in fa:
                next_x = fa[ti.next]

        # difference penalization
        if (prev_x is not None) and (next_x is not None):
            crit += (fa[trace_key] * 2 - prev_x - next_x) ** 2 * diff_weight

        # inverse measurement penalization
        sl, rl = trace_key
        inv_trace_key = (rl, sl)
        if inv_trace_key in trace_to_xi:
            crit += (fa[trace_key] - fa[inv_trace_key]) ** 2 * inv_weight

    return crit


def xdiff_create_bounds(seismic_measurement, trace_to_xi, max_diff, long_max_vel, long_min_vel):
    """
    Create bounds intervals for x.
    :param seismic_measurement:
    :param trace_to_xi:
    :param max_diff: time maximal difference from previous arrival
    :param long_max_vel: for distant source location and first receiver
    :param long_min_vel: for distant source location and first receiver
    :return: bounds
    """
    max_i = 0
    for xi in trace_to_xi.values():
        m = max(xi)
        if m > max_i:
            max_i = m

    bounds = [(0.0, max_diff)] * (max_i + 1)

    # long sources
    for trace_key, xi in trace_to_xi.items():
        ti = seismic_measurement.trace_dict[trace_key]
        if len(xi) == 1:
            if (ti.prev is not None) and (ti.prev[0] == ti.prev[1]):
                continue
            if (ti.next is not None) and (ti.next[0] == ti.next[1]):
                continue
            dist = math.fabs(trace_key[1] - trace_key[0])
            bounds[xi[0]] = (dist / long_max_vel, dist / long_min_vel)
            # if trace_key == (-40.0, 0.0):
            #     bounds[xi[0]] = (bounds[xi[0]][0], 0.045)
            # print("{}: {}".format(trace_key, bounds[xi[0]]))

    return bounds


def xdiff_first_arrival_from_x(required_first_arrival, trace_to_xi, x):
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
            first_arrival[rfa] = math.fsum([x[i] for i in trace_to_xi[rfa]])
        else:
            first_arrival[rfa] = 0.0
    return first_arrival


# forward first arrival
#######################

def forward_fa(layers, receiver_locations):
    """
    Compute first arrivals from layers definition.
    :param layers: list of tuples (layer width, speed in layer)
    :param receiver_locations: list of receiver locations
    :return: (list of first arrivals, traces)
    """
    # critical angles
    critical_angles = []
    for i in range(len(layers) - 1):
        if layers[i + 1][1] > layers[i][1]:
            critical_angles.append(math.asin(layers[i][1] / layers[i + 1][1]))
        else:
            critical_angles.append(-1.0)

    # surface times
    surface_times = [rl / layers[0][1] for rl in receiver_locations]

    def dist(alpha0, num_lays):
        d = [math.tan(alpha0) * layers[0][0]]
        for i in range(1, num_lays):
            alpha = math.asin(math.sin(alpha0) / layers[0][1] * layers[i][1])
            d.append(math.tan(alpha) * layers[i][0])
        return d

    def trace_time(d):
        t = 0.0
        for i in range(len(d)):
            t += math.sqrt(math.pow(layers[i][0], 2) + math.pow(d[i], 2)) / layers[i][1]
        return t * 2

    def crit(alpha0, num_lays, h):
        return math.pow(math.fsum(dist(alpha0, num_lays)) - h, 2)

    traces = [[] for _ in receiver_locations]
    first_arrivals = surface_times.copy()

    # first layer
    alpha0_max = math.pi / 2
    if critical_angles[0] > 0.0:
        for i in range(len(receiver_locations)):
            h = receiver_locations[i] / 2
            alpha = math.atan(h / layers[0][0])
            if alpha > critical_angles[0]:
                tr = ([0.0, h, receiver_locations[i]], [0.0, -layers[0][0], 0.0])
                traces[i].append(tr)
        alpha0_max = critical_angles[0]

    # next layers
    for j in range(1, len(layers) - 1):
        if critical_angles[j] <= 0.0:
            continue

        r = math.sin(critical_angles[j]) / layers[j][1] * layers[0][1]
        if r >= 1.0:
            continue

        alpha0_min = math.asin(r)
        if alpha0_min >= alpha0_max:
            continue

        num_lays = j + 1
        for i in range(len(receiver_locations)):
            h = receiver_locations[i] / 2
            if math.fsum(dist(alpha0_min, num_lays)) >= h:
                continue

            res = minimize_scalar(crit, bounds=(alpha0_min, alpha0_max), args=(num_lays, h),
                                  method="bounded", options={"xatol": 1e-5})
            alpha0 = res.x

            # trace time
            d = dist(alpha0, num_lays)
            t = trace_time(d)
            if t < first_arrivals[i]:
                first_arrivals[i] = t

            # trace
            tr = ([0.0], [0.0])
            d_cum = 0.0
            y_cum = 0.0
            for k in range(num_lays):
                d_cum += d[k]
                y_cum -= layers[k][0]
                tr[0].append(d_cum)
                tr[1].append(y_cum)
            for k in reversed(range(num_lays)):
                d_cum += d[k]
                tr[0].append(d_cum)
                tr[1].append(tr[1][k])
            traces[i].append(tr)
        alpha0_max = alpha0_min

    return first_arrivals, traces


def plot_traces(layers, receiver_location, traces):
    """
    Plots traces.
    :param layers:
    :param receiver_location:
    :param traces:
    :return:
    """
    # borderlines
    max_d = max(receiver_location)
    plt.plot([0.0, max_d], [0.0, 0.0], "k")
    y = 0.0
    for lay in layers:
        y -= lay[0]
        plt.plot([0.0, max_d], [y, y], "k")

    # traces
    for receiver in traces:
        lines = []
        for tr in receiver:
            if tr[0]:
                if lines:
                    color = lines[0].get_color()
                else:
                    color = None
                lines = plt.plot(tr[0], tr[1], color=color)

    # source location
    plt.plot([0.0], [0.0], "gx")

    # receiver locations
    plt.plot(receiver_location, [0.0] * len(receiver_location), "rx")

    plt.grid(True)
    plt.show()
