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


def create_x0(seismic_measurement, xi_to_trace, bounds):
    """
    Creates starting point in maximum of characteristic function.
    :param seismic_measurement:
    :param xi_to_trace:
    :param bounds:
    :return:
    """
    x0 = []
    for j in range(len(bounds)):
        sr = seismic_measurement.sampling_rate
        cft = seismic_measurement.trace_dict[xi_to_trace[j][0]].cft
        x0.append(np.argmax(cft[int(bounds[j][0] * sr):int(bounds[j][1] * sr)]) / sr + bounds[j][0])
    return x0


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


def plot_results(seismic_measurement, first_arrival, first_arrival2={}, plt_bounds={}, all_traces=True,
                 title="First arrivals"):
    """
    Plot first arrivals.
    :param seismic_measurement:
    :param first_arrival: {(source_location, receiver_location) -> first_arrival}
    :param first_arrival2: {(source_location, receiver_location) -> first_arrival}
    :param plt_bounds: {(source_location, receiver_location) -> (min, max)}
    :param all_traces: if True show all traces
    :param title:
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
    cols = 2
    rows = plt_num // cols
    if plt_num % cols > 0:
        rows += 1

    for i in range(plt_num):
        sl = sl_list[i]

        # plot traces by ObsPy
        if all_traces:
            fa_keys = seismic_measurement.trace_dict.keys()
        else:
            fa_keys = first_arrival.keys()
        st = obspy.Stream()
        for fa in fa_keys:
            if fa[0] == sl:
                st.append(obspy.Trace(seismic_measurement.trace_dict[fa].data,
                                      header={"distance": fa[1], "sampling_rate": seismic_measurement.sampling_rate}))
        ax = plt.subplot(rows, cols, i + 1)
        st.plot(type='section', orientation='horizontal', show=False, fig=ax.figure, scale=4, norm_method="trace")

        sm = seismic_measurement
        sr = sm.sampling_rate
        first_arrival_sl = {fa: x for fa, x in first_arrival.items() if fa[0] == sl}

        # characteristic function
        # for fa in first_arrival_sl:
        #     cft = sm.trace_dict[fa].cft
        #     ax.plot(np.array(range(len(cft))) / sr, cft / 500 + fa[1] / 1000)

        # bounds
        xl = [plt_bounds[fa][0] for fa in first_arrival_sl if fa in plt_bounds]
        xr = [plt_bounds[fa][1] for fa in first_arrival_sl if fa in plt_bounds]
        y = [fa[1] / 1000 for fa in first_arrival_sl if fa in plt_bounds]
        ax.plot(xl, y, 'y<', fillstyle="none", label="Min bounds")
        ax.plot(xr, y, 'y>', fillstyle="none", label="Max bounds")

        # first arrivals 2
        x = [first_arrival2[fa] for fa in first_arrival_sl if fa in first_arrival2]
        y = [fa[1] / 1000 for fa in first_arrival_sl if fa in first_arrival2]
        ax.plot(x, y, 'gx', label="First arrivals ref.")

        # max of characteristic function
        x = []
        for fa in first_arrival_sl:
            if fa in plt_bounds:
                x.append(np.argmax(sm.trace_dict[fa].cft[int(plt_bounds[fa][0] * sr):int(plt_bounds[fa][1] * sr)])
                         / sr + plt_bounds[fa][0])
            else:
                x.append(sm.trace_dict[fa].cft_max_i / sr)
        y = [fa[1] / 1000 for fa in first_arrival_sl]
        ax.plot(x, y, 'b+', label="Max of char. func.")

        # first arrivals
        x = [x for x in first_arrival_sl.values()]
        y = [fa[1] / 1000 for fa in first_arrival_sl]
        ax.plot(x, y, 'rx', label="First arrivals")

        # remove unwanted labels
        if i % 2 != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel(None)
        if i < 8:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel(None)

    #plt.suptitle(title)

    plt.legend(bbox_to_anchor=(1.05, 0.9), loc="upper left")
    plt.tight_layout(pad=0, w_pad=0.0, h_pad=-0.9)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.05, wspace=0.01)

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
    :param diff_weight: weight of penalization difference of difference first arrivals
    :param inv_weight: weight of penalization if inverse measurement is not same
    :return: criterium
    """
    crit = 0.0
    fa = xdiff_first_arrival_from_x(required_first_arrival, trace_to_xi, x)
    for trace_key in trace_to_xi:
        # characteristic function
        ti = seismic_measurement.trace_dict[trace_key]
        ind = int(fa[trace_key] * seismic_measurement.sampling_rate)
        if ind < len(ti.cft):
            c = ti.cft[ind]
        else:
            c = 0.0
        crit += (ti.cft_max - c) ** 2

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


# first arrivals from measurement - xdiff2 version
##################################################

def xdiff2_create_map(seismic_measurement, required_first_arrival):
    """
    Creates map from (source_location, receiver_location) to list of index in optimize vector and base (sl, rl).
    :param seismic_measurement:
    :param required_first_arrival: List of (source_location, receiver_location)
    :return: {(source_location, receiver_location) -> [(xi, (sl_base, rl_base)), ...]}
    """
    trace_to_xi = {}
    next_ind = 0

    def base_trace(trace):
        ti = seismic_measurement.trace_dict[trace]
        if trace[0] < trace[1]:
            base = ti.prev
        else:
            base = ti.next
        if (base is not None) and (base[1] == base[0]):
            return None
        return base

    for rfa in sorted(required_first_arrival):
        if rfa[1] != rfa[0]:
            rfa_inv = (rfa[1], rfa[0])
            if rfa_inv in trace_to_xi:
                trace_to_xi[rfa] = trace_to_xi[rfa_inv]
                continue
            xi = [(next_ind, base_trace(rfa))]
            next_ind += 1
            if rfa_inv in required_first_arrival:
                xi.append((next_ind, base_trace(rfa_inv)))
                next_ind += 1
            trace_to_xi[rfa] = xi

    return trace_to_xi


def xdiff2_crit_fun(x, seismic_measurement, required_first_arrival, trace_to_xi, diff_weight=20):
    """
    Criterial function.
    :param x: Parameter vector
    :param seismic_measurement:
    :param required_first_arrival
    :param trace_to_xi:
    :param diff_weight: weight of penalization difference of difference first arrivals
    :return: criterium
    """
    crit = 0.0
    fa = xdiff2_first_arrival_from_x(required_first_arrival, trace_to_xi, x)
    for trace_key in trace_to_xi:
        # characteristic function
        ti = seismic_measurement.trace_dict[trace_key]
        ind = int(fa[trace_key] * seismic_measurement.sampling_rate)
        if ind < len(ti.cft):
            c = ti.cft[ind]
        else:
            c = 0.0
        crit += (ti.cft_max - c) ** 2

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

    return crit


def xdiff2_create_bounds(seismic_measurement, trace_to_xi, max_diff, long_max_vel, long_min_vel):
    """
    Create bounds intervals for x.
    :param seismic_measurement:
    :param trace_to_xi:
    :param max_diff: time maximal difference from previous arrival
    :param long_max_vel: for distant source location and first receiver
    :param long_min_vel: for distant source location and first receiver
    :return: bounds
    """
    bounds = [(0.0, max_diff)] * len(trace_to_xi)

    # long sources
    for trace_key, xi in trace_to_xi.items():
        ti = seismic_measurement.trace_dict[trace_key]
        for xi_item in xi:
            if xi_item[1] is None:
                if (ti.prev is not None) and (ti.prev[0] == ti.prev[1]):
                    continue
                if (ti.next is not None) and (ti.next[0] == ti.next[1]):
                    continue
                dist = math.fabs(trace_key[1] - trace_key[0])
                bounds[xi_item[0]] = (dist / long_max_vel, dist / long_min_vel)

    return bounds


def xdiff2_first_arrival_from_x(required_first_arrival, trace_to_xi, x):
    """
    Converts first arrivals from x to {(source_location, receiver_location) -> first_arrival}.
    :param required_first_arrival:
    :param trace_to_xi:
    :param x:
    :return: {(source_location, receiver_location) -> first_arrival}
    """
    first_arrival = {}

    def fa(trace):
        if trace is None:
            return 0.0

        if trace in first_arrival:
            return first_arrival[trace]

        if trace not in trace_to_xi:
            return 0.0

        v_max = 0.0
        for xi_item in trace_to_xi[trace]:
            v = fa(xi_item[1]) + x[xi_item[0]]
            if v > v_max:
                v_max = v
        first_arrival[trace] = v_max
        return v_max

    for rfa in required_first_arrival:
        fa(rfa)

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
    fastest_trace_indexes = [-1] * len(traces)
    first_arrivals = surface_times.copy()

    # first layer
    alpha0_max = math.pi / 2
    if critical_angles[0] > 0.0:
        for i in range(len(receiver_locations)):
            h = receiver_locations[i] / 2
            alpha = math.atan(h / layers[0][0])
            if alpha > critical_angles[0]:
                tr = [[0.0, h, receiver_locations[i]], [0.0, -layers[0][0], 0.0], False]
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
                fastest_trace_indexes[i] = len(traces[i])

            # trace
            tr = [[0.0], [0.0], False]
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

    # mark fastest trace
    for receiver, ind in zip(traces, fastest_trace_indexes):
        if ind > 0:
            receiver[ind][2] = True

    return first_arrivals, traces


def plot_traces(layers, receiver_location, first_arrivals, traces):
    """
    Plots traces.
    :param layers:
    :param receiver_location:
    :param first_arrivals:
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

                if tr[2]:
                    lw = 1.5
                    ls = "-"
                else:
                    lw = 0.5
                    ls = "--"

                lines = plt.plot(tr[0], tr[1], color=color, lw=lw, ls=ls)

    # source location
    plt.plot([0.0], [0.0], "gx")

    # receiver locations
    plt.plot(receiver_location, [0.0] * len(receiver_location), "rx")

    # speed
    y = 0.0
    for lay in layers:
        plt.text(max_d + 1.0, y - lay[0] / 2, "v = {:.0f} m/s".format(lay[1]), va="center")
        y -= lay[0]

    # first arrivals
    for rl, fa in zip(receiver_location, first_arrivals):
        plt.text(rl, 1.0, "t = {:.3f} s".format(fa), rotation=45, rotation_mode="anchor")

    # labels
    plt.title("Traces")
    plt.xlabel("Geophone distance [m]")
    plt.ylabel("Depth [m]")

    # space for labels
    bottom, top = plt.ylim()
    new_top = top + (top - bottom) * 0.1
    plt.ylim(top=new_top)

    left, right = plt.xlim()
    new_fight = right + (right - left) * 0.1
    plt.xlim(right=new_fight)

    plt.grid(True)
    plt.show()


def plot_depth_speed(layers_list, title="Velocity profile"):
    """
    Plots depth speed.
    :param layers_list: list of layers
    :param title
    :return:
    """
    for layers in layers_list:
        xx = []
        yy = []
        x = 0.0
        for lay in layers:
            xx.append(x + lay[0] / 2)
            yy.append(lay[1])
            x += lay[0]
        plt.plot(xx, yy, "o-", label="{} layers".format(len(layers)))

    # labels
    #plt.title(title)
    plt.xlabel("Depth [m]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()

    plt.grid(True)
    plt.show()


# forward first arrival, layers are defined as polylines
########################################################

def forward_fa_poly(grid, layers_geo, layers_speed, required_first_arrival, return_traces=True, check_inputs=True, max_depth = 100, min_alpha0_span = 1e-5, xtol=1e-10, test_traces=False):
    """
    Compute first arrivals from layers definition. Layers are defined as polylines.
    :param grid: list of points, where layers geometry is defined, must be rising sequence
    :param layers_geo: list of layers geometry
    :param layers_speed: list of speeds in particular layers
    ([surface speed, layer 0 speed, ..., layer n - 1 speed, speed below last layer])
    :param required_first_arrival: list of (source_location, receiver_location)
    :param return_traces: if True returns traces for graphical representation
    :param check_inputs: if True checks inputs for errors
    :return: ({(source_location, receiver_location) -> first_arrival}, traces)
    """
    #sys.setrecursionlimit(1000)
    if check_inputs:
        # grid rising test
        for i in range(1, len(grid)):
            if grid[i] <= grid[i - 1]:
                raise ValueError("grid must be rising sequence")

        # check geometry
        for i in range(len(grid)):
            for j in range(1, len(layers_geo)):
                if layers_geo[j][i] > layers_geo[j - 1][i]:
                    raise ValueError("layer {} has negative width".format(j - 1))

        # check source/receiver inside grid
        for rfa in required_first_arrival:
            if rfa[0] < grid[0] or rfa[1] < grid[0] or rfa[0] > grid[-1] or rfa[1] > grid[-1]:
                raise ValueError("source/receiver {} is outside of grid".format(rfa))

    #first_arrival = {rfa: math.inf for rfa in required_first_arrival}

    # reshape required_first_arrival
    required_first_arrival_dict = {}
    processed = set()
    inverse_required_first_arrival = []
    for rfa in required_first_arrival:
        if (rfa[1], rfa[0]) in processed:
            inverse_required_first_arrival.append(rfa)
        else:
            if rfa[0] in required_first_arrival_dict:
                required_first_arrival_dict[rfa[0]].append(rfa[1])
            else:
                required_first_arrival_dict[rfa[0]] = [rfa[1]]

            processed.add(rfa)

    compute_traces = return_traces or test_traces

    # if flat_surface:
    #     min_surface = layers_geo[0][0]
    # else:
    #     min_surface = min(layers_geo[0])

    max_inside_speed = max(layers_speed[1:-1])

    hit_seg = []
    seg_num = len(grid) - 1
    for i in range(len(layers_geo)):
        hit_seg.append([0] * seg_num)
    hit_seg_map = {}

    class V2(tuple):
        """Simple 2d vector based on tuple. Much faster than use NumPy."""
        def __add__(self, other):
            return V2((self[0] + other[0], self[1] + other[1]))

        def __sub__(self, other):
            return V2((self[0] - other[0], self[1] - other[1]))

        def __mul__(self, other):
            return V2((self[0] * other, self[1] * other))

        def __truediv__(self, other):
            return V2((self[0] / other, self[1] / other))

        def __neg__(self):
            return V2((-self[0], -self[1]))

    # class np:
    #     @staticmethod
    #     def fromiter(a, b, c):
    #         return V2((a[0], a[1]))
    #         #return V2(a)

    def fun_cache(fun):
        cache = {}

        def wrapper(alpha0):
            if alpha0 in cache:
                return cache[alpha0]
            ret = fun(alpha0)
            cache[alpha0] = ret
            return ret

        return wrapper

    @fun_cache
    def find_source_pos(x):
        seg = 0
        while x >= grid[seg + 1]:
            seg += 1
            if seg >= len(grid) - 1:
                break
        if x == grid[seg]:
            if seg == 0:
                seg_list = [seg]
            elif seg >= len(grid) - 1:
                seg_list = [seg - 1]
            else:
                seg_list = [seg - 1, seg]
            y = layers_geo[0][seg]
        else:
            seg_list = [seg]
            y = layers_geo[0][seg] + (layers_geo[0][seg + 1] - layers_geo[0][seg]) * ((x - grid[seg]) / (grid[seg + 1] - grid[seg]))
        return V2((x, y)), seg_list

    traces = []
    traces_map = {}

    def norm(v):
        return math.hypot(v[0], v[1])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def normalize(v):
        return v / norm(v)

    def vec_ori(a, b):
        return a[0] * b[1] - b[0] * a[1]

    def line_intersect(a, b, c, d):
        """
        Returns point of lines intersection. Lines are defined by its points.
        If a[0] == b[0] than is used faster version.
        """
        if a[0] == b[0]:
            py = c[1] + (d[1] - c[1]) * (a[0] - c[0]) / (d[0] - c[0])
            return V2((a[0], py))

        a0b0 = a[0] - b[0]
        c1d1 = c[1] - d[1]
        a1b1 = a[1] - b[1]
        c0d0 = c[0] - d[0]
        ab = a[0] * b[1] - a[1] * b[0]
        cd = c[0] * d[1] - c[1] * d[0]
        den = a0b0 * c1d1 - a1b1 * c0d0
        px = (ab * c0d0 - a0b0 * cd) / den
        py = (ab * c1d1 - a1b1 * cd) / den
        return V2((px, py))

    def line_point_dist(a, b, p):
        """Returns distance from point (p) to line segment (a, b)."""
        # outside a, b
        ab = b - a
        ap = p - a
        if dot(ab, ap) <= 0.0:
            return norm(ap)
        bp = p - b
        if dot(ab, bp) >= 0.0:
            return norm(bp)

        # inside a, b
        return math.fabs(ab[1] * p[0] - ab[0] * p[1] + b[0] * a[1] - b[1] * a[0]) / norm(ab)

    def find_path_quad(quad, lay, fun, alpha0_min, alpha0_max):
        #print(quad)
        pos, dir, tr, time = fun(alpha0_min)
        min_line = 1
        p2 = quad[2] - pos
        if vec_ori(p2, dir) > 0:
            min_line = 2
        p3 = quad[3] - pos
        if vec_ori(p3, dir) > 0:
            min_line = 3
        #print(min_line)

        pos, dir, tr, time = fun(alpha0_max)
        max_line = 1
        p2 = quad[2] - pos
        if vec_ori(p2, dir) > 0:
            max_line = 2
        p3 = quad[3] - pos
        if vec_ori(p3, dir) > 0:
            max_line = 3
        #print(max_line)

        ret = []

        def four_is_zero(x):
            if x == 4:
                return 0
            return x

        if max_line != min_line:
            if max_line > min_line:
                base_line = min_line
                base_point = min_line
                line_dir = 1
            else:
                base_line = min_line
                base_point = min_line + 1
                line_dir = -1

            q = quad[base_point + line_dir]

            def f(alpha0):
                pos, dir, tr, time = fun(alpha0)
                #p2 = quad[base_point + line_dir] - pos
                p2 = q - pos
                return vec_ori(p2, dir)

            # todo: nemusi fungovat kdyz uhel bude vetsi necz 90, mozna lepsi bisect - neni monotoni, myslim ze mono neni potreba - brenth je vyrazne rychlejsi
            #alpha0 = brenth(f, alpha0_min, alpha0_max, xtol=1e-5)
            # print("-------------------")
            # print(base_line)
            # print(base_point)
            # print(line_dir)
            # print(alpha0_min)
            # print(alpha0_max)
            # print(f(alpha0_min))
            # print(f(alpha0_max))
            alpha0 = brenth(f, alpha0_min, alpha0_max, xtol=xtol)

            if abs(max_line - min_line) > 1 and alpha0_max - alpha0 >= min_alpha0_span:
                q2 = quad[base_point + 2 * line_dir]

                def f(alpha0):
                    pos, dir, tr, time = fun(alpha0)
                    #p2 = quad[base_point + 2 * line_dir] - pos
                    p2 = q2 - pos
                    return vec_ori(p2, dir)

                # print(alpha0)
                # print(alpha0_max)
                # print(f(alpha0))
                # print(f(alpha0_max))
                alpha0_2 = brenth(f, alpha0, alpha0_max, xtol=xtol)

            qa1 = quad[base_line]
            qa2 = quad[four_is_zero(base_line + 1)]

            @fun_cache
            def fun2(alpha0):
                pos, dir, tr, time = fun(alpha0)

                #new_pos = line_intersect(pos, pos + dir, quad[base_line], quad[four_is_zero(base_line + 1)])
                new_pos = line_intersect(qa1, qa2, pos, pos + dir)

                if compute_traces:
                    tr = [tr[0] + [new_pos[0]], tr[1] + [new_pos[1]]]

                time += norm(new_pos - pos) / layers_speed[lay + 1]

                return new_pos, dir, tr, time
                #return pos, dir, trace, time

            if alpha0 - alpha0_min >= min_alpha0_span:
                ret.append((min_line, fun2, alpha0_min, alpha0))


            qb1 = quad[base_line + line_dir]
            qb2 = quad[four_is_zero(base_line + line_dir + 1)]

            @fun_cache
            def fun2(alpha0):
                pos, dir, tr, time = fun(alpha0)

                #print(min_line)
                #new_pos = line_intersect(pos, pos + dir, quad[base_line + line_dir], quad[four_is_zero(base_line + line_dir + 1)])
                new_pos = line_intersect(qb1, qb2, pos, pos + dir)

                if compute_traces:
                    tr = [tr[0] + [new_pos[0]], tr[1] + [new_pos[1]]]

                time += norm(new_pos - pos) / layers_speed[lay + 1]

                return new_pos, dir, tr, time

            if abs(max_line - min_line) > 1 and alpha0_max - alpha0 >= min_alpha0_span:
                if alpha0_2 - alpha0 >= min_alpha0_span:
                    ret.append((min_line + line_dir, fun2, alpha0, alpha0_2))
            else:
                if alpha0_max - alpha0 >= min_alpha0_span:
                    ret.append((min_line + line_dir, fun2, alpha0, alpha0_max))

            if abs(max_line - min_line) > 1 and alpha0_max - alpha0 >= min_alpha0_span:
                qc1 = quad[base_line + 2 * line_dir]
                qc2 = quad[four_is_zero(base_line + 2 * line_dir + 1)]

                @fun_cache
                def fun2(alpha0):
                    pos, dir, tr, time = fun(alpha0)

                    #print(min_line)
                    #new_pos = line_intersect(pos, pos + dir, quad[base_line + 2 * line_dir], quad[four_is_zero(base_line + 2 * line_dir + 1)])
                    new_pos = line_intersect(qc1, qc2, pos, pos + dir)

                    if compute_traces:
                        tr = [tr[0] + [new_pos[0]], tr[1] + [new_pos[1]]]

                    time += norm(new_pos - pos) / layers_speed[lay + 1]

                    return new_pos, dir, tr, time

                if alpha0_max - alpha0_2 >= min_alpha0_span:
                    ret.append((min_line + 2 * line_dir, fun2, alpha0_2, alpha0_max))
        else:
            qd1 = quad[min_line]
            qd2 = quad[four_is_zero(min_line + 1)]

            @fun_cache
            def fun2(alpha0):
                pos, dir, tr, time = fun(alpha0)

                #new_pos = line_intersect(pos, pos + dir, quad[min_line], quad[four_is_zero(min_line + 1)])
                new_pos = line_intersect(qd1, qd2, pos, pos + dir)

                if compute_traces:
                    tr = [tr[0] + [new_pos[0]], tr[1] + [new_pos[1]]]

                time += norm(new_pos - pos) / layers_speed[lay + 1]

                return new_pos, dir, tr, time

            if alpha0_max - alpha0_min >= min_alpha0_span:
                ret.append((min_line, fun2, alpha0_min, alpha0_max))

        #print(ret)
        return ret

    def find_path_lay(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        if lay > 0:
            hit_seg_list = hit_seg_list + [(lay, seg)]

        if False:#lay == 0:
            find_path_receiver(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
            find_path_test(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)

        else:
            if depth >= max_depth:
                return
            depth += 1

            n = normalize(V2((layers_geo[lay][seg] - layers_geo[lay][seg + 1], grid[seg + 1] - grid[seg])))
            pos_min, dir_min, _, _ = fun(alpha0_min)
            if vec_ori(V2((grid[seg + 1], layers_geo[lay][seg + 1])) - V2((grid[seg], layers_geo[lay][seg])), dir_min) > 0.0:
                n = -n

            #pos_min, dir_min, _ = fun(alpha0_min)
            pos_max, dir_max, _, _ = fun(alpha0_max)
            det_min = vec_ori(n, dir_min)
            det_max = vec_ori(n, dir_max)

            if (det_min > 0.0 and det_max < 0.0) or (det_min < 0.0 and det_max > 0.0):
                def f(alpha0):
                    pos, dir, tr, time = fun(alpha0)
                    return vec_ori(n, dir)

                alpha0 = brenth(f, alpha0_min, alpha0_max, xtol=xtol)

                priority = priority_fun(fun, alpha0_min, alpha0, source_loc)
                if not math.isinf(priority) or compute_traces:
                    pq_push(priority, find_path_lay2, (lay, seg, fun, alpha0_min, alpha0, depth, source_loc, hit_seg_list))

                priority = priority_fun(fun, alpha0, alpha0_max, source_loc)
                if not math.isinf(priority) or compute_traces:
                    pq_push(priority, find_path_lay2, (lay, seg, fun, alpha0, alpha0_max, depth, source_loc, hit_seg_list))
            else:
                priority = priority_fun(fun, alpha0_min, alpha0_max, source_loc)
                if not math.isinf(priority) or compute_traces:
                    pq_push(priority, find_path_lay2, (lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list))

    def find_path_lay2(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        if lay == 0:
            find_path_receiver(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
            find_path_test(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
            return

        # todo: predpocitat
        n = normalize(V2((layers_geo[lay][seg] - layers_geo[lay][seg + 1], grid[seg + 1] - grid[seg])))
        pos_min, dir_min, _, _ = fun(alpha0_min)
        #print("dir")
        #print(dir_min)

        speed_ratio = layers_speed[lay + 1] / layers_speed[lay]

        upward_ray = False
        if vec_ori(V2((grid[seg + 1], layers_geo[lay][seg + 1])) - V2((grid[seg], layers_geo[lay][seg])), dir_min) > 0.0:
            n = -n
            speed_ratio = 1.0 / speed_ratio
            upward_ray = True

        def min_max(a, b):
            if a > b:
                return b, a
            return a, b


        #print(n)
        if speed_ratio > 1.0:


            def f(alpha0):
                pos, dir, tr, time = fun(alpha0)

                i = dir

                cos_i = -dot(i, n)
                sin_t2 = (speed_ratio) ** 2 * (1 - cos_i ** 2)

                return sin_t2 - 1.0

            # todo: zbytecne se pocita fun, muze se vyjit z predchoziho vypoctu
            f_min = f(alpha0_min)
            f_max = f(alpha0_max)

            if f_min > f_max:
                f_min, f_max = f_max, f_min
                alpha0_min, alpha0_max = alpha0_max, alpha0_min

            if (f_min > 0.0 and f_max < 0.0) or (f_min < 0.0 and f_max > 0.0):
                alpha0_2 = brenth(f, alpha0_min, alpha0_max, xtol=xtol)

            #def fa():
            if f_min < 0.0:
                @fun_cache
                def fun2(alpha0):
                    pos, dir, tr, time = fun(alpha0)
                    i = dir
                    cos_i = -dot(i, n)
                    sin_t2 = (speed_ratio) ** 2 * (1 - cos_i ** 2)
                    if sin_t2 > 1.0:
                        #print("sin!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        sin_t2 = 1.0
                    t = i * (speed_ratio) + n * ((speed_ratio) * cos_i - math.sqrt(1.0 - sin_t2))
                    dir = t
                    return pos, dir, tr, time

                if f_max > 0.0:
                    if lay >= len(layers_geo) - 1:
                        find_path_test(lay, seg, fun, *min_max(alpha0_min, alpha0_2), depth, source_loc, hit_seg_list)
                    else:
                        find_path_lay3(lay, seg, fun2, *min_max(alpha0_min, alpha0_2), depth, source_loc, hit_seg_list)
                else:
                    if lay >= len(layers_geo) - 1:
                        find_path_test(lay, seg, fun, *min_max(alpha0_min, alpha0_max), depth, source_loc, hit_seg_list)
                    else:
                        find_path_lay3(lay, seg, fun2, *min_max(alpha0_min, alpha0_max), depth, source_loc, hit_seg_list)

            #def fb():
            if f_max > 0.0:
                @fun_cache
                def fun2(alpha0):
                    pos, dir, tr, time = fun(alpha0)
                    i = dir
                    cos_i = -dot(i, n)
                    r = i + n * 2 * cos_i
                    dir = r
                    return pos, dir, tr, time

                if f_min < 0.0:
                    find_path_lay3(lay, seg, fun2, *min_max(alpha0_2, alpha0_max), depth, source_loc, hit_seg_list)
                else:
                    find_path_lay3(lay, seg, fun2, *min_max(alpha0_min, alpha0_max), depth, source_loc, hit_seg_list)

            # if upward_ray:
            #     fb()
            #     fa()
            # else:
            #     fa()
            #     fb()
        else:
            if lay >= len(layers_geo) - 1:
                find_path_test(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
            else:

                @fun_cache
                def fun2(alpha0):
                    pos, dir, tr, time = fun(alpha0)

                    i = dir

                    cos_i = -dot(i, n)
                    sin_t2 = (speed_ratio) ** 2 * (1 - cos_i ** 2)

                    if sin_t2 > 1.0:
                        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                        sin_t2 = 1.0

                    t = i * (speed_ratio) \
                        + n * ((speed_ratio) * cos_i - math.sqrt(1.0 - sin_t2))

                    dir = t
                    # print(dir)


                    return pos, dir, tr, time

                find_path_lay3(lay, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)

    def find_path_lay3(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        # todo: staci opravdu alpha0_min? asi muze byt rovnobezne, muzu dat neco mezi min max
        pos_min, dir_min, _, _ = fun(alpha0_min)
        pos_max, dir_max, _, _ = fun(alpha0_max)
        # print(dir_min)
        # print("ori !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(vec_ori(np.array([grid[seg + 1], layers_geo[lay][seg + 1]]) - np.array([grid[seg], layers_geo[lay][seg]]), dir_min))
        # print(vec_ori(np.array([grid[seg + 1], layers_geo[lay][seg + 1]]) - np.array([grid[seg], layers_geo[lay][seg]]), dir_max))
        if vec_ori(V2((grid[seg + 1], layers_geo[lay][seg + 1])) - V2((grid[seg], layers_geo[lay][seg])), dir_min) > 0.001:
        #if dir_min[1] > 0:
            quad = [V2((grid[seg], layers_geo[lay][seg])),
                    V2((grid[seg + 1], layers_geo[lay][seg])),
                    V2((grid[seg + 1], layers_geo[lay - 1][seg + 1])),
                    V2((grid[seg], layers_geo[lay - 1][seg]))]
            # print(alpha0_min)
            # print(alpha0_max)
            # alpha0_min = 1.0
            #print("tadygggggggggggggggggg")
            try:
                q_ret = find_path_quad(quad, lay - 1, fun, alpha0_min, alpha0_max)
            except ValueError:
                return
            #print(q_ret)

            # first process upward ray
            #q_ret = sorted(q_ret, key=lambda x: x[0] != 2)


            for line, fun2, alpha0_min, alpha0_max in q_ret:
                if line == 1:
                    find_path_grid(lay - 1, seg + 1, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                elif line == 2:
                    find_path_lay(lay - 1, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                else:
                    find_path_grid(lay - 1, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
        elif vec_ori(V2((grid[seg + 1], layers_geo[lay][seg + 1])) - V2((grid[seg], layers_geo[lay][seg])), dir_min) < 0.001:
        #else:
            # print(seg)
            # print(lay)
            # if lay == 3:
            #     1/0
            quad = [V2((grid[seg + 1], layers_geo[lay][seg + 1])),
                    V2((grid[seg], layers_geo[lay][seg])),
                    V2((grid[seg], layers_geo[lay + 1][seg])),
                    V2((grid[seg + 1], layers_geo[lay + 1][seg + 1]))]
            # print(alpha0_min)
            # print(alpha0_max)
            # alpha0_min = 1.0
            try:
                q_ret = find_path_quad(quad, lay, fun, alpha0_min, alpha0_max)
            except ValueError:
                return

            for line, fun2, alpha0_min, alpha0_max in q_ret:
                if line == 1:
                    find_path_grid(lay, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                elif line == 2:
                    find_path_lay(lay + 1, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                else:
                    find_path_grid(lay, seg + 1, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)


    def find_path_grid(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        pos_min, dir_min, _, _ = fun(alpha0_min)

        if (seg == 0 and dir_min[0] < 0.0) or (seg >= len(grid) - 1 and dir_min[0] > 0.0) or depth >= max_depth:
            find_path_test(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
        else:

            pos_max, dir_max, _, _ = fun(alpha0_max)

            if (dir_min[1] > 0.0 and dir_max[1] < 0.0) or (dir_min[1] < 0.0 and dir_max[1] > 0.0):
                def f(alpha0):
                    pos, dir, tr, time = fun(alpha0)
                    return dir[1]

                alpha0 = brenth(f, alpha0_min, alpha0_max, xtol=xtol)

                priority = priority_fun(fun, alpha0_min, alpha0, source_loc)
                if not math.isinf(priority) or compute_traces:
                    pq_push(priority, find_path_grid2, (lay, seg, fun, alpha0_min, alpha0, depth, source_loc, hit_seg_list))

                priority = priority_fun(fun, alpha0, alpha0_max, source_loc)
                if not math.isinf(priority) or compute_traces:
                    pq_push(priority, find_path_grid2, (lay, seg, fun, alpha0, alpha0_max, depth, source_loc, hit_seg_list))
            else:
                priority = priority_fun(fun, alpha0_min, alpha0_max, source_loc)
                if not math.isinf(priority) or compute_traces:
                    pq_push(priority, find_path_grid2, (lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list))

    def find_path_grid2(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        pos_min, dir_min, _, time_min = fun(alpha0_min)

        if False:#(seg == 0 and dir_min[0] < 0.0) or (seg >= len(grid) - 1 and dir_min[0] > 0.0) or depth >= max_depth:
            find_path_test(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
        else:
            depth += 1

            if dir_min[0] > 0.001:
                quad = [V2((grid[seg], layers_geo[lay][seg])),
                        V2((grid[seg], layers_geo[lay + 1][seg])),
                        V2((grid[seg + 1], layers_geo[lay + 1][seg + 1])),
                        V2((grid[seg + 1], layers_geo[lay][seg + 1]))]
                try:
                    q_ret = find_path_quad(quad, lay, fun, alpha0_min, alpha0_max)
                except ValueError:
                    return

                for line, fun2, alpha0_min, alpha0_max in q_ret:
                    if line == 1:
                        find_path_lay(lay + 1, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                    elif line == 2:
                        find_path_grid(lay, seg + 1, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                    else:
                        find_path_lay(lay, seg, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
            elif dir_min[0] < 0.001:
                quad = [V2((grid[seg], layers_geo[lay + 1][seg])),
                        V2((grid[seg], layers_geo[lay][seg])),
                        V2((grid[seg - 1], layers_geo[lay][seg - 1])),
                        V2((grid[seg - 1], layers_geo[lay + 1][seg - 1]))]
                try:
                    q_ret = find_path_quad(quad, lay, fun, alpha0_min, alpha0_max)
                except ValueError:
                    return

                for line, fun2, alpha0_min, alpha0_max in q_ret:
                    if line == 1:
                        find_path_lay(lay, seg - 1, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                    elif line == 2:
                        find_path_grid(lay, seg - 1, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)
                    else:
                        find_path_lay(lay + 1, seg - 1, fun2, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)


    def find_path_receiver(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        #return
        num = 0
        receivers = required_first_arrival_dict[source_loc]
        for rec_loc in receivers:
            pos, rec_seg_list = find_source_pos(rec_loc)
            for rec_seg in rec_seg_list:
                # print(seg)
                # print(fun(alpha0_min)[0][0])
                # print(fun(alpha0_max)[0][0])
                # todo: poresit pripad, ze je receiver na kraji segmentu, diky numerice muze byt mimo interval
                if rec_seg == seg and (fun(alpha0_min)[0][0] <= rec_loc <= fun(alpha0_max)[0][0] or fun(alpha0_min)[0][0] >= rec_loc >= fun(alpha0_max)[0][0]):
                    min_time = min(fun(alpha0_min)[3], fun(alpha0_max)[3])
                    if min_time < first_arrival[(source_loc, rec_loc)] or (return_traces and not test_traces):
                        def f(alpha0):
                            pos, dir, tr, time = fun(alpha0)
                            return pos[0] - rec_loc

                        num += 1
                        alpha0 = brenth(f, alpha0_min, alpha0_max, xtol=xtol)
                        pos, dir, tr, time = fun(alpha0)
                    else:
                        time = math.inf
                    if time < first_arrival[(source_loc, rec_loc)]:
                        first_arrival[(source_loc, rec_loc)] = time

                        if return_traces and not test_traces:
                            if (source_loc, rec_loc) in traces_map:
                                trx = traces[traces_map[(source_loc, rec_loc)]]
                            else:
                                trx = []
                                traces.append(trx)
                                traces_map[(source_loc, rec_loc)] = len(traces) - 1
                            trx.insert(0, [tr[0], tr[1], True, (source_loc, rec_loc), time])

                        hit_seg_map[(source_loc, rec_loc)] = hit_seg_list
                    else:
                        if return_traces and not test_traces:
                            if (source_loc, rec_loc) in traces_map:
                                trx = traces[traces_map[(source_loc, rec_loc)]]
                            else:
                                trx = []
                                traces.append(trx)
                                traces_map[(source_loc, rec_loc)] = len(traces) - 1
                            trx.append([tr[0], tr[1], False, (source_loc, rec_loc), time])
        #print(num)



    def find_path_test(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        pass
        #print("test")
        find_path_test2(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list)

    num_path = 0
    num_no_mono = 0

    def find_path_test2(lay, seg, fun, alpha0_min, alpha0_max, depth, source_loc, hit_seg_list):
        nonlocal num_path, num_no_mono

        #return
        if test_traces:
            #print(alpha0_min, alpha0_max)


            num_path += 1
            trx = []
            mono = []
            n = 10
            for i in range(n):
                pos, dir, tr, time = fun(alpha0_min + (alpha0_max - alpha0_min) / n * i)
                # print(tr)
                trx.append([tr[0], tr[1], True, (source_loc, 0.0), time])

                mono.append(time)

            # mono test
            up = True
            down = True
            for i in range(1, len(mono)):
                if mono[i] < mono[i - 1]:
                    up = False
                if mono[i] > mono[i - 1]:
                    down = False
            if not up and not down:
                num_no_mono += 1
                # print("neni mono")
                # print(alpha0_max - alpha0_min)
                # print(mono)
                #plt.plot(mono)
                #plt.show()


                #if num_no_mono == 1:
            traces.append(trx)

    def mono_test(fun, alpha0_min, alpha0_max):
        mono = []
        n = 100
        for i in range(n):
            pos, dir, tr, time = fun(alpha0_min + (alpha0_max - alpha0_min) / n * i)
            mono.append(time)
        up = True
        down = True
        for i in range(1, len(mono)):
            if mono[i] < mono[i - 1]:
                up = False
            if mono[i] > mono[i - 1]:
                down = False
        if not up and not down:
            print("neni mono")
            #raise KeyboardInterrupt
            # print(alpha0_max - alpha0_min)
            # print(mono)
            plt.plot(mono)
            plt.show()

    def get_max_time(source_loc):
        max_time = 0.0
        for rl in required_first_arrival_dict[source_loc]:
            if first_arrival[(source_loc, rl)] > max_time:
                max_time = first_arrival[(source_loc, rl)]
        return max_time

    def priority_fun(fun, alpha0_min, alpha0_max, source_loc):
        #return 0.0
        #mono_test(fun, alpha0_min, alpha0_max)
        pos_min, _, _, time_min = fun(alpha0_min)
        pos_max, _, _, time_max = fun(alpha0_max)
        min_time = min(time_min, time_max)

        # max_y_pos = max(pos_min[1], pos_max[1])
        # min_surf = min(layers_geo[0])
        # if max_y_pos < min_surf:
        #     min_time += (min_surf - max_y_pos) / max_inside_speed
        #
        # mt = get_max_time(source_loc)
        # if min_time >= mt:
        #     return math.inf
        # else:
        #     return min_time

        priority = math.inf
        for rl in required_first_arrival_dict[source_loc]:
            rec_pos, _ = find_source_pos(rl)
            if min_time >= priority:# or min_time >= first_arrival[(source_loc, rl)]:
                break
            if min_time >= first_arrival[(source_loc, rl)]:
                continue
            t = min_time + line_point_dist(pos_min, pos_max, rec_pos) / max_inside_speed
            if t < first_arrival[(source_loc, rl)]:
                if t < priority:
                    priority = t
        return priority


    def find_path_first(source_loc):



        source_x = find_source_pos(source_loc)


        def fff1(seg):
            source = (source_x[0], seg)
            if source_loc != grid[seg]:
                @fun_cache
                def fun(alpha0):
                    pos = V2((grid[source[1]], source[0][1] - (source[0][0] - grid[source[1]]) * math.tan(alpha0 + math.pi / 2)))
                    nps = norm(pos - source[0])
                    dir = (pos - source[0]) / nps
                    if compute_traces:
                        trace = [[source[0][0], pos[0]], [source[0][1], pos[1]]]
                    else:
                        trace = []
                    time = nps / layers_speed[1]
                    return pos, dir, trace, time

                alpha0_min = math.atan2(-source[0][0] + grid[source[1]], source[0][1] - layers_geo[0][source[1]])
                alpha0_max = math.atan2(source[0][1] - layers_geo[1][source[1]], source[0][0] - grid[source[1]]) - math.pi / 2
                if alpha0_max - alpha0_min >= min_alpha0_span:
                    find_path_grid(0, source[1], fun, alpha0_min, alpha0_max, 1, source_loc, [])
                    # priority = priority_fun(fun, alpha0_min, alpha0_max, source_loc)
                    # if not math.isinf(priority) or compute_traces:
                    #     pq_push(priority, find_path_grid, (0, source[1], fun, alpha0_min, alpha0_max, 1, source_loc))
                    #find_path_test(0, seg, fun, alpha0_min, alpha0_max, 1, source_loc)

        def fff2(seg):
            source = (source_x[0], seg)

            alpha0_min = math.atan2(source[0][1] - layers_geo[1][source[1]], source[0][0] - grid[source[1]]) - math.pi / 2
            alpha0_max = math.atan2(-source[0][1] + layers_geo[1][source[1] + 1], -source[0][0] + grid[source[1] + 1]) + math.pi / 2

            lay_dir = (V2((grid[source[1] + 1] - grid[source[1]],
                                 layers_geo[1][source[1] + 1] - layers_geo[1][source[1]]))) / norm(V2(
                (grid[source[1] + 1] - grid[source[1]], layers_geo[1][source[1] + 1] - layers_geo[1][source[1]])))
            lay_len = norm(V2(
                (grid[source[1] + 1] - grid[source[1]], layers_geo[1][source[1] + 1] - layers_geo[1][source[1]])))
            aaa = lay_dir * (lay_len / (alpha0_max - alpha0_min))

            @fun_cache
            def fun(alpha0):
                pos = V2((grid[source[1]], layers_geo[1][source[1]])) + aaa * (alpha0 - alpha0_min)
                nps = norm(pos - source[0])
                dir = (pos - source[0]) / nps
                if compute_traces:
                    trace = [[source[0][0], pos[0]], [source[0][1], pos[1]]]
                else:
                    trace = []
                time = nps / layers_speed[1]
                return pos, dir, trace, time

            if alpha0_max - alpha0_min >= min_alpha0_span:
                find_path_lay(1, source[1], fun, alpha0_min, alpha0_max, 1, source_loc, [])
                # priority = priority_fun(fun, alpha0_min, alpha0_max, source_loc)
                # if not math.isinf(priority) or compute_traces:
                #     pq_push(priority, find_path_lay, (1, source[1], fun, alpha0_min, alpha0_max, 1, source_loc))
                #find_path_test(1, seg, fun, alpha0_min, alpha0_max, 1, source_loc)

        def fff3(seg):
            source = (source_x[0], seg)

            if source_loc != grid[seg + 1]:
                @fun_cache
                def fun(alpha0):
                    pos = V2((grid[source[1] + 1], source[0][1] + (-source[0][0] + grid[source[1] + 1]) * math.tan(alpha0 - math.pi / 2)))
                    nps = norm(pos - source[0])
                    dir = (pos - source[0]) / nps
                    if compute_traces:
                        trace = [[source[0][0], pos[0]], [source[0][1], pos[1]]]
                    else:
                        trace = []
                    time = nps / layers_speed[1]
                    return pos, dir, trace, time


                alpha0_min = math.atan2(-source[0][1] + layers_geo[1][source[1] + 1], -source[0][0] + grid[source[1] + 1]) + math.pi / 2
                alpha0_max = math.atan2(-source[0][1] + layers_geo[0][source[1] + 1], -source[0][0] + grid[source[1] + 1]) + math.pi / 2
                if alpha0_max - alpha0_min >= min_alpha0_span:
                    find_path_grid(0, source[1] + 1, fun, alpha0_min, alpha0_max, 1, source_loc, [])
                    # priority = priority_fun(fun, alpha0_min, alpha0_max, source_loc)
                    # if not math.isinf(priority) or compute_traces:
                    #     pq_push(priority, find_path_grid, (0, source[1] + 1, fun, alpha0_min, alpha0_max, 1, source_loc))
                    #find_path_test(0, seg, fun, alpha0_min, alpha0_max, 1, source_loc)

        for seg in source_x[1]:
            fff1(seg)
            fff2(seg)
            fff3(seg)

    # priority queue
    priority_queue = []
    pq_item_count = 0

    def pq_clear():
        nonlocal pq_item_count
        priority_queue.clear()
        pq_item_count = 0

    def pq_push(priority, fun, args):
        nonlocal pq_item_count
        pq_item_count += 1
        heapq.heappush(priority_queue, (priority, pq_item_count, fun, args))
        return pq_item_count

    def pq_pop():
        priority, pq_item_count, fun, args = heapq.heappop(priority_queue)
        return priority, pq_item_count, fun, args

    first_arrival = {}

    def surface_arrival():
        """Arrivals from surface speed."""
        flat_surface = True
        first = layers_geo[0][0]
        for i in range(1, len(grid)):
            if layers_geo[0][i] != first:
                flat_surface = False

        surface_speed = layers_speed[0]
        if not flat_surface:
            seg_num = len(grid) - 1
            seg_len = [0.0] * seg_num
            for i in range(seg_num):
                seg_len[i] = norm(V2((grid[i + 1], layers_geo[0][i + 1])) - V2((grid[i], layers_geo[0][i])))
        for rfa in processed:  # todo: processed prejmenovat pokud bude daleko od vytvoreni
            if rfa[1] == rfa[0]:
                first_arrival[rfa] = 0.0
            elif surface_speed == 0.0:
                first_arrival[rfa] = math.inf
            elif flat_surface:
                first_arrival[rfa] = math.fabs(rfa[1] - rfa[0]) / surface_speed
            else:
                if rfa[1] > rfa[0]:
                    left = rfa[0]
                    right = rfa[1]
                else:
                    left = rfa[1]
                    right = rfa[0]
                l_pos, l_seg_list = find_source_pos(left)
                r_pos, r_seg_list = find_source_pos(right)
                l_seg = max(l_seg_list)
                r_seg = min(r_seg_list)
                if r_seg == l_seg:
                    dist = norm(r_pos - l_pos)
                else:
                    dist = norm(V2((grid[l_seg + 1], layers_geo[0][l_seg + 1])) - l_pos) + norm(r_pos - V2((grid[r_seg], layers_geo[0][r_seg])))
                dist += math.fsum(seg_len[l_seg + 1: r_seg])
                first_arrival[rfa] = dist / surface_speed

    surface_arrival()

    for source_loc in required_first_arrival_dict:
        pq_clear()

        find_path_first(source_loc)

        fid = 0
        while priority_queue:
            old_priority, pq_item_id, fun, args = pq_pop()
            #p = False
            #print(fun, args)
            # if fid != pq_item_id:
            #     priority = priority_fun(*args[2:5], args[6])
            #     if priority > old_priority and priority_queue and priority > priority_queue[0][0] and not math.isinf(priority):
            #         id = pq_push(priority, fun, args)
            #         if fid == 0:
            #             fid = id
            #         p = True
            #         continue
            #         # todo: muze se zbytecne pocitat znovu priorita puvodni polozky
            # p = False
            # fid = 0
            if not math.isinf(old_priority) or compute_traces:
                fun(*args)


    # inverse first arrivals
    for rfa in inverse_required_first_arrival:
        first_arrival[rfa] = first_arrival[(rfa[1], rfa[0])]

    # unmark slow traces
    if return_traces and not test_traces:
        for receiver in traces:
            if len(receiver) >= 2:
                for i in range(1, len(receiver)):
                    receiver[i][2] = False

    # hit segments
    #def hhhh():
    for k, hit_seg_list in hit_seg_map.items():
        if (k[1], k[0]) in inverse_required_first_arrival:
            inc = 2
        else:
            inc = 1
        for lay, seg in hit_seg_list:
            hit_seg[lay][seg] += inc
    #hhhh()

    # print("num_path")
    # print(num_path)
    #print(num_no_mono)

    return first_arrival, traces, hit_seg


def plot_traces_poly(grid, layers_geo, layers_speed, first_arrivals, traces, hit_seg, ref_first_arrivals=None, show_ref_dif=False, axis_equal=False):
    """
    Plots traces.
    :param grid:
    :param layers_geo:
    :param layers_speed:
    :param first_arrivals:
    :param traces:
    :param ref_first_arrivals:
    :param show_ref_dif:
    :param axis_equal:
    :return:
    """
    # borderlines
    for lay in layers_geo:
        plt.plot(grid, lay, "k")

    # traces
    dif_min = 0.0
    dif_max = 0.0
    if show_ref_dif:
        for receiver in traces:
            for tr in receiver:
                dif = tr[4] - ref_first_arrivals[tr[3]]
                if dif < dif_min:
                    dif_min = dif
                elif dif > dif_max:
                    dif_max = dif

        norm = mpl.colors.Normalize(vmin=dif_min, vmax=dif_max)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
        cmap.set_array([])

        cbar = plt.colorbar(cmap)
        cbar.ax.set_ylabel("First arrival difference [s]")

    for receiver in traces:
        lines = []
        for tr in receiver:
            if tr[0]:
                if lines:
                    color = lines[0].get_color()
                else:
                    color = None

                if tr[2]:
                    lw = 1.5
                    ls = "-"
                else:
                    lw = 0.5
                    ls = "--"

                if show_ref_dif:
                    dif = tr[4] - ref_first_arrivals[tr[3]]
                    color = cmap.to_rgba(dif)

                lines = plt.plot(tr[0], tr[1], c=color, lw=lw, ls=ls)

    # receiver locations
    x = list({fa[1] for fa in first_arrivals})
    plt.plot(x, [0.0] * len(x), "r+")

    # source location
    x = list({fa[0] for fa in first_arrivals})
    plt.plot(x, [0.0] * len(x), "gx")

    # speed
    y_span = max(layers_geo[0]) - min(layers_geo[-1])
    for i in range(len(layers_geo)):
        if i < len(layers_geo) - 1:
            y = (layers_geo[i][-1] + layers_geo[i + 1][-1]) / 2
        else:
            y = layers_geo[i][-1] - y_span / 50
        plt.text(grid[-1] + 1.0, y, "v = {:.0f} m/s".format(layers_speed[i + 1]), va="center")

    # hit segments
    for lay in range(len(layers_geo)):
        for seg in range(len(grid) - 1):
            num = hit_seg[lay][seg]
            if num > 0:
                x = (grid[seg] + grid[seg + 1]) / 2
                y = (layers_geo[lay][seg] + layers_geo[lay][seg + 1]) / 2 + y_span / 75
                plt.text(x, y, num, ha="center")

    # labels
    plt.title("Traces")
    plt.xlabel("Position [m]")
    plt.ylabel("Depth [m]")

    # space for labels
    left, right = plt.xlim()
    new_fight = right + (right - left) * 0.1
    plt.xlim(right=new_fight)

    plt.grid(True)
    if axis_equal:
        plt.axis("equal")

    plt.show()
