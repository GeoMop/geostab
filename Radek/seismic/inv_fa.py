import seismic_operations as so

from scipy.optimize import differential_evolution, dual_annealing

import time
import math
import json


# load first arrivals
first_arrival = so.load_first_arrival("out_fa_xls.txt")
#first_arrival = {k:v for k, v in first_arrival.items() if k[0] in [40.0]}

# create synthetic first arrivals
# layers_syn = [(10.0, 400.0), (10.0, 300.0), (20.0, 1000.0), (20.0, 2000.0), (10.0, 3000.0)]
# distances_syn = [i * 4.0 for i in range(1, 25)]
# fa, tr = so.forward_fa(layers_syn, distances_syn)
# first_arrival = {(0.0, d):f for d, f in zip(distances_syn, fa)}

# find distances
distances = []
trace_to_di = {}
for trace_key in first_arrival:
    d = math.fabs(trace_key[1] - trace_key[0])
    if d > 0.0:
        try:
            i = distances.index(d)
        except ValueError:
            distances.append(d)
            trace_to_di[trace_key] = len(distances) - 1
        else:
            trace_to_di[trace_key] = i


def x_to_layers(x, lay_width, num_lays):
    return [(lay_width, x[i]) for i in range(num_lays)]


def crit_fun(x, lay_width, num_lays, first_arrival, distances, trace_to_di, speed_decrease_weight, speed_diff_weight):
    layers = x_to_layers(x, lay_width, num_lays)

    crit = 0.0
    for i in range(1, len(layers)):
        vi = layers[i][1]
        vp = layers[i - 1][1]

        # speed decrease penalization
        if vi < vp:
            crit += (vp - vi) * speed_decrease_weight

        # speed difference penalization
        crit += (vi - vp) ** 2 * speed_diff_weight

    fa, tr = so.forward_fa(layers, distances)

    # first arrival penalization
    for k, v in first_arrival.items():
        if k in trace_to_di:
            t = fa[trace_to_di[k]]
            crit += math.pow(t - v, 2)

    return crit


# total depth
total_depth = 50.0

# number of layers
# num_lays = 10

layers_list = []
for num_lays in [5, 7, 10, 14, 20]:
    # width of one layer
    lay_width = total_depth / num_lays

    # speed limits
    speed_min = 500.0
    speed_max = 6000.0
    bounds = [(speed_min, speed_max)] * num_lays

    # create x0
    step = (speed_max - speed_min) / (num_lays + 1)
    x0 = [speed_min + i * step for i in range(1, num_lays + 1)]
    #x0 = None

    t = time.time()
    speed_decrease_weight = 1e+5
    speed_diff_weight = 1e-10
    args = (lay_width, num_lays, first_arrival, distances, trace_to_di, speed_decrease_weight, speed_diff_weight)

    # differential evolution
    result = differential_evolution(crit_fun, bounds, args, init="latinhypercube",
                                    strategy="best1bin", popsize=100, tol=1e-5, maxiter=100,
                                    polish=True, disp=True)  # workers=-1

    # dual annealing
    callback = lambda x, f, context: print("f(x)= {:.6g}".format(f))
    #result = dual_annealing(crit_fun, bounds, args, maxiter=100, callback=callback, x0=x0)

    print("message: {}".format(result.message))
    print("nfev: {}".format(result.nfev))
    print("Final f(x)= {:.6g}".format(result.fun))
    print("Optimization time: {:.3f} s".format(time.time() - t))
    result_x = result.x
    print(result_x)

    # plot results
    layers = x_to_layers(result_x, lay_width, num_lays)
    fa, tr = so.forward_fa(layers, distances)

    layers_list.append(layers)

    # print("(source, receiver) -> measured, from inversion")
    # for trace_key in sorted(first_arrival):
    #     if trace_key in trace_to_di:
    #         print("({:5}, {:5}) -> {:8.4f}, {:8.4f}".format(trace_key[0], trace_key[1],
    #                                                         first_arrival[trace_key], fa[trace_to_di[trace_key]]))

    #so.plot_traces(layers, distances, fa, tr)

with open("layers_list.txt", 'w') as fd:
    json.dump(layers_list, fd, indent=4, sort_keys=True)

so.plot_depth_speed(layers_list)
