import seismic_operations as so

from scipy.optimize import differential_evolution

import time
import math


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


def crit_fun(x, lay_width, num_lays, first_arrival, distances, trace_to_di, speed_decrease_weight):
    layers = x_to_layers(x, lay_width, num_lays)

    # speed decrease penalization
    crit = 0.0
    for i in range(1, len(layers)):
        vi = layers[i][1]
        vp = layers[i - 1][1]
        if vi < vp:
            crit += (vp - vi) * speed_decrease_weight

    fa, tr = so.forward_fa(layers, distances)

    # first arrival penalization
    for k, v in first_arrival.items():
        if k in trace_to_di:
            t = fa[trace_to_di[k]]
            crit += math.pow(t - v, 2)

    return crit


# width of one layer
lay_width = 10.0

# number of layers
num_lays = 5

# speed limits
bounds = [(100.0, 5000.0)] * num_lays

t = time.time()
speed_decrease_weight = 1e+5
args = (lay_width, num_lays, first_arrival, distances, trace_to_di, speed_decrease_weight)
result = differential_evolution(crit_fun, bounds, args, init="latinhypercube",
                                strategy="best1bin", popsize=100, tol=1e-5, maxiter=100,
                                polish=True, disp=True)  # workers=-1
print("Final f(x)= {:.6g}".format(result.fun))
print("Optimization time: {:.3f} s".format(time.time() - t))
result_x = result.x
print(result_x)

# plot results
layers = x_to_layers(result_x, lay_width, num_lays)
fa, tr = so.forward_fa(layers, distances)

print("(source, receiver) -> measured, from inversion")
for trace_key in sorted(first_arrival):
    if trace_key in trace_to_di:
        print("({:5}, {:5}) -> {:8.4f}, {:8.4f}".format(trace_key[0], trace_key[1],
                                                        first_arrival[trace_key], fa[trace_to_di[trace_key]]))

so.plot_traces(layers, distances, tr)
