import seismic_operations as so

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing

import sys
import time
import json


# load measurements from files
files = ["16{}.dat".format(i) for i in range(1, 10)]
sm = so.load_measurement(files, nsta=40, nlta=60)

# plot characteristic function
# so.plot_cft(sm, (40.0, 44.0))
# sys.exit()

# load first arrivals form xls
first_arrival_xls = so.load_fa_xls("vyčíslení_P3.xls")

# define which first arrivals will be found
#required_fa = [(j*20.0, i*4.0) for j in range(0, 1) for i in range(0, 21)]
required_fa = first_arrival_xls.keys()

# create maps between (source_location, receiver_location) and index in optimize vector
trace_to_xi, xi_to_trace = so.create_map(required_fa)

# create bounds
bounds = so.create_bounds(sm, xi_to_trace, 0.00, 1500, 0.015, 1000)

# load or compute optimized vector
if len(sys.argv) > 1:
    with open(sys.argv[1], "r") as fd:
        result_x = np.array(json.load(fd))
else:
    # create initial_population
    #init = so.create_initial_population2(sm, xi_to_trace, bounds, popsize=100)
    init = "latinhypercube"
    #init = "random"

    # create x0
    x0 = so.create_x0(sm, xi_to_trace, bounds)
    #x0 = None

    t = time.time()
    diff_weight = 1e+3
    sequence_weight = 1e+5
    args = (sm, trace_to_xi, xi_to_trace, diff_weight, sequence_weight)

    # differential evolution
    # result = differential_evolution(so.crit_fun, bounds, args, init=init,
    #                                 strategy="best1bin", popsize=100, tol=1e-5, maxiter=1000,
    #                                 polish=True, disp=True)  # workers=-1

    # dual annealing
    callback = lambda x, f, context: print("f(x)= {:.6g}".format(f))
    result = dual_annealing(so.crit_fun, bounds, args, maxiter=100, callback=callback, x0=x0)

    print("message: {}".format(result.message))
    print("nfev: {}".format(result.nfev))
    print("Final f(x)= {:.6g}".format(result.fun))
    print("Optimization time: {:.3f} s".format(time.time() - t))
    result_x = result.x
    with open("out.txt", 'w') as fd:
        json.dump(result_x.tolist(), fd, indent=4, sort_keys=True)

# plot results
first_arrival = so.first_arrival_from_x(required_fa, trace_to_xi, result_x)
#first_arrival = {k:v for k, v in first_arrival.items() if k[0] in [40.0, 60.0]}
plt_bounds = so.plt_bounds_from_x_bounds(required_fa, trace_to_xi, bounds)
so.plot_results(sm, first_arrival, first_arrival2=first_arrival_xls, plt_bounds=plt_bounds, all_traces=True)

# save first arrivals
#so.save_first_arrival(first_arrival, "out_fa.txt")
#so.save_first_arrival(first_arrival_xls, "out_fa_xls.txt")
