import seismic_operations as so

import numpy as np
from scipy.optimize import differential_evolution

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
#required_fa = [(j*20.0, i*4.0) for j in range(0, 5) for i in range(0, 21)]
required_fa = first_arrival_xls.keys()
#required_fa = [k for k in required_fa if 80.0 >= k[0] >= 0.0]

# create map from (source_location, receiver_location) to index in optimize vector
trace_to_xi = so.xdiff_create_map(sm, required_fa)

# create bounds
bounds = so.xdiff_create_bounds(sm, trace_to_xi, 0.008, 1500, 600)

# load or compute optimized vector
if len(sys.argv) > 1:
    with open(sys.argv[1], "r") as fd:
        result_x = np.array(json.load(fd))
else:
    # create initial_population
    init = "latinhypercube"
    #init = "random"

    t = time.time()
    diff_weight = 2e+1
    inv_weight = 1e+5
    args = (sm, required_fa, trace_to_xi, diff_weight, inv_weight)
    result = differential_evolution(so.xdiff_crit_fun, bounds, args, init=init,
                                    strategy="best1bin", popsize=100, tol=1e-5, maxiter=1000,
                                    polish=True, disp=True)  # workers=-1
    print("Final f(x)= {:.6g}".format(result.fun))
    print("Optimization time: {:.3f} s".format(time.time() - t))
    result_x = result.x
    with open("out.txt", 'w') as fd:
        json.dump(result_x.tolist(), fd, indent=4, sort_keys=True)

# plot results
first_arrival = so.xdiff_first_arrival_from_x(required_fa, trace_to_xi, result_x)
#first_arrival = {k:v for k, v in first_arrival.items() if k[0] in [40.0, 60.0]}
so.plot_results(sm, first_arrival, first_arrival2=first_arrival_xls, all_traces=True)

# save first arrivals
#so.save_first_arrival(first_arrival, "out_fa.txt")
#so.save_first_arrival(first_arrival_xls, "out_fa_xls.txt")
