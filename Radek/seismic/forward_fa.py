import seismic_operations as so

import time


# define layers
layers = [(10.0, 400.0), (10.0, 300.0), (20.0, 1000.0), (20.0, 2000.0), (10.0, 3000.0)]

receiver_locations = [i * 20.0 for i in range(1, 6)]

# find first arrivals
t = time.time()
fa, tr = so.forward_fa(layers, receiver_locations)
print("Computation time: {:.3f} ms".format((time.time() - t) * 1000))
print("First arrivals: {}".format(fa))

so.plot_traces(layers, receiver_locations, fa, tr)
