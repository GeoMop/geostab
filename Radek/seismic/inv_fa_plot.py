import seismic_operations as so

import sys
import json


with open(sys.argv[1], "r") as fd:
    layers_list = json.load(fd)

so.plot_depth_speed(layers_list)
