from data_types import ElectrodeGroup, Electrode, Measurement

import pandas as pd
import math


def parse(file_name):
    ret = {"warnings": [],
           "errors": [],
           "electrode_groups": [],
           "measurements": []}

    with pd.ExcelFile(file_name) as xls:
        if len(xls.sheet_names) < 2:
            ret["errors"].append("ExcelFile must have at least two sheets.")
            return ret
        if xls.sheet_names[0] != "elektrody":
            ret["warnings"].append('Name of first sheet should be "elektrody".')
        if xls.sheet_names[1] != "měření":
            ret["warnings"].append('Name of second sheet should be "měření".')

        df_e = pd.read_excel(xls, sheet_name=0, usecols="A:H", skiprows=2, header=None, dtype=object)
        df_m = pd.read_excel(xls, sheet_name=1, usecols="I,J", skiprows=2, header=None, dtype=object)

    # parse electrodes
    gallery_last = ""
    wall_last = ""
    height_last = ""
    for i in range(df_e.shape[0]):
        # id
        v = df_e[0][i]
        if type(v) is not int:
            continue
        e_id = v

        # gallery
        v = df_e[1][i]
        if type(v) is str and v != "":
            gallery_last = v

        # wall
        v = df_e[2][i]
        if type(v) is str and v != "":
            wall_last = v

        # height
        v = df_e[3][i]
        if type(v) is str and v != "":
            height_last = v

        eg = _add_electrode_group(ret["electrode_groups"], gallery_last, wall_last, height_last)
        _add_electrode(eg, e_id, df_e[4][i], df_e[5][i], df_e[6][i], df_e[7][i])

    # parse measurements
    for i in range(df_m.shape[0]):
        # number
        v = df_m[0][i]
        if type(v) is not str or v == "":
            continue
        number = v

        # date
        v = df_m[1][i]
        if type(v) is not pd.Timestamp:
            continue
        date = v.to_pydatetime()

        ret["measurements"].append(Measurement(number=number, date=date))

    return ret


def _add_electrode_group(electrode_groups, gallery, wall, height):
    eg = None
    for item in electrode_groups:
        if item.gallery == gallery and item.wall == wall and item.height == height:
            eg = item
            break
    if eg is None:
        eg = ElectrodeGroup(gallery=gallery, wall=wall, height=height)
        electrode_groups.append(eg)
    return eg


def _add_electrode(group, id, offset, x, y, z):
    def conv(v):
        if type(v) is float:
            if math.isnan(v):
                return None
            else:
                return v
        elif type(v) is int:
            return float(v)
        return None

    offset = conv(offset)
    x = conv(x)
    y = conv(y)
    z = conv(z)

    if (offset is None) or (x is None) or (y is None) or (z is None):
        return

    group.electrodes.append(Electrode(id=id, offset=offset, x=x, y=y, z=z))
