"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""

import numpy as np
import sys
from dataFormat import *
from functools import reduce

def ifmr(name, shape_x, src_type):
    sys.stdout.write("Info: writing input for %s...\n"%name)
    shape_str = ""
    for dim in shape_x:
        shape_str += str(dim) + "_"
    feature_name = shape_str + src_type

    data = np.random.uniform(-2, 2, shape_x).astype(src_type)
    dumpData(data, name + "_input1_" + feature_name + ".data",
             fmt="binary", data_type=src_type,
             path="../data/" + name + "/" + feature_name)

    max_percentile = 0.9
    min_percentile = 0.9
    search_range = [0.7, 1.3]
    search_step = 0.01
    with_offset = True
    bins_num = 512

    data_num = reduce(lambda x, y: x * y, shape_x)
    data_max = np.max(data)
    data_min = np.min(data)
    data_max = np.array([data_max], dtype=src_type)
    data_min = np.array([data_min], dtype=src_type)
    bins, threshold = np.histogram(data, bins_num)
    cumsum = np.cumsum(bins).astype(np.int32)
    cdf = cumsum / data_num
    max_index = np.where(cdf > max_percentile, 0, 1).sum()
    min_index = np.where(cdf > 1 - min_percentile, 0, 1).sum()
    max_init = max_index / bins_num * (data_max - data_min) + data_min
    min_init = min_index / bins_num * (data_max - data_min) + data_min
    step = np.arange(search_range[0], search_range[1], search_step)
    max_list = max_init * step
    min_list = min_init * np.ones(step.shape)
    scale = (max_list - min_list) / 255
    offset = np.round(min_list / scale)
    offset = -(offset + 128)
    data_list = data.flatten()
    loss_list = np.zeros(len(step))
    for i in range(len(step)):
        loss = np.sum(np.square(np.round(data_list / scale[i]) * scale[i] - data_list))
        loss_list[i] = loss

    index = np.unravel_index(np.argmin(loss_list), loss_list.shape)

    output_scale = scale[index]
    output_offset = offset[index]
    output = np.array([output_scale, output_offset], dtype=src_type)


    dumpData(output, name + "_output_" + feature_name + ".data",
             fmt="binary", data_type=src_type,
             path="../data/" + name + "/" + feature_name)
    sys.stdout.write("Info: writing output for %s done!!!\n"%name)


#def get_shape(shape_list, dims, limit=1024):
#    shape = np.random.randint(1, limit)
#    shape_list.append(shape)
#    dims = dims - 1
#    if dims <= 0:
#        return
#    else:
#        return get_shape(shape_list, dims, limit)





def gen_ifmr_data(isBBIT=False):
    ifmr("ifmr", (1, 1), "float32")
    ifmr("ifmr", (16, 32), "float32")

if __name__ =="__main__":
    gen_ifmr_data()
