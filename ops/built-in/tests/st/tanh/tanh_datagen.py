# -*- coding: utf-8 -*-
import numpy as np
from dataFormat import dumpData
import mathUtil
import sys


def tanh(name, shape, src_type="fp16", miu=1, sigma=1):
    sys.stdout.write("Info: writing input for %s...\n" % name)

    if src_type == "fp16" or src_type == "float16":
        s_type = np.float16
    elif src_type == "fp32" or src_type == "float32":
        s_type = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % src_type)
    input_arr = mathUtil.randomGaussian(shape,
                                       miu=miu, sigma=sigma).astype(s_type)

    shape_str = ""
    for dim in shape:
        shape_str += str(dim) + "_"
    case_name = shape_str[0:len(shape_str) - 1] + "_" + src_type

    path = "./../data/" + name + "/" + case_name

    dumpData(input_arr, name + "_input_" + case_name + ".txt", fmt="float",
             data_type=src_type, path=path)
    dumpData(input_arr, name + "_input_" + case_name + ".data", fmt="binary",
             data_type=src_type, path=path)

    sys.stdout.write("Info: writing input for %s done!!!\n" % name)

    if src_type == "fp16" or src_type == "float16":
        input_arr = input_arr.astype(np.float32)

    output_arr = np.tanh(input_arr).astype(np.float32)

    if src_type == "fp16" or src_type == "float16":
        output_arr = output_arr.astype(np.float16)

    dumpData(output_arr, name + "_output_" + case_name + ".txt",
             fmt="float", data_type=src_type, path=path)
    dumpData(output_arr, name + "_output_" + case_name + ".data",
             fmt="binary", data_type=src_type, path=path)
    sys.stdout.write("Info: writing output for %s done!!!\n" % name)


def gen_tanh_data():
    tanh("tanh", (3, 3, 5, 6, 7), src_type="float16", miu=0, sigma=1)
    tanh("tanh", (3, 3, 5, 6, 7), src_type="float32", miu=0, sigma=1)
    tanh("tanh", (1, 1), src_type="float16", miu=0, sigma=1)
    tanh("tanh", (1, 1), src_type="float32", miu=0, sigma=1)


if __name__ == '__main__':
    gen_tanh_data()
