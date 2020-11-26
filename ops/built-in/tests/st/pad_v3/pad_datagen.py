# -*- coding: utf-8 -*-
"""
This is a gelu_grad data file.
"""
import numpy as NP
import sys
from dataFormat import *


def gen_pad_data(isBBIT = False):
    # def _gen_pad_data(name, input_A_shape, input_B_shape, input_C_shape,case_name,  dtype="float16"):
    def _gen_pad_data(shape, paddings, mode, constant_values, src_type="fp16"):
        name = 'pad'
        sys.stdout.write("Info: writing input for %s...\n" % name)

        if src_type == "fp16" or src_type == "float16":
            s_type = NP.float16
        elif src_type == "fp32" or src_type == "float32":
            s_type = NP.float32
        elif src_type == "int32":
            s_type = NP.int32
        else:
            raise RuntimeError("unsupported dtype:%s " % src_type)

        inputArr = NP.random.uniform(-10, 10, shape).astype(s_type)

        shape_str = str(shape) + '_' + src_type + '_' + str(paddings) + '_' + mode + '_' + str(constant_values)
        shape_str = shape_str.replace(',', '_')
        shape_str = shape_str.replace(' ', '')
        shape_str = shape_str.replace('[', '').replace(']', '')
        case_name = shape_str.replace('(', '').replace(')', '')

        path = "./../data/" + name + "/" + case_name
        dumpData(inputArr, name + "_input_" + case_name + ".data",
                 fmt="binary", data_type=src_type, path=path)
        sys.stdout.write("Info: writing input for %s done!!!\n" % name)

        outputArr = NP.pad(inputArr, paddings, mode=mode, constant_values=constant_values)


        output_shape = [0 for _ in range(len(shape))]
        for i in range(len(shape)):
            output_shape[i]=paddings[i][0]+paddings[i][1]+shape[i]

        shape_str = str(output_shape) + '_' + src_type + '_' + str(paddings) + '_' + mode + '_' + str(constant_values)
        shape_str = shape_str.replace(',', '_')
        shape_str = shape_str.replace(' ', '')
        shape_str = shape_str.replace('[', '').replace(']', '')
        case_name = shape_str.replace('(', '').replace(')', '')
        dumpData(outputArr, name + "_output_" + case_name + ".data",
                 fmt="binary", data_type=src_type, path=path)
        sys.stdout.write("Info: writing output for %s done!!!\n" % name)


    _gen_pad_data([1], [[1, 1]], "constant", 0, "int32")
    _gen_pad_data([17], [[1, 17]], "constant", 0, "float16")
    _gen_pad_data([3, 17], [[3, 3], [0, 17]], "constant", 0, "int8")
    _gen_pad_data([3, 1024 + 3], [[3, 0], [0, 0]], "constant", 0, "uint8")
    _gen_pad_data([2, 2, 16], [[0, 0], [1, 3], [0, 0]], "constant", 0, "float16")
    _gen_pad_data([2, 2, 9], [[7, 7], [7, 7], [7, 7]], "constant", 0, "float32")
    _gen_pad_data([2, 2, 1027], [[0, 0], [0, 16], [0, 0]], "constant", 0, "float32")
    _gen_pad_data([2, 2, 1027], [[0, 0], [0, 7], [0, 7]], "constant", 0, "float16")
    _gen_pad_data([2, 2, 9], [[0, 0], [0, 16], [0, 0]], "constant", 0, "float32")
    _gen_pad_data([2, 2, 63], [[0, 0], [0, 7], [0, 7]], "constant", 0, "float32")
    _gen_pad_data([2, 2, 63], [[0, 0], [0, 16], [0, 0]], "constant", 0, "float32")
    _gen_pad_data([3, 17], [[3, 3], [0, 3]], "constant", 0, "float32")

    if isBBIT:
        _gen_pad_data([2, 2, 32640], [[0, 0], [0, 1], [0, 0]], "constant", 0, "float32")
        _gen_pad_data([32, 128, 1024], [[0, 0], [0, 384], [0, 0]], "constant", 0, "float16")
        _gen_pad_data([3, 3, 128 * 1024 + 3], [[0, 0], [0, 3], [0, 0]], "constant", 0, "float32")
        _gen_pad_data([3, 128 * 1024 + 3], [[3, 0], [0, 0]], "constant", 0, "float16")
        _gen_pad_data([128 * 1024 + 3], [[128 * 1024 + 3, 128 * 1024 + 3]], "constant", 0, "float16")
        _gen_pad_data([3, 128 * 1024 + 3], [[3, 3], [0, 0]], "constant", 0, "float32")
        _gen_pad_data([32, 128, 1024], [[0, 0], [0, 384], [0, 0]], "constant", 0, "float32")
        _gen_pad_data([32, 2, 128 * 1024], [[0, 0], [2, 0], [0, 0]], "constant", 0, "float16")
        _gen_pad_data([2, 2, 16310], [[0, 0], [1, 0], [9, 1]], "constant", 0, "float32")

