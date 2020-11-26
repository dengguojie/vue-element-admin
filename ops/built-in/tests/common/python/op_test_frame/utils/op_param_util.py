#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import random
import json
import copy
import numpy as np


def build_op_param(params_info):
    """
    build op_param dict by info list/tuple
    tuple info [dtype, shape, format, ori_shape, ori_format]
    dict info {"dtype": dtype, "shape": shape, "format": format, "ori_shape": ori_shape, "format": ori_format}

    :param params_info: [dtype, shape, format, ori_shape, ori_format], like ["float16", [3,4,5,6], "ND", [3,4,5,6], "ND"]
    :return: op param
    """
    if len(params_info) == 3:
        params_info += params_info[1:3]
    return {
        "dtype": params_info[0],
        "shape": params_info[1],
        "format": params_info[2],
        "ori_shape": params_info[3],
        "ori_format": params_info[4]
    }


def broadcast_shape(shape1, shape2):
    max_len = max(len(shape1), len(shape2))
    sub_len = abs(len(shape1) - len(shape2))
    if len(shape1) < len(shape2):
        tmp = shape1
        shape1 = shape2
        shape2 = tmp
    b_shape = []
    for idx in range(max_len):
        if idx >= sub_len:
            dim = max(shape1[-max_len + idx], shape2[-max_len + idx])
        else:
            dim = shape1[idx]
        b_shape.append(dim)
    return b_shape


def trans_shape(shape, ori_format, cur_format):
    if cur_format == "NC1HWC0":
        if ori_format in ["NCHW", "NHWC", "HWCN"]:
            s_f = dict(zip(list(ori_format), shape))
            cur_shape = [s_f["N"], (s_f["C"] + 15) // 16, s_f["H"], s_f["W"], 16]
            return cur_shape
        else:
            cur_shape = [1, (shape[0] + 15) // 16, 1, 1, 16]
            return cur_shape
    if cur_format == "FRACTAL_NZ":
        cur_shape = list(shape[:-2]) + [(shape[-1] + 15) // 16, (shape[-2] + 15) // 16, 16, 16]
        return cur_shape
    if cur_format == "FRACTAL_Z":  # C1HW, N, Co, C0
        s_f = dict(zip(list(ori_format), shape))
        cur_shape = [(s_f["C"] + 15) // 16 * s_f["H"] * s_f["W"], (s_f["N"] + 15) // 16, 16, 16]
        return cur_shape
    if cur_format == "C1HWNCoC0":
        s_f = dict(zip(list(ori_format), shape))
        cur_shape = [(s_f["C"] + 15) // 16, s_f["H"], s_f["W"], (s_f["N"] + 15) // 16, 16, 16]
        return cur_shape
    if cur_format == "NC1HWC0_C04":
        s_f = dict(zip(list(ori_format), shape))
        cur_shape = [s_f["N"], (s_f["C"] + 3) // 4, s_f["H"], s_f["W"], 4]
        return cur_shape
    if cur_format == "ND":
        return shape


def change_cur_format(op_params, format_list):
    for idx, cur_fm in enumerate(format_list):
        op_params[idx]["format"] = cur_fm
        op_params[idx]["shape"] = trans_shape(op_params[idx]["ori_shape"],
                                              op_params[idx]["ori_format"], cur_fm)
    return op_params


def gen_all_format_params(format_res, op_params):
    format_info = json.loads(format_res)
    input_list = []
    output_list = []
    for key in format_info.keys():
        if str(key).startswith("input"):
            input_list.append(key)
        if str(key).startswith("output"):
            output_list.append(key)

    input_outputs = []
    for input_name in input_list:
        f_l = format_info[input_name]["format"].split(",")
        d_l = format_info[input_name]["dtype"].split(",")
        input_outputs.append(zip(d_l, f_l))
    for ouput_name in output_list:
        f_l = format_info[ouput_name]["format"].split(",")
        d_l = format_info[ouput_name]["dtype"].split(",")
        input_outputs.append(zip(d_l, f_l))

    f_d_table = zip(*input_outputs)

    res = []
    for col in f_d_table:
        match_flag = True
        new_op_param = copy.deepcopy(op_params)
        for idx, row in enumerate(col):
            if row[0] != new_op_param[idx]["dtype"]:
                match_flag = False
        if match_flag:
            f_l = []
            for row in col:
                f_l.append(row[1])
            new_op_param = change_cur_format(new_op_param, f_l)
            res.append(new_op_param)
    return res


def cartesian_set_format_dtype(name_list, dtype_list, format_list):
    d_l = len(dtype_list[0])
    f_l = len(format_list[0])
    new_dtype_list = copy.deepcopy(dtype_list)
    new_format_list = copy.deepcopy(format_list)
    for idx, dt in enumerate(new_dtype_list):
        new_dtype = []
        for dtype in dt:
            new_dtype += [dtype, ] * f_l
        new_dtype_list[idx] = new_dtype

    for idx, fm in enumerate(new_format_list):
        new_format_list[idx] = new_format_list[idx] * d_l
    result = {}
    for idx, ipt_name in enumerate(name_list[0]):
        result["input" + str(idx)] = {
            "name": ipt_name,
            "dtype": ",".join(new_dtype_list[idx]),
            "format": ",".join(new_format_list[idx])
        }
    ipt_len = len(name_list[0])
    for idx, opt_name in enumerate(name_list[1]):
        result["output" + str(idx)] = {
            "name": opt_name,
            "dtype": ",".join(new_dtype_list[idx + ipt_len]),
            "format": ",".join(new_format_list[idx + ipt_len])
        }
    return result


def gen_shape(rank_range=None, size_range=None, dim_limit=None):
    if size_range is None:
        size_range = [1, 2000 - 1]
    if rank_range is None:
        rank_range = [1, 8]
    if dim_limit is None:
        dim_range = 2000000 - 1
    rank = random.randint(*rank_range)
    size = random.randint(*size_range)
    p_d = np.exp(np.log(size) / rank)
    dims = np.random.normal(loc=10, scale=3.0, size=(rank,))
    dims = np.maximum(dims, 0)
    dim_prod = np.prod(dims)
    dim_prod_norm = np.exp(np.log(dim_prod) / rank)
    prod_dims = dims / dim_prod_norm * p_d
    ceil_dims = np.ceil(prod_dims)
    maxi_dims = np.maximum(ceil_dims, 1)
    mini_dims = np.minimum(maxi_dims, dim_range)
    return [int(x) for x in list(mini_dims)]


def gen_broadcast_shape():
    def _random_broadcast_dims(shape_rank):
        cnt = random.randint(0, shape_rank)
        b_dims = []
        for idx in range(cnt):
            b_dim = random.randint(0, shape_rank - 1)
            if b_dim not in b_dims:
                b_dims.append(b_dim)
        return b_dims

    shape = gen_shape()
    rank = len(shape)
    a_brd_dims = _random_broadcast_dims(rank)
    b_brd_dims = _random_broadcast_dims(rank)
    a_shape = shape[:]
    b_shape = shape[:]
    for dim in a_brd_dims:
        if dim not in b_brd_dims:
            a_shape[dim] = 1
    for dim in b_brd_dims:
        if dim not in a_brd_dims:
            b_shape[dim] = 1

    return a_shape, b_shape


def random_dtype(dtype_list=("float16", "float32", "int32")):
    d_len = len(dtype_list)
    idx = random.randint(0, d_len - 1)
    return dtype_list[idx]
