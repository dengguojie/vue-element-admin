# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:12:57 2020

@author: l00557915
"""
import math
from dformat import *
import numpy as np
import os


def expand_4_tensorlist(tensorlist):
    ex_tensorlist = []
    for tensor in tensorlist:
        ex_tensorlist.append(np.expand_dims(tensor, 0))
    return ex_tensorlist


def expand_4_tensorlist_and_concat(tensorlist, ex_tensorlist):
    index = 0
    for tensor, xtensor in zip(tensorlist, ex_tensorlist):
        xtensor = np.concatenate((xtensor, np.expand_dims(tensor, 0)), axis=0)
        ex_tensorlist[index] = xtensor
        index = index + 1
    return ex_tensorlist


def expand_4_tensorlist_and_pushforward(tensorlist,ex_tensorlist):
    index = 0
    for tensor, xtensor in zip(tensorlist, ex_tensorlist):
        xtensor = np.concatenate((np.expand_dims(tensor, 0), xtensor), axis=0)
        ex_tensorlist[index] = xtensor
        index = index + 1
    return ex_tensorlist


def reorder_data(actual_data, axes):
    r, i, n = np.split(actual_data, 3, axis=axes)
    return np.concatenate((i, r, n), axis=axes)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def single_gru_cell(xt, h_prev, weight, bias_x, bias_h):
    input_dim = xt.shape[1]
    hidden_dim = h_prev.shape[1]

    weight_x, weight_h = np.split(weight, [input_dim], axis=0)

    matmul_x = np.matmul(xt, weight_x) + bias_x
    matmul_x_i, matmul_x_r, matmul_x_n = np.split(matmul_x, 3, axis=1)

    matmul_h = np.matmul(h_prev, weight_h) + bias_h
    matmul_h_i, matmul_h_r, matmul_h_n = np.split(matmul_h, 3, axis=1)

    it = sigmoid(matmul_x_i + matmul_h_i)
    rt = sigmoid(matmul_x_r + matmul_h_r)
    nt = np.tanh(matmul_x_n + rt * matmul_h_n)

    ht = (h_prev - nt) * it + nt
    return rt, it, nt, ht, matmul_h_n


def gru_grad_cell(xt, h_prev, weight, dht, it, rt, nt, nt_mid, dyt):
    dht = dht + dyt
    batch = xt.shape[0]
    input_dim = xt.shape[1]
    weight_x, weight_h = np.split(weight, [input_dim], axis=0)
    dit = dht * (h_prev - nt) * (1 - it) * it
    dnt_x = dht * (1 - it) * (1 - np.power(nt, 2))
    dnt_h = dnt_x * rt
    drt = dnt_x * nt_mid * rt * (1 - rt)

    dgate_h = np.concatenate((dit, drt, dnt_h), axis=1)
    # dh_prev = np.matmul(dgate_h, weight_h.T) + dht * it
    dh_prev = dht * it
    # dh_prev = dnt_x
    # print(dgate_h)
    ########################
    dgate_x = np.concatenate((dit, drt, dnt_x), axis=1)
    dxt = np.matmul(dgate_x, weight_x.T)

    dwt_x = np.matmul(xt.T, dgate_x)
    dwt_h = np.matmul(h_prev.T, dgate_h)
    dwt = np.concatenate((dwt_x, dwt_h), axis=0)

    dbt_x = np.matmul(np.ones((1, batch)), dgate_x)
    dbt_h = np.matmul(np.ones((1, batch)), dgate_h)

    # dbt与dwt还需要沿t轴进行累加求和
    return dxt, dh_prev, dwt, dbt_x, dbt_h, dnt_x, dgate_h


def dynamic_gru(t_step, batch, input_dim, hidden_dim, src_type, ranges=(0, 1)):
    # 初始化左矩阵的值，X H
    xh0 = np.random.uniform(ranges[0], ranges[1], (batch, (input_dim + hidden_dim))).astype(src_type)
    # 当前时刻的x1和上一层的输入h0,c0
    x1, h0 = np.split(xh0, [input_dim], axis=1)
    # 初始化右矩阵的值,W
    weight = np.random.uniform(ranges[1], ranges[1], ((input_dim + hidden_dim),
                                                      hidden_dim * 3)).astype(src_type)/(input_dim + hidden_dim)
    # bias ，3*hidden
    bias_x = np.random.uniform(ranges[0], ranges[1], (hidden_dim * 3)).astype(src_type)
    bias_h = np.random.uniform(ranges[0], ranges[1], (hidden_dim * 3)).astype(src_type)

    # 每一次cell的输入
    x_1 = x1
    h_0 = h0
    for i in range(t_step):
        # 单步的正向cell运算 input_gate, reset_gate, new_gate, output, new_gate_mid
        r_1, i_1, n_1, h_1, n_hh = single_gru_cell(x_1, h_0, weight, bias_x, bias_h)

        # 结果累计
        if i == 0:
            input_x, output_h, output_i, output_r, output_n, output_n_mid = expand_4_tensorlist(
                [x_1, h_1, i_1, r_1, n_1, n_hh])
        else:
            input_x, output_h, output_i, output_r, output_n, output_n_mid = expand_4_tensorlist_and_concat(
                [x_1, h_1, i_1, r_1, n_1, n_hh], [input_x, output_h, output_i, output_r, output_n, output_n_mid])

        x_1 = np.random.uniform(ranges[0], ranges[1], (batch, input_dim)).astype(src_type)
        h_0 = h_1
    return input_x, output_h, output_i, output_r, output_n, output_n_mid, weight, bias_x, bias_h, h0


def dynamic_gru_grad(t_step, batch, input_dim, hidden_dim, src_type, ranges=(0, 1), gate_order="zrh",
                     kenel_name="gru_grad"):
    input_x, output_h, output_i, output_r, output_n, output_n_mid, weight, bias_x, bias_h, h0 = dynamic_gru(t_step,
                                                                                                            batch,
                                                                                                            input_dim,
                                                                                                            hidden_dim,
                                                                                                            src_type,
                                                                                                            ranges)

    ori_dh2 = np.zeros_like(h0).astype(src_type)
    ori_loss = np.random.uniform(ranges[0], ranges[1], (t_step, batch, hidden_dim)).astype(src_type)
    dh2 = ori_dh2

    for i in range(t_step - 1, -1, -1):
        h1 = output_h[i - 1] if i != 0 else h0

        dy = ori_loss[i]
        dx2, dh1, dw2, db_x, db_h, dnt_x, dgate_h = gru_grad_cell(input_x[i], h1, weight, dh2, output_i[i], output_r[i],
                                                                  output_n[i], output_n_mid[i], dy)
        if i == t_step - 1:
             ouput_dx2, ouput_dh1, ouput_dw2, ouput_db_x, ouput_db_h, ouput_dnt_x, ouput_dgate_h = expand_4_tensorlist(
                [dx2, dh1, dw2, db_x, db_h, dnt_x, dgate_h])
        else:
            # ouput_dx2, ouput_dh1, ouput_dw2, ouput_db_x, ouput_db_h, ouput_dnt_x, ouput_dgate_h = expand_4_tensorlist_and_concat(
            ouput_dx2, ouput_dh1, ouput_dw2, ouput_db_x, ouput_db_h, ouput_dnt_x, ouput_dgate_h = expand_4_tensorlist_and_pushforward(
                [dx2, dh1, dw2, db_x, db_h, dnt_x, dgate_h],
                [ouput_dx2, ouput_dh1, ouput_dw2, ouput_db_x, ouput_db_h, ouput_dnt_x, ouput_dgate_h])

        dh2 = dh1

    output_dw = np.sum(ouput_dw2, axis=0)
    ouput_db_x = np.sum(ouput_db_x, axis=0)
    ouput_db_h = np.sum(ouput_db_h, axis=0)

    batch_size = batch
    input_size = input_dim
    hidden_size = hidden_dim
    return input_x, output_h, output_i, output_r, output_n, output_n_mid, weight, bias_x, bias_h,\
           h0, ouput_dx2, ouput_dh1, output_dw, ouput_db_x, ouput_db_h, ori_loss, ori_dh2, ouput_dnt_x, ouput_dgate_h


def gruv2_grad_data(t_step, batch, input_dim, hidden_dim, src_type, ranges=(0, 1), gate_order="zrh",
                    kenel_name="gru_grad"):
    return dynamic_gru_grad(t_step, batch, input_dim, hidden_dim, src_type, ranges, gate_order, kenel_name)


def convert_fp16nz(tensor, dtype="float16"):
    tensor_Nz, shape_nz = matrix_to_zN(tensor, tensor.shape, dtype) if len(tensor.shape) > 1 else (tensor, 0)
    if shape_nz is 0:
        return tensor_Nz
    else:
        return tensor_Nz.reshape(shape_nz)


def gruv2_hidden_grad_data(t_step, batch, input_dim, hidden_dim, src_type, ranges=(0, 1), t_state=0, gate_order="zrh",
                           kenel_name="gru_grad"):
    input_x, output_h, output_i, output_r, output_n, output_n_mid, weight, bias_x, bias_h, h0, \
    ouput_dx2, ouput_dh1, output_dw, ouput_db_x, ouput_db_h, ori_loss, ori_dh2, ouput_dnt_x, ouput_dgate_h = dynamic_gru_grad(
        t_step, batch, input_dim, hidden_dim, src_type, ranges, gate_order, kenel_name)

    if gate_order == "rzh":
        weight = reorder_data(weight, 1)
        bias_x = reorder_data(bias_x, 0)
        bias_h = reorder_data(bias_h, 0)
        output_dw = reorder_data(output_dw, 1)
        ouput_db_x = reorder_data(ouput_db_x, 1)
        ouput_db_h = reorder_data(ouput_db_h, 1)
        ouput_dgate_h = reorder_data(ouput_dgate_h, 1)
    weight_ih, weight_hh = np.split(weight.T, [input_dim], axis=1)
    return {
        # "weight_hidden": convert_fp16nz(weight_hh),
            "init_h": convert_fp16nz(h0, src_type),
            "h": convert_fp16nz(output_h, src_type),
            "dy": convert_fp16nz(ori_loss, src_type),
            "dh": convert_fp16nz(ori_dh2, src_type),
            "update": convert_fp16nz(output_i, src_type),
            "reset": convert_fp16nz(output_r, src_type),
            "new": convert_fp16nz(output_n, src_type),
            "hidden_new": convert_fp16nz(output_n_mid, src_type),
            "dh_pre_t": convert_fp16nz(output_n_mid, src_type), # TODO
            "dh_prev": convert_fp16nz(ouput_dh1[0], src_type),
            "dgate_h": convert_fp16nz(ouput_dgate_h, src_type),
            "dnt_x": convert_fp16nz(ouput_dnt_x, src_type)}


'''
if __name__ == "__main__":
    t_size = 4
    batch_size = 16
    input_size = 48
    hidden_size = 32
    layer_num = 1
    input_x,output_h,output_i,output_r,output_n,output_n_mid,weight,bias_x,bias_h,h0, \
    ouput_dx2, ouput_dh1, output_dw, ouput_db_x, ouput_db_h, ori_loss, ori_dh2,ouput_dnt_x,ouput_dgate_h = gruv2_hidden_grad_data(t_size, batch_size, input_size, hidden_size, 'float64', (-1, 1))
'''
