# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("dynamic_gru_v2_hidden")

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def matrix_to_zZ(matrix, shape, dtype):  # m , k
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if (h == 1):
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w):
                    tmp[idx] = matrix[batch][0][idx]
                    idx = idx + 1
        else:
            for j in range(0, w):
                tmp[idx] = matrix[0][idx]
                idx = idx + 1
    else:
        if len(shape) > 2:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h // 16):
                    for j in range(0, w // 16):
                        for ii in range(0, 16):
                            for jj in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
        else:
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for ii in range(0, 16):
                        for jj in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp

def matrix_to_nZ(matrix, shape, dtype):  # k,n
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if (w == 1):
        if len(shape)>2:
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h):
                    tmp[idx] = matrix[batch][idx][0]
                    idx = idx + 1
        else:
            for i in range(0, h // 16):
                tmp[idx] = matrix[idx][0]
                idx = idx + 1
    else:
        if len(shape)>2:
            for batch in range(0, np.prod(shape[:-2])):
                for i in range(0, h // 16):
                    for j in range(0, w // 16):
                        for jj in range(0, 16):
                            for ii in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
        else:
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for jj in range(0, 16):
                        for ii in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    return tmp

def matrix_to_zN(matrix, shape, dtype):  # m, n
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0
    if len(shape) > 2:
        if (h == 1):
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w):
                    tmp[idx] = matrix[batch][0][idx]
                    idx = idx + 1
        elif (w == 1):
            for batch in range(np.prod(shape[:-2])):
                for i in range(0, h):
                    tmp[idx] = matrix[batch][idx][0]
                    idx = idx + 1
        else:
            for batch in range(np.prod(shape[:-2])):
                for j in range(0, w // 16):
                    for i in range(0, h // 16):
                        for ii in range(0, 16):
                            for jj in range(0, 16):
                                tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                                idx = idx + 1
    else:
        if (h == 1):
            for j in range(0, w):
                tmp[idx] = matrix[0][idx]
                idx = idx + 1
        elif (w == 1):
            for i in range(0, h):
                tmp[idx] = matrix[idx][0]
                idx = idx + 1
        else:
            for j in range(0, w // 16):
                for i in range(0, h // 16):
                    for ii in range(0, 16):
                        for jj in range(0, 16):
                            tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
    tmp = tmp.reshape(shape[:-2] + (w // 16, h // 16, 16, 16))
    return tmp

def maxtrix_zN_reverse(matrix, shape, dtype):
    idx = 0
    j_outer,i_outer,i_inner,j_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner
    matrix_data = matrix.reshape(np.prod(matrix.shape))

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        # print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for j in range(0, j_outer):
                for i in range(0, i_outer):
                    for ii in range(0, i_inner):
                        for jj in range(0, j_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix_data[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for j in range(0, j_outer):
            for i in range(0, i_outer):
                for ii in range(0, i_inner):
                    for jj in range(0, j_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix_data[idx]
                        idx = idx + 1
        # print((h,w))

    return tmp


    idx = 0
    if len(shape)==2:
        h = shape[0]*16
        tmp = np.zeros((h,1), dtype=dtype)
        for i in range(0, h // 16):
            tmp[idx][0]= matrix[idx]
            idx = idx + 1
    if len(shape)==3:
        batch = shape[0]
        h = shape[1]*16
        tmp = np.zeros((batch,h,1), dtype=dtype)
        for batch in range(np.prod(shape[:-2])):
            for i in range(0, h):
                tmp[batch][i][0] = matrix[idx]
                idx = idx + 1
    elif len(shape)==4:
        h,w = shape[0]*16,shape[1]*16
        tmp = np.zeros((h,w), dtype=dtype)
        for i in range(0, h // 16):
            for j in range(0, w // 16):
                for jj in range(0, 16):
                    for ii in range(0, 16):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
    elif len(shape)==5:
        batch = shape[0]
        h,w = shape[1]*16,shape[2]*16
        tmp = np.zeros((batch,h,w), dtype=dtype)
        for batch in range(0, np.prod(shape[:-4])):
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for jj in range(0, 16):
                        for ii in range(0, 16):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    return tmp

def maxtrix_nZ_reverse(matrix, shape, dtype):

    idx = 0
    i_outer,j_outer,j_inner,i_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner
    matrix_data = matrix.reshape(np.prod(matrix.shape))

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        # print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for i in range(0, i_outer):
                for j in range(0, j_outer):
                    for jj in range(0, j_inner):
                        for ii in range(0, i_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix_data[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for i in range(0, i_outer):
            for j in range(0, j_outer):
                for jj in range(0, j_inner):
                    for ii in range(0, i_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix_data[idx]
                        idx = idx + 1
        # print((h,w))

    return tmp

def calc_expect_func(x_weight_input, weight_hidden, bias_hidden, seq_length, init_h,
                     y, output_h, update, reset, new, hidden_new,
                     direction= "UNIDIRECTIONAL", cell_depth=1, keep_prob=1.0,
                     cell_clip=-1.0, num_proj=0, time_major=False, activation="tanh",
                     gate_order="zrh", reset_after=True, is_training=True, kernel_name="dynamic_gru_v2"):
    is_gate_output = update is not None
    is_global_init = init_h is not None
    shape_x_weight_input = x_weight_input.get("shape")
    shape_output = y.get("shape")
    t_size = shape_x_weight_input[0]
    m_size = shape_x_weight_input[2]
    hidden_size = shape_output[1]
    bias_dtype = y.get("dtype")

    if is_global_init:
        h_data_init = init_h.get("value")
        h_data_init = maxtrix_zN_reverse(h_data_init, h_data_init.shape, np.float32)
        h_data_init = h_data_init.reshape([16*m_size, 16*hidden_size])
    else:
        h_data_init = np.zeros([16*m_size, 16*hidden_size]).astype('float32')

    cache_data_t = x_weight_input.get("value")
    cache_data_t = maxtrix_zN_reverse(cache_data_t, cache_data_t.shape, np.float32)
    cache_data_t = np.split(cache_data_t, t_size, axis = 0)
    # print("cache_data_t[0].shape="+str(cache_data_t[0].shape))

    w2_data = weight_hidden.get("value")
    w2_data = maxtrix_nZ_reverse(w2_data, w2_data.shape, np.float32)
    b_data_2 = w2_data

    if bias_hidden is not None:
        bias_num_2 = bias_hidden.get("value")

    h_pre = h_data_init.astype('float32')

    for i in range(t_size):
        cache_data = cache_data_t[i]
        cache_data = cache_data.reshape((16*m_size, 3*16*hidden_size))
        res = np.split(cache_data, 3, axis=1)

        h_pre_fp16 = h_pre.astype('float16')
        res_2 = np.matmul(h_pre_fp16, b_data_2).astype('float32')
        if bias_hidden is not None:
            res_2 = res_2 + bias_num_2
        res_2 = np.split(res_2, 3, axis=1)
        if gate_order == "zrh":
            res_i = res[0]
            res_r = res[1]
            res_n = res[2]
            res_2_i = res_2[0]
            res_2_r = res_2[1]
            res_2_n = res_2[2]
        else:
            res_r = res[0]
            res_i = res[1]
            res_n = res[2]
            res_2_r = res_2[0]
            res_2_i = res_2[1]
            res_2_n = res_2[2]
        res_r = sigmoid(res_r + res_2_r)
        res_i = sigmoid(res_i + res_2_i)
        res_n = np.tanh(res_n + res_r * res_2_n)

        if is_gate_output:
            res_r_output = res_r.reshape((1, res_r.shape[0], res_r.shape[1]))
            if i == 0:
                r_output = res_r_output
            else:
                r_output = np.concatenate((r_output,res_r_output), axis=0)

            res_i_output = res_i.reshape((1, res_i.shape[0], res_i.shape[1]))
            if i == 0:
                i_output = res_i_output
            else:
                i_output = np.concatenate((i_output,res_i_output), axis=0)

            res_n_output = res_n.reshape((1, res_n.shape[0], res_n.shape[1]))
            if i == 0:
                n_output = res_n_output
            else:
                n_output = np.concatenate((n_output,res_n_output), axis=0)

            res_hn_output = res_2_n.reshape((1, res_2_n.shape[0], res_2_n.shape[1]))
            if i == 0:
                hn_output = res_hn_output
            else:
                hn_output = np.concatenate((hn_output,res_hn_output), axis=0)

        # (1 − i) * h^ + i * h_t-1
        res_h = (1 - res_i) * res_n + res_i * h_pre

        h_pre = res_h

        res_h_output = res_h.reshape((1, res_h.shape[0], res_h.shape[1]))
        if i == 0:
            output = res_h_output
        else:
            output = np.concatenate((output,res_h_output), axis = 0)

    bias_dtype_obj = np.float32 if bias_dtype == "float32" else np.float16
    output = matrix_to_zN(output, output.shape, bias_dtype_obj)
    ret = [output, output]
    if is_gate_output:
        r_output = matrix_to_zN(r_output, r_output.shape, bias_dtype_obj)
        i_output = matrix_to_zN(i_output, i_output.shape, bias_dtype_obj)
        n_output = matrix_to_zN(n_output, n_output.shape, bias_dtype_obj)
        hn_output = matrix_to_zN(hn_output, hn_output.shape, bias_dtype_obj)
        ret.extend([i_output, r_output, n_output, hn_output])
    return ret

def get_params(t_size, m_size, hidden_size, bias_dtype, data_range=[0.01, 0.1], init_data_range=[0.01, 0.1], seq_dtype='int32'):
    dtype = 'float16'
    shape_w2 = [hidden_size, 3*hidden_size, 16, 16]
    if seq_dtype == 'int32':
        shape_seq = [m_size * 16,]
    else:
        shape_seq = [t_size, hidden_size, m_size, 16, 16]
    shape_c = [t_size, hidden_size, m_size, 16, 16]
    shape_c_1 = [1, hidden_size, m_size, 16, 16]
    shape_bias = [3* hidden_size*16,]
    shape_x_weight_input = [t_size, 3*hidden_size, m_size, 16, 16]

    cache = {"shape":shape_x_weight_input, "dtype":'float32', "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    w2 = {"shape":shape_w2, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    b2 = {"shape":shape_bias, "dtype":bias_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    seq = {"shape":shape_seq, "dtype":seq_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    s_init_h_gm = {"shape":shape_c_1, "dtype":bias_dtype, "param_type": "input", "value_range": init_data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    output_y = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    output_h = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    i = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    r = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    n = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    hn = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_c}
    return cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn


# for cov
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float32')
ut_case.add_case("all", {
    "params": [cache, w2, b2, seq, None, output_y, output_h, None, None, None, None]
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None]
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn]
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, s_init_h_gm, output_y, output_h, i, r, n, hn]
})
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float16')
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, True, "tanh", "rzh"]
})
ut_case.add_case("all", {
    "params": [cache, w2, None, None, s_init_h_gm, output_y, output_h, i, r, n, hn]
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "aa", 1, 1.0, -1.0, 0, True, "tanh"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 11, 1.0, -1.0, 0, True, "tanh"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 11.0, -1.0, 0, True, "tanh"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -11.0, 0, True, "tanh"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 10, True, "tanh"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, False, "tanh"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, True, "sin"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, True, "tanh", "sdfs"],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, None, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, True, "tanh", "zrh", False],
    "expect": RuntimeError
})
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float16')
r["shape"] = [0]
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, r, i, n, hn],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, i, r, hn],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, i, i, r],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, r, output_h, i, i, n, hn],
    "expect": RuntimeError
})
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float16')
b2["shape"] = [0]
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float16')
w2["shape"] = [0,0,0]
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float16')
w2["shape"][1] = 1
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})
cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float16')
r["dtype"] = 'float32'
ut_case.add_case("all", {
    "params": [cache, w2, b2, None, None, output_y, output_h, i, r, n, hn],
    "expect": RuntimeError
})

cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, hidden_size=32, bias_dtype='float32', seq_dtype='float16')
ut_case.add_case("all", {
    "params": [cache, w2, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn]
})
ut_case.add_case("all", {
    "params": [cache, w2, b2, seq, None, output_y, output_h, None, None, None, None]
})