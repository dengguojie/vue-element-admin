# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("DynamicGru", None, None)

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
        print((batch_shape,h,w),matrix.shape)
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
        print((h,w))

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
        print((batch_shape,h,w),matrix.shape)
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
        print((h,w))

    return tmp

def calc_expect_func(x_input, w1_input, b1_input, w2_input, b2_input,
                     seq_length, s_init_h_gm,
                     output_y, output_h,
                     r, i, n,
                     direction= "UNIDIRECTIONAL", cell_depth=1, keep_prob=1.0,
                     cell_clip=-1.0, num_proj=0, time_major=False, activation="tanh", is_training=True,
                     kernel_name="dynamic_rnn"):
    is_gate_output = r is not None
    is_global_init = s_init_h_gm is not None

    shape_x_input = x_input.get("shape")
    shape_w1_input = w1_input.get("shape")
    shape_b1_input = b1_input.get("shape")
    shape_w2_input = w2_input.get("shape")
    shape_b2_input = b2_input.get("shape")
    shape_output = output_y.get("shape")

    t_size = shape_x_input[0]
    m_size = shape_x_input[2]
    k_size = shape_w1_input[0]
    hidden_size = shape_output[1]
    in_x = shape_x_input[1]

    bias_dtype = b1_input.get("dtype")

    if is_global_init:
        h_data_init = s_init_h_gm.get("value")
        h_data_init = maxtrix_zN_reverse(h_data_init, h_data_init.shape, np.float32)
        h_data_init = h_data_init.reshape([16*m_size, 16*hidden_size])
    else:
        h_data_init = np.zeros([16*m_size, 16*hidden_size]).astype('float32')

    x_data_t = x_input.get("value")
    x_data_t = maxtrix_zN_reverse(x_data_t, x_data_t.shape, np.float32)
    x_data_t = np.split(x_data_t, t_size, axis = 0)
    print("x_data_t[0].shape="+str(x_data_t[0].shape))

    w1_data = w1_input.get("value")
    w1_data = maxtrix_nZ_reverse(w1_data, w1_data.shape, np.float32)
    b_data_1 = w1_data
    w2_data = w2_input.get("value")
    w2_data = maxtrix_nZ_reverse(w2_data, w2_data.shape, np.float32)
    b_data_2 = w2_data

    bias_num_1 = b1_input.get("value")
    bias_num_2 = b2_input.get("value")

    h_pre = h_data_init

    for i in range(t_size):
        x_data = x_data_t[i]
        x_data = x_data.reshape((16*m_size, 16*in_x))
        x_new = np.concatenate((x_data,h_pre),axis=1)
        x_new = x_new.astype('float16')
        b_data_1 = b_data_1.astype('float16')
        res = np.matmul(x_new, b_data_1).astype('float32')
        res = res + bias_num_1
        res = np.split(res, [16*hidden_size], axis = 1)
        res_r = res[0]
        res_i = res[1]
        res_r = sigmoid(res_r)
        res_i = sigmoid(res_i)

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

        res_r_h_pre = res_r * h_pre
        x_new = np.concatenate((x_data,res_r_h_pre),axis=1)
        x_new = x_new.astype('float16')
        b_data_2 = b_data_2.astype('float16')
        res_h = np.matmul(x_new, b_data_2).astype('float32')
        res_h = res_h + bias_num_2
        res_h = np.tanh(res_h)

        res_n = res_h
        if is_gate_output:
            res_n_output = res_n.reshape((1, res_n.shape[0], res_n.shape[1]))
            if i == 0:
                n_output = res_n_output
            else:
                n_output = np.concatenate((n_output,res_n_output), axis=0)

        # (1 − i) * h^ + i * h_t-1
        res_h = (1 - res_i) * res_h + res_i * h_pre

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
        ret.extend([r_output, i_output, n_output])
    return ret

def get_params(t_size, m_size, in_x, hidden_size, bias_dtype, data_range=[0.01, 0.1], init_data_range=[0.01, 0.1]):
    # t_size = 3
    # m_size = 2
    # in_x = 64
    # hidden_size = 32
    # bias_dtype = 'float32'
    # data_range = [0.01, 0.1]
    # init_data_range = [0.01, 0.1]

    k_size = in_x + hidden_size
    dtype = 'float16'
    shape_w1 = [k_size, 2*hidden_size, 16, 16]
    shape_w2 = [k_size, hidden_size, 16, 16]
    shape_c = [t_size, hidden_size, m_size, 16, 16]
    shape_c_1 = [1, hidden_size, m_size, 16, 16]
    shape_bias1 = [2* hidden_size*16,]
    shape_bias2 = [hidden_size*16,]
    shape_x = [t_size, in_x, m_size, 16, 16]

    x = {"shape":shape_x, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    w1 = {"shape":shape_w1, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    w2 = {"shape":shape_w2, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    b1 = {"shape":shape_bias1, "dtype":bias_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    b2 = {"shape":shape_bias2, "dtype":bias_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    s_init_h_gm = {"shape":shape_c_1, "dtype":bias_dtype, "param_type": "input", "value_range": init_data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    output_y = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    output_h = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    r = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    i = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    n = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    return x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n


precision = precision_info.PrecisionStandard(0.001, 0.002)
no_init_precision = precision_info.PrecisionStandard(0.001, 0.003)

# x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=1, m_size=1, in_x=64, hidden_size=32, bias_dtype='float32')
# ut_case.add_precision_case("all", {
#     "case_name": "init_gateoutput",
#     "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": no_init_precision
# })
#
# x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=1, m_size=1, in_x=64, hidden_size=64, bias_dtype='float32')
# ut_case.add_precision_case("all", {
#     "case_name": "no_init_gateoutput",
#     "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": no_init_precision
# })
#
# x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=1, m_size=1, in_x=128, hidden_size=64, bias_dtype='float32')
# ut_case.add_precision_case("all", {
#     "case_name": "init_no_gateoutput",
#     "params": [x, w1, b1, w2, b2, None, s_init_h_gm, output_y, output_h, None, None, None],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision
# })
#
# x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=1, m_size=1, in_x=64, hidden_size=32, bias_dtype='float32')
# ut_case.add_precision_case("all", {
#     "case_name": "no_init_no_gateoutput",
#     "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": no_init_precision
# })
#
# x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
# ut_case.add_precision_case("all", {
#     "case_name": "init_gateoutput_fp16",
#     "params": [x, w1, b1, w2, b2, None, s_init_h_gm, output_y, output_h, r, i, n],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision
# })

# for cov
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float32')
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None]
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, s_init_h_gm, output_y, output_h, r, i, n]
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None]
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, s_init_h_gm, output_y, output_h, r, i, n]
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "aa", 1, 1.0, -1.0, 0, True, "tanh", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "UNIDIRECTIONAL", 11, 1.0, -1.0, 0, True, "tanh", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "UNIDIRECTIONAL", 1, 11.0, -1.0, 0, True, "tanh", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -11.0, 0, True, "tanh", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 10, True, "tanh", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, False, "tanh", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, None, None, None, "UNIDIRECTIONAL", 1, 1.0, -1.0, 0, True, "sin", True],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, x, None, output_y, output_h, None, None, None],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
r["shape"] = [0]
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, i, r, n],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, i, i, r],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, r, output_h, r, i, n],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
b2["shape"] = [0]
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w1, b2, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
w2["shape"] = [0,0,0]
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
ut_case.add_case("all", {
    "params": [x, w2, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
w1["shape"][1] = 1
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
x["shape"][0] = 99
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
x["shape"][2] = 2
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})
x, w1, b1, w2, b2, s_init_h_gm, output_y, output_h, r, i, n = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float16')
r["dtype"] = 'float32'
ut_case.add_case("all", {
    "params": [x, w1, b1, w2, b2, None, None, output_y, output_h, r, i, n],
    "expect": RuntimeError
})