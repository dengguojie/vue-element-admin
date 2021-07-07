#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from matmul_cpu_dsl import matmul_cpu_validation

ut_case = OpUT("MatMul", None, None)

case1 = {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND"},
                    {"shape": (4, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND"},
                    {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND"},
                    None,
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND"},
                    False, False],
         "case_name": "MatMul_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 32),"ori_format": "ND"},
                    {"shape": (6, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96),"ori_format": "ND"},
                    {"shape": (64,), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND"},
                    None,
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND"},
                    True, True],
         "case_name": "MatMul_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND"},
                    {"shape": (6, 4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96),"ori_format": "ND"},
                    {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND"},
                    None,
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND"},
                    False, True],
         "case_name": "MatMul_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (64, 1280, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20480, 1024),"ori_format": "ND"},
                    {"shape": (64, 1280,16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20480, 1024),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (64, 64, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (1024, 1024),"ori_format": "ND"},
                    True, False],
         "case_name": "MatMul_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910B"], case4)
#precision cases

def maxtrix_zN_reverse(matrix, shape, dtype):
    idx = 0
    j_outer,i_outer,i_inner,j_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        # print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for j in range(0, j_outer):
                for i in range(0, i_outer):
                    for ii in range(0, i_inner):
                        for jj in range(0, j_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for j in range(0, j_outer):
            for i in range(0, i_outer):
                for ii in range(0, i_inner):
                    for jj in range(0, j_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
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
    return tmp

def calc_expect_func(x1, x2, bias, offset_w, y, trans_a, trans_b):
    #only support FRACTAL_NZ FRACTAL_NZ False False
    a = maxtrix_zN_reverse(x1['value'].flatten(), x1['shape'], x1['dtype'])
    b = maxtrix_zN_reverse(x2['value'].flatten(), x2['shape'], x2['dtype'])

    res = np.matmul(a, b) + bias['value']
    res = matrix_to_zN(res, res.shape, res.dtype)
    res = res.reshape(y['shape']).astype(y['dtype'])
    return res

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ),"ori_format": "ND", "param_type": "input"},
                                                    None,
                                                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND", "param_type": "output"},
                                                    False, False],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})


# cpu dsl validation case
dsl_case = {
    "params": [
        {"shape": (6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96), "ori_format": "ND"},
        {"shape": (4, 6, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64), "ori_format": "ND"},
        {"shape": (64, ), "dtype": "float16", "format": "ND", "ori_shape": (64, ), "ori_format": "ND"},
        None,
        {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64), "ori_format": "ND"},
        False,
        False
    ],
    "case_name": "MatMul_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}


def test_matmul_cpu_validation(test_arg):
    params = dsl_case["params"]
    matmul_cpu_validation(params)

ut_case.add_cust_test_func(test_func=test_matmul_cpu_validation)

def test_check_support(test_arg):
    from impl.mat_mul import check_supported
    check_supported({"shape": (0, 10), "ori_shape": (0, 10), "dtype": "float16", "format": "ND", "ori_format": "ND"},
                    {"shape": (10, 20), "ori_shape": (10, 20), "dtype": "float16", "format": "ND", "ori_format": "ND"},
                    None,
                    {},
                    {"shape": (0, 20), "ori_shape": (0, 20), "dtype": "float16", "format": "ND", "ori_format": "ND"},
                    trans_a=False,
                    trans_b=False,
                    offset_x=0,
                    kernel_name="matmul")

ut_case.add_cust_test_func(test_func=test_check_support)
