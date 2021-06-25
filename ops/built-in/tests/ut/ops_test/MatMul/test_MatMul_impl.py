#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm
from te.lang.cce import vadds
from te.lang.cce import cce_build_code
from te.tvm.target import cce
from topi.generic import auto_schedule
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
from impl.mat_mul import get_op_support_info
from impl.mat_mul import mat_mul_compute
from impl.leaky_relu import leaky_relu_compute
from impl.mat_mul import check_supported
from impl.confusion_transpose_d import confusion_transpose_d_compute
from te.platform.cce_conf import te_set_version
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

case4 = {"params": [{"shape": (32, 96), "dtype": "float32", "format": "ND", "ori_shape": (32, 96),"ori_format": "ND"},
                    {"shape": (64, 96), "dtype": "float32", "format": "ND", "ori_shape": (64, 96),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (32, 64), "dtype": "float32", "format": "ND", "ori_shape": (32, 64),"ori_format": "ND"},
                    False, True],
         "case_name": "MatMul_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

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
ut_case.add_precision_case("Ascend920A", {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND", "param_type": "input"},
                                                    None,
                                                    None,
                                                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND", "param_type": "output"},
                                                    False, False],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

ut_case.add_precision_case("Ascend920A", {"params": [{"shape": (6, 2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 96),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (4, 6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (64,), "dtype": "float32", "format": "ND", "ori_shape": (64,),"ori_format": "ND", "param_type": "input"},
                                                    None,
                                                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64),"ori_format": "ND", "param_type": "output"},
                                                    False, False],
                                         "expect": "success",
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})



case_fp16_transpose_nz_1  = {"params": [{"shape": (4,6,16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND"},
                    {"shape": (4,6,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (96, 64),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (4,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64),"ori_format": "ND"},
                    True, False],
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case_fp16_transpose_nz_2  = {"params": [{"shape": (6,4,16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96),"ori_format": "ND"},
                    {"shape": (6,4,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96),"ori_format": "ND"},
                    None,
                    None,
                    {"shape": (4,4,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (64, 64),"ori_format": "ND"},
                    False, True],
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend920A"], case_fp16_transpose_nz_1)
ut_case.add_case(["Ascend920A"], case_fp16_transpose_nz_2)



def test_split_matmul(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (2, 1, 16, 16), "ori_shape": (16, 32)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (1, 2, 16, 16), "ori_shape": (32, 16)}
    get_op_support_info(x1, x2, None)
ut_case.add_cust_test_func(test_func=test_split_matmul)

def test_matmul_confusion_transpose_910(test_arg):
    te_set_version("Ascend910")
    with cce():
        x1 = tvm.placeholder((64, 768, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (12288, 1024)}, dtype="float16")
        x2 = tvm.placeholder((64, 64, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 1024)}, dtype="float16")
        output_y = {"shape": (64, 768, 16, 16), "dtype": "float16", "ori_shape": (12288, 1024), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
        y = {"shape": (24, 16, 4, 32, 16, 16), "ori_shape": (24, 16, 512, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (24, 512, 16, 64), False)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "matmul_confusion_transpose_910",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)
    te_set_version("Ascend310")

def test_matmul_confusion_transpose_710(test_arg):
    te_set_version("Ascend710")
    with cce():
        x1 = tvm.placeholder((48, 64, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 768)}, dtype="float16")
        x2 = tvm.placeholder((48, 48, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (768, 768)}, dtype="float16")
        output_y = {"shape": (48, 64, 16, 16), "dtype": "float16", "ori_shape": (1024, 768), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
        y = {"shape": (8, 12, 4, 8, 16, 16), "ori_shape": (8, 12, 128, 64), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (8, 128, 12, 64), False)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "matmul_confusion_transpose_710",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)
    te_set_version("Ascend310")

ut_case.add_cust_test_func(test_func=test_matmul_confusion_transpose_910)
ut_case.add_cust_test_func(test_func=test_matmul_confusion_transpose_710)


def test_check_support(test_arg):
    from tbe.common.context import op_context
    def _test_supported(case):
        try:
            check_supported(*case["params"], kernel_name="matmul")
        except Exception as e:
            print("The case is not supported")
    def _test_fuzzily_supported(case):
        with op_context.OpContext("dynamic"):
            context = op_context.get_context()
            context.set_build_type("fuzzily_build")
            try:
                check_supported(*case["params"], kernel_name="matmul")
            except Exception as e:
                print("The case is not supported")
    _test_fuzzily_supported(case4)
    _test_supported(case1)


ut_case.add_cust_test_func(test_func=test_check_support)


def test_matmul_multi_output(test_arg):
    with cce():
        x1 = tvm.placeholder((48, 64, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (1024, 768)}, dtype="float16")
        x2 = tvm.placeholder((48, 48, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (768, 768)}, dtype="float16")
        output_y = {"shape": (48, 64, 16, 16), "dtype": "float16", "ori_shape": (1024, 768), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = mat_mul_compute(x1, x2, None, None, output_y)
        relu_out = leaky_relu_compute(matmul_out, None)
        add_out = vadds(relu_out, 1)
        tensor_list = [x1, x2, matmul_out, add_out]
        out = [matmul_out, add_out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "matmul_multi_output",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)
ut_case.add_cust_test_func(test_func=test_matmul_multi_output)