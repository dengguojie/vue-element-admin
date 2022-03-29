#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

from tbe.dsl.compute.gemm_compute import GEMMCompute
from tbe.dsl.compute.gemm_compute import GEMMComputeParam
from impl.util.platform_adapter import tvm

ut_case = OpUT("GEMM", "impl.gemm", "gemm")
ALPHA_BETA_SHAPE = [1]
for_old_gemm_op_testcase = [
    ((32, 64), (64, 96), 'float16', 'float16', 'FRACTAL_NZ', False, False),
    ((64, 32), (64, 96), 'float16', 'float16', 'FRACTAL_NZ', True, False),
    ((32, 64), (96, 64), 'float16', 'float16', 'FRACTAL_NZ', False, True),
    ((64, 32), (96, 64), 'float16', 'float16', 'FRACTAL_NZ', True, True),
    ((32, 256), (256, 96), 'int8', 'float32', 'ND', False, False),
    ((255, 256), (256, 96), 'int8', 'float32', 'ND', False, False),
    ((32, 256), (96, 256), 'int8', 'float32', 'ND', False, True),
    ((256, 1024), (96, 256), 'int8', 'float32', 'ND', True, True),
    ((255, 1024), (96, 255), 'int8', 'float32', 'ND', True, True),
    ((254, 32), (254, 96), 'int8', 'float32', 'ND', True, False),
    ((32, 256), (256, 96), 'int8', 'float32', 'FRACTAL_NZ', False, False),
    ((16, 16), (16, 16), 'float16', 'float16', 'ND', False, False),
    ((16, 16), (16, 16), 'float16', 'float16', 'ND', True, True),
    ((32, 64), (64, 96), 'int8', 'int32', 'ND', False, False),
    ((512, 512), (512, 512), 'int8', 'int32', 'ND', True, True),
    ((63, 31), (96, 63), 'int8', 'int32', 'ND', True, True),
    ((1024, 512), (96, 512), 'int8', 'int32', 'ND', False, True),
    ((127, 127), (127, 64), 'int8', 'int32', 'ND', False, False),
    ((32, 64), (64, 96), 'int8', 'int32', 'FRACTAL_NZ', False, False),
    ((31, 255), (255, 96), 'float16', 'float32', 'ND', False, False),
    ((256, 32), (256, 96), 'float16', 'float32', 'ND', True, False),
    ((32, 256), (96, 256), 'float16', 'float32', 'ND', False, True),
    ((256, 32), (96, 256), 'float16', 'float32', 'ND', True, True),
    ((1023, 1023), (96, 1023), 'float16', 'float32', 'ND', True, True),
]


def get_kernel_name(shape_a, shape_b, shape_bias, src_dtype, dst_dtype, fractal):
    kernel_name = 'gemm_' + '_'.join(map(str, shape_a)) + '_' + '_'.join(map(str, shape_b)) + \
                '_' + '_'.join(map(str, shape_bias)) + \
                '_' + src_dtype + '_' + dst_dtype + '_' + fractal
    return kernel_name


def gen_trans_data_case(shape_a_ori, shape_b_ori, src_dtype, dst_dtype, trans_a, trans_b, data_format):
    block_reduce = 16
    block_in_out = 16
    output_m = shape_a_ori[0]
    output_n = shape_b_ori[1]
    if trans_a == True:
        output_m = shape_a_ori[1]
    if trans_b == True:
        output_n = shape_b_ori[0]
    output_shape_ori = [output_m, output_n]
    kernel_name = get_kernel_name(shape_a_ori, shape_b_ori, output_shape_ori, src_dtype, dst_dtype, data_format)
    alpha_beta_dtype = dst_dtype
    shape_a = shape_a_ori
    shape_b = shape_b_ori
    output_shape = output_shape_ori
    if data_format == "FRACTAL_NZ":
        format_a = 'FRACTAL_NZ'
        if src_dtype == "int8":
            block_reduce = 32
            format_b = 'FRACTAL_Z'
        else:
            format_b = 'FRACTAL_NZ'
        shape_a = [shape_a_ori[1] // block_reduce, shape_a_ori[0] // block_in_out, block_in_out, block_reduce]
        shape_b = [shape_b_ori[1] // block_in_out, shape_b_ori[0] // block_reduce, block_reduce, block_in_out]
        if format_b == "FRACTAL_Z":
            shape_b[0], shape_b[1] = shape_b[1], shape_b[0]
            shape_b[2], shape_b[3] = shape_b[3], shape_b[2]
        output_shape = [
            output_shape_ori[1] // block_in_out, output_shape_ori[0] // block_in_out, block_in_out, block_in_out
        ]
    else:
        format_a = "ND"
        format_b = "ND"

    tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype)
    tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_beta_dtype)
    tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=alpha_beta_dtype)
    tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
    if format_a == "FRACTAL_NZ":
        trans_a = not trans_a
    if format_b == "FRACTAL_NZ":
        trans_b = not trans_b

    para_dict = {
        "alpha": tensor_alpha,
        "beta": tensor_beta,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": format_a,
        "format_b": format_b,
        "dst_dtype": dst_dtype,
        "tensor_c": tensor_bias,
        "kernel_name": kernel_name
    }
    return tensor_a, tensor_b, para_dict


def test_old_gemm(test_arg):
    for t in for_old_gemm_op_testcase:
        print("current test info:", t)
        tensor_a, tensor_b, para_dict = gen_trans_data_case(t[0], t[1], t[2], t[3], t[5], t[6], t[4])
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()


ut_case.add_cust_test_func(test_func=test_old_gemm)


def test_check_support_alpha_and_beta_dtype_same(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float32"
        dst_dtype = "float32"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test alpha and beta dtype not same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_alpha_and_output_dtype_same(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float32"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test alpha and output dtype not same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_a_and_b_dtype_same(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 32]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "int8"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tensor_a and tenspr_b dtype not same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_output_dtype_support(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 32]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "int8"
        beta_dtype = "int8"
        dst_dtype = "int8"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test output dtype not support pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_input_format(test_arg):
    try:
        format_a = "NCHW"
        format_b = "NCHW"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test input format not support pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_a_and_b_format_not_same(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "ND"
        shape_a = [1, 1, 16, 16]
        shape_b = [16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tensor_a and tensor_b format not same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_shape_and_format_match_a_1(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [16, 16]
        shape_b = [16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tnesor_b input dtype and format not fit pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_dim_same(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tnesor_a and tensor_b's dim should be same. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_shape_and_format_match_a_2(test_arg):
    try:
        format_a = "ND"
        format_b = "ND"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tensor_a and tensor_b format not same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_shape_and_format_match_b_1(test_arg):
    try:
        format_a = "ND"
        format_b = "FRACTAL_NZ"
        shape_a = [16, 16]
        shape_b = [16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tnesor_b input dtype and format not fit pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_shape_and_format_match_b_2(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "ND"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test tensor_a and tensor_b format not same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_batch_dim_must_same(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [4, 1, 1, 16, 16]
        shape_b = [2, 1, 1, 16, 16]
        output_shape = [4, 1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test batch should be same pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_block_reduce(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 32]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error block_reduce case1 pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_gevm_m_shape(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 2, 1, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 2, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error m_shape in gevm pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_gevm_k_shape(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 1, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error k_shape in gevm pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_gemv_n_shape(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [2, 1, 16, 1]
        output_shape = [2, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error n_shape in gemv pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_gemv_n_shape2(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 1]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error k_shape in gemv pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_k_shape(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [2, 1, 16, 16]
        shape_b = [1, 4, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error k_shape pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_bias_shape(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        bias_shape = [1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(bias_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error bias shape pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_n_not_align(test_arg):
    try:
        format_a = "ND"
        format_b = "ND"
        shape_a = [32, 32]
        shape_b = [32, 31]
        bias_shape = [32, 31]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(bias_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test n not align error pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_output_dtype(test_arg):
    try:
        format_a = "ND"
        format_b = "ND"
        shape_a = [32, 32]
        shape_b = [32, 32]
        bias_shape = [32, 32]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "int8"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(bias_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test error output dtype pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_b_format(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "NCHW"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test input b format not support pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_b_nz_k0(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 32, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test input b reduce k error. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_b_nz_k0(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 32, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float16"
        beta_dtype = "float16"
        dst_dtype = "float16"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test input b reduce k error. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_b_uint8(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_Z"
        shape_a = [1, 1, 16, 32]
        shape_b = [1, 1, 16, 32]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "int32"
        src_dtype_b = "int32"
        alpha_dtype = "int32"
        beta_dtype = "uint8"
        dst_dtype = "int8"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test input b reduce k error. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_output_dtype_support_fp16_int32(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_dtype = "float32"
        beta_dtype = "float32"
        dst_dtype = "int32"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test output dtype error pass. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_both_uint8(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_Z"
        shape_a = [1, 1, 16, 32]
        shape_b = [1, 1, 16, 32]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "int32"
        src_dtype_b = "int32"
        alpha_dtype = "int32"
        beta_dtype = "uint8"
        dst_dtype = "uint8"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=alpha_dtype)
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=beta_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test input both uint8. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def init_gemm_compute_param(test_arg):
    GEMMComputeParam()


def test_check_support_alpha_none(test_arg):
    format_a = "FRACTAL_NZ"
    format_b = "FRACTAL_Z"
    shape_a = [1, 1, 16, 32]
    shape_b = [1, 1, 16, 32]
    src_dtype_a = "int8"
    src_dtype_b = "int8"
    dst_dtype = "int32"
    trans_a = False
    trans_b = False

    tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
    tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
    tensor_alpha = None
    tensor_beta = None
    if format_a == "FRACTAL_NZ":
        trans_a = not trans_a
    if format_b == "FRACTAL_NZ":
        trans_b = not trans_b

    para_dict = {
        "alpha": tensor_alpha,
        "beta": tensor_beta,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": format_a,
        "format_b": format_b,
        "dst_dtype": dst_dtype
    }
    gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
    gemm_compute.calculate()


def test_check_support_matmul_bias(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        dst_dtype = "float32"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = 1.0
        tensor_beta = 1.0
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test matmul bias shape error. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_alpha_error(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        alpha_beta_dtype = "int32"
        dst_dtype = "float32"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_alpha = 1
        tensor_beta = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_beta", dtype=dst_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test alpha error. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_check_support_beta_error(test_arg):
    try:
        format_a = "FRACTAL_NZ"
        format_b = "FRACTAL_NZ"
        shape_a = [1, 1, 16, 16]
        shape_b = [1, 1, 16, 16]
        output_shape = [1, 1, 16, 16]
        src_dtype_a = "float16"
        src_dtype_b = "float16"
        dst_dtype = "float32"
        trans_a = False
        trans_b = False

        tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
        tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
        tensor_beta = 1
        tensor_alpha = tvm.placeholder(ALPHA_BETA_SHAPE, name="tensor_alpha", dtype=dst_dtype)
        tensor_bias = tvm.placeholder(output_shape, name="tensor_bias", dtype=dst_dtype)
        if format_a == "FRACTAL_NZ":
            trans_a = not trans_a
        if format_b == "FRACTAL_NZ":
            trans_b = not trans_b

        para_dict = {
            "alpha": tensor_alpha,
            "beta": tensor_beta,
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": dst_dtype,
            "tensor_c": tensor_bias
        }
        gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
        gemm_compute.calculate()
    except RuntimeError as e:
        print("[GEMM] test alpha error. error message is: ", e)
    else:
        raise RuntimeError("run case fail.")


def test_old_matmul_batch(test_arg):
    format_a = "FRACTAL_NZ"
    format_b = "FRACTAL_NZ"
    shape_a = [5, 1, 1, 16, 16]
    shape_b = [5, 1, 1, 16, 16]
    src_dtype_a = "float16"
    src_dtype_b = "float16"
    dst_dtype = "float16"
    trans_a = False
    trans_b = False

    tensor_a = tvm.placeholder(shape_a, name="tensor_a", dtype=src_dtype_a)
    tensor_b = tvm.placeholder(shape_b, name="tensor_b", dtype=src_dtype_b)
    if format_a == "FRACTAL_NZ":
        trans_a = not trans_a
    if format_b == "FRACTAL_NZ":
        trans_b = not trans_b

    para_dict = {
        "trans_a": trans_a,
        "trans_b": trans_b,
        "offset_a": 0,
        "offset_b": None,
        "format_a": format_a,
        "format_b": format_b,
        "dst_dtype": dst_dtype
    }
    gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
    gemm_compute.calculate()


ut_case.add_cust_test_func(test_func=test_check_support_a_and_b_dtype_same)
ut_case.add_cust_test_func(test_func=test_check_support_alpha_and_output_dtype_same)
ut_case.add_cust_test_func(test_func=test_check_support_alpha_and_beta_dtype_same)
ut_case.add_cust_test_func(test_func=test_check_support_output_dtype_support)
ut_case.add_cust_test_func(test_func=test_check_support_input_format)
ut_case.add_cust_test_func(test_func=test_check_support_a_and_b_format_not_same)
ut_case.add_cust_test_func(test_func=test_check_support_dim_same)
ut_case.add_cust_test_func(test_func=test_check_support_shape_and_format_match_a_1)
ut_case.add_cust_test_func(test_func=test_check_support_shape_and_format_match_a_2)
ut_case.add_cust_test_func(test_func=test_check_support_shape_and_format_match_b_1)
ut_case.add_cust_test_func(test_func=test_check_support_shape_and_format_match_b_2)
ut_case.add_cust_test_func(test_func=test_check_support_batch_dim_must_same)
ut_case.add_cust_test_func(test_func=test_check_support_block_reduce)
ut_case.add_cust_test_func(test_func=test_check_support_gevm_m_shape)
ut_case.add_cust_test_func(test_func=test_check_support_gevm_k_shape)
ut_case.add_cust_test_func(test_func=test_check_support_gemv_n_shape)
ut_case.add_cust_test_func(test_func=test_check_support_gemv_n_shape2)
ut_case.add_cust_test_func(test_func=test_check_support_k_shape)
ut_case.add_cust_test_func(test_func=test_check_support_bias_shape)
ut_case.add_cust_test_func(test_func=test_check_support_n_not_align)
ut_case.add_cust_test_func(test_func=test_check_support_output_dtype)
ut_case.add_cust_test_func(test_func=test_check_support_b_format)
ut_case.add_cust_test_func(test_func=test_check_support_b_nz_k0)
ut_case.add_cust_test_func(test_func=test_check_support_b_uint8)
ut_case.add_cust_test_func(test_func=test_check_support_output_dtype_support_fp16_int32)
ut_case.add_cust_test_func(test_func=test_check_support_both_uint8)
ut_case.add_cust_test_func(test_func=init_gemm_compute_param)
ut_case.add_cust_test_func(test_func=test_check_support_alpha_none)
ut_case.add_cust_test_func(test_func=test_check_support_matmul_bias)
ut_case.add_cust_test_func(test_func=test_check_support_alpha_error)
ut_case.add_cust_test_func(test_func=test_check_support_beta_error)
ut_case.add_cust_test_func(test_func=test_old_matmul_batch)

if __name__ == '__main__':
    ut_case.run()
    exit(0)
