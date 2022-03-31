#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from gemm_ut_testcase import gemm_op_testcase


ut_case = OpUT("GEMM", "impl.gemm", "gemm")


def get_kernel_name(shape_a, shape_b, shape_bias, src_dtype, dst_dtype, fractal):
    kernel_name = 'gemm_' + '_'.join(map(str, shape_a)) + '_' + '_'.join(map(str, shape_b)) + \
                '_' + '_'.join(map(str, shape_bias)) + \
                '_' + src_dtype + '_' + dst_dtype + '_' + fractal
    return kernel_name

def gen_trans_data_case(shape_a_ori, shape_b_ori, src_dtype,dst_dtype,trans_a,trans_b,data_format, expect):
    block_reduce = 16
    block_in_out = 16
    output_m = shape_a_ori[0]
    output_n = shape_b_ori[1]
    if trans_a == True:
        output_m =  shape_a_ori[1]
    if trans_b == True:
        output_n =  shape_b_ori[0]
    output_shape_ori = [output_m,output_n]
    kernel_name = get_kernel_name(shape_a_ori, shape_b_ori, output_shape_ori,src_dtype, dst_dtype, data_format)
    alpha_beta_dtype = dst_dtype
    alpha = {"ori_shape":[1], "shape":[1], "format":"ND", "ori_format":"ND", "dtype":alpha_beta_dtype}
    beta = {"ori_shape":[1], "shape":[1], "format":"ND", "ori_format":"ND", "dtype":alpha_beta_dtype}
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
        output_shape = [output_shape_ori[1] // block_in_out, output_shape_ori[0] // block_in_out, block_in_out, block_in_out]
    else:
        format_a = "ND"
        format_b = "ND"
    input_x = {"shape":shape_a, "ori_shape": shape_a_ori, "dtype":src_dtype, "format":format_a, "ori_format":"ND"}
    input_y = {"shape":shape_b, "ori_shape": shape_b_ori, "dtype":src_dtype, "format":format_b, "ori_format":"ND"}
    input_bias = {"shape":output_shape, "ori_shape":output_shape_ori, "dtype":dst_dtype, "format": format_a, "ori_format":"ND"}
    if data_format == "FRACTAL_NZ":
        if dst_dtype == "int32":
            input_bias["format"] = "ND"
    else:
        input_bias["format"] = "ND"
    output = {"ori_shape":output_shape, "shape":output_shape_ori, "ori_format": format_a, "format":format_a, "dtype":dst_dtype}
    return {"params": [input_x,input_y,input_bias,alpha,beta,output,trans_a,trans_b],
            "case_name": kernel_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def test_op_check_supported(test_arg):
    from impl.gemm import check_supported
    input_x1 = {"ori_shape": (16, 5), "dtype": "float16", "ori_format": "ND"}
    input_x2 = {"ori_shape": (5, 32), "dtype": "float16", "ori_format": "ND"}
    bias = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    ret = check_supported(input_x1,
                          input_x2,
                          bias,
                          alpha,
                          beta,
                          output_y=output_y,
                          trans_a=False,
                          trans_b=False,
                          kernel_name="gemm")
    assert ret[0], f"the result should be true"


def test_op_check_supported_nz(test_arg):
    from impl.gemm import check_supported
    input_x1 = {"ori_shape": (1, 1, 16, 16), "dtype": "float16", "ori_format": "FRACTAL_NZ"}
    input_x2 = {"ori_shape": (1, 1, 16, 16), "dtype": "float16", "ori_format": "FRACTAL_NZ"}
    bias = {"ori_shape": (2, 2), "dtype": "float16", "ori_format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    output_y = {"ori_shape": (2, 2), "dtype": "float16", "ori_format": "ND"}
    ret = check_supported(input_x1,
                          input_x2,
                          bias,
                          alpha,
                          beta,
                          output_y=output_y,
                          trans_a=False,
                          trans_b=False,
                          kernel_name="gemm")
    assert ret[0], f"the result should be true"


def test_op_check_supported_a_shape_len_error(test_arg):
    from impl.gemm import check_supported
    input_x1 = {"ori_shape": (1, 16, 5), "dtype": "float16", "ori_format": "ND"}
    input_x2 = {"ori_shape": (5, 32), "dtype": "float16", "ori_format": "ND"}
    bias = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    ret = check_supported(input_x1,
                          input_x2,
                          bias,
                          alpha,
                          beta,
                          output_y=output_y,
                          trans_a=False,
                          trans_b=False,
                          kernel_name="gemm")
    assert (not ret[0]), f"the result should be false"



def test_op_check_supported_b_shape_len_error(test_arg):
    from impl.gemm import check_supported
    input_x1 = {"ori_shape": (16, 5), "dtype": "float16", "ori_format": "ND"}
    input_x2 = {"ori_shape": (1, 5, 32), "dtype": "float16", "ori_format": "ND"}
    bias = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    ret = check_supported(input_x1,
                          input_x2,
                          bias,
                          alpha,
                          beta,
                          output_y=output_y,
                          trans_a=False,
                          trans_b=False,
                          kernel_name="gemm")
    assert (not ret[0]), f"the result should be false"


def test_op_check_supported_k_dim_not_equal(test_arg):
    from impl.gemm import check_supported
    input_x1 = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    input_x2 = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    bias = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    ret = check_supported(input_x1,
                    input_x2,
                    bias,
                    alpha,
                    beta,
                    output_y=output_y,
                    trans_a=False,
                    trans_b=False,
                    kernel_name="gemm")
    assert (not ret[0]), f"the result should be false"


def test_op_check_supported_bias_dim_not_equal_output(test_arg):
    from impl.gemm import check_supported
    input_x1 = {"ori_shape": (16, 5), "dtype": "float16", "ori_format": "ND"}
    input_x2 = {"ori_shape": (5, 32), "dtype": "float16", "ori_format": "ND"}
    bias = {"ori_shape": (16, 64), "dtype": "float16", "ori_format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "ori_format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "ori_format": "ND"}
    ret = check_supported(input_x1,
                          input_x2,
                          bias,
                          alpha,
                          beta,
                          output_y=output_y,
                          trans_a=False,
                          trans_b=False,
                          kernel_name="gemm")
    assert (not ret[0]), f"the result should be false"


for t in gemm_op_testcase:
    print("adding gemm op testcases")
    ut_case.add_case("Ascend910A", gen_trans_data_case(t[0], t[1], t[2], t[3], t[7], t[8], t[6], "success"))

ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_check_supported_nz)
ut_case.add_cust_test_func(test_func=test_op_check_supported_a_shape_len_error)
ut_case.add_cust_test_func(test_func=test_op_check_supported_b_shape_len_error)
ut_case.add_cust_test_func(test_func=test_op_check_supported_k_dim_not_equal)
ut_case.add_cust_test_func(test_func=test_op_check_supported_bias_dim_not_equal_output)


def test_op_select_support_nd_not_trans_align_n(test_arg):
    from impl.gemm import op_select_format
    input_x1 = {"ori_shape": (16, 5), "dtype": "float16", "format": "ND"}
    input_x2 = {"ori_shape": (5, 32), "dtype": "float16", "format": "ND"}
    bias = {"ori_shape": (16, 32), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "format": "ND"}
    op_select_format(input_x1,
                     input_x2,
                     bias,
                     alpha,
                     beta,
                     output_y=output_y,
                     trans_a=False,
                     trans_b=False,
                     kernel_name="gemm")


def test_op_select_support_nd_trans_align_n(test_arg):
    from impl.gemm import op_select_format
    input_x1 = {"ori_shape": (16, 5), "dtype": "float16", "format": "ND"}
    input_x2 = {"ori_shape": (32, 5), "dtype": "float16", "format": "ND"}
    bias = {"ori_shape": (16, 32), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (16, 32), "dtype": "float16", "format": "ND"}
    op_select_format(input_x1,
                     input_x2,
                     bias,
                     alpha,
                     beta,
                     output_y=output_y,
                     trans_a=False,
                     trans_b=True,
                     kernel_name="gemm")


def test_op_select_support_nd_not_trans_not_align_n(test_arg):
    from impl.gemm import op_select_format
    input_x1 = {"ori_shape": (16, 5), "dtype": "float16", "format": "ND"}
    input_x2 = {"ori_shape": (5, 31), "dtype": "float16", "format": "ND"}
    bias = {"ori_shape": (16, 31), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (16, 31), "dtype": "float16", "format": "ND"}
    op_select_format(input_x1,
                     input_x2,
                     bias,
                     alpha,
                     beta,
                     output_y=output_y,
                     trans_a=False,
                     trans_b=False,
                     kernel_name="gemm")


def test_op_select_support_nz(test_arg):
    from impl.gemm import op_select_format
    input_x1 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    input_x2 = {"ori_shape": (2, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    bias = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    op_select_format(input_x1,
                     input_x2,
                     bias,
                     alpha,
                     beta,
                     output_y=output_y,
                     trans_a=False,
                     trans_b=False,
                     kernel_name="gemm")


def test_get_op_support_info_a_nz_b_nz_bias_nz(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    input_x2 = {"ori_shape": (2, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    bias = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=False,
                        trans_b=False,
                        kernel_name="gemm")


def test_get_op_support_info_a_nz_b_nz_bias_nd(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    input_x2 = {"ori_shape": (2, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    bias = {"ori_shape": (32, 32), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=False,
                        trans_b=False,
                        kernel_name="gemm")


def test_get_op_support_info_a_not_trans_nd_b_not_trans_nd_bias_nd(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (32, 16), "dtype": "float16", "format": "ND"}
    input_x2 = {"ori_shape": (16, 32), "dtype": "float16", "format": "ND"}
    bias = {"ori_shape": (32, 32), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (32, 32), "dtype": "float16", "format": "ND"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=False,
                        trans_b=False,
                        kernel_name="gemm")


def test_get_op_support_info_a_trans_nd_b_trans_nd_bias_nd(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (16, 32), "dtype": "float16", "format": "ND"}
    input_x2 = {"ori_shape": (32, 16), "dtype": "float16", "format": "ND"}
    bias = {"ori_shape": (32, 32), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (32, 32), "dtype": "float16", "format": "ND"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=True,
                        trans_b=True,
                        kernel_name="gemm")


def test_get_op_support_info_int8(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (16, 32), "dtype": "int8", "format": "ND"}
    input_x2 = {"ori_shape": (32, 16), "dtype": "int8", "format": "ND"}
    bias = {"ori_shape": (32, 32), "dtype": "int8", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "int32", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "int32", "format": "ND"}
    output_y = {"ori_shape": (32, 32), "dtype": "int32", "format": "ND"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=True,
                        trans_b=True,
                        kernel_name="gemm")


def test_get_op_support_info_a_nz_b_zn_bias_nd(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    input_x2 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_Z"}
    bias = {"ori_shape": (32, 32), "dtype": "float16", "format": "ND"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=False,
                        trans_b=False,
                        kernel_name="gemm")


def test_get_op_support_info_a_nz_b_zn_bias_nz(test_arg):
    from impl.gemm import get_op_support_info
    input_x1 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    input_x2 = {"ori_shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_Z"}
    bias = {"ori_shape": (2, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    alpha = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    beta = {"ori_shape": (1, ), "dtype": "float16", "format": "ND"}
    output_y = {"ori_shape": (2, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ"}
    get_op_support_info(input_x1,
                        input_x2,
                        bias,
                        alpha,
                        beta,
                        output_y=output_y,
                        trans_a=False,
                        trans_b=False,
                        kernel_name="gemm")

ut_case.add_cust_test_func(test_func=test_op_select_support_nd_not_trans_align_n)
ut_case.add_cust_test_func(test_func=test_op_select_support_nd_trans_align_n)
ut_case.add_cust_test_func(test_func=test_op_select_support_nd_not_trans_not_align_n)
ut_case.add_cust_test_func(test_func=test_op_select_support_nz)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_a_nz_b_nz_bias_nz)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_a_nz_b_nz_bias_nd)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_a_not_trans_nd_b_not_trans_nd_bias_nd)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_a_trans_nd_b_trans_nd_bias_nd)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_int8)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_a_nz_b_zn_bias_nd)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_a_nz_b_zn_bias_nz)


def test_gemm_shapa_a_and_b_dtype_not_equal(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "int8", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_shapa_a_and_b_dtype_not_equal fail.")

def test_gemm_shapa_alpha_beta_dtype_not_equal(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_shapa_alpha_beta_dtype_not_equal fail.")

def test_gemm_shapa_alpha_out_dtype_not_equal(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_shapa_alpha_out_dtype_not_equal fail.")


def test_gemm_input_int8_error_out_dtype(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "int8", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "int8", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_input_int8_error_out_dtype fail.")


def test_gemm_input_float16_error_out_dtype(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "int32",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_input_float16_error_out_dtype fail.")

def test_gemm_error_input_dtype(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float32",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float32", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_input_dtype fail.")

def test_gemm_error_shape_a(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (1, 32, 16), "ori_shape": (1, 32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_shape_a fail.")

def test_gemm_error_shape_b(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (1, 16, 32), "ori_shape": (1, 16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_shape_a fail.")


def test_gemm_error_k_shape_a_and_b(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_k_shape_a_and_b fail.")


def test_gemm_error_dtype_bias_and_out(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (32, 32),
            "dtype": "float32",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_dtype_bias_and_out fail.")


def test_gemm_error_flow(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (1, 1, 16, 32), "ori_shape": (32, 16), "dtype": "int8", "format": "FRACTAL_NZ", "ori_format": "ND"}
        input_x2 = {"shape": (1, 1, 16, 32), "ori_shape": (16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_format": "ND"}
        bias = {
            "shape": (2, 2, 16, 16),
            "ori_shape": (32, 32),
            "dtype": "int32",
            "format": "FRACTAL_NZ",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (2, 2, 16, 16), "ori_shape": (32, 32), "dtype": "int32", "format": "FRACTAL_NZ", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_flow fail.")

def test_gemm_error_bias_shape(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "int8", "format": "ND", "ori_format": "ND"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "int8", "format": "ND", "ori_format": "ND"}
        bias = {
            "shape": (32, 64),
            "ori_shape": (32, 64),
            "dtype": "int32",
            "format": "ND",
            "ori_format": "ND"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_bias_shape fail.")


def test_gemm_error_bias_shape_len(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "FRACTAL_NZ"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "FRACTAL_NZ"}
        bias = {
            "shape": (32, 32),
            "ori_shape": (1, 2, 2, 16, 16),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "FRACTAL_NZ"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "FRACTAL_NZ"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_bias_shape fail.")


def test_gemm_error_bias_shape_2(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "float16", "format": "ND", "ori_format": "FRACTAL_NZ"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "float16", "format": "ND", "ori_format": "FRACTAL_NZ"}
        bias = {
            "shape": (32, 64),
            "ori_shape": (32, 64),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "FRACTAL_NZ"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "float16", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "float16", "format": "ND", "ori_format": "FRACTAL_NZ"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_bias_shape_2 fail.")

def test_gemm_error_bias_shape_3(test_arg):
    from impl.gemm import gemm
    try:
        input_x1 = {"shape": (32, 16), "ori_shape": (32, 16), "dtype": "int8", "format": "ND", "ori_format": "FRACTAL_NZ"}
        input_x2 = {"shape": (16, 32), "ori_shape": (16, 32), "dtype": "int8", "format": "ND", "ori_format": "FRACTAL_NZ"}
        bias = {
            "shape": (32, 64, 16, 16),
            "ori_shape": (32, 64, 16, 16),
            "dtype": "int32",
            "format": "ND",
            "ori_format": "FRACTAL_NZ"
        }
        alpha = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        beta = {"shape": (1, ), "ori_shape": (1, ), "dtype": "int32", "format": "ND", "ori_format": "ND"}
        output_y = {"shape": (32, 32), "ori_shape": (32, 32), "dtype": "int32", "format": "ND", "ori_format": "FRACTAL_NZ"}
        gemm(input_x1, input_x2, bias, alpha, beta, output_y=output_y, trans_a=False, trans_b=False, kernel_name="gemm")
    except RuntimeError as e:
        print(e)
    else:
        raise RuntimeError("run test_gemm_error_bias_shape_2 fail.")

ut_case.add_cust_test_func(test_func=test_gemm_shapa_a_and_b_dtype_not_equal)
ut_case.add_cust_test_func(test_func=test_gemm_shapa_alpha_beta_dtype_not_equal)
ut_case.add_cust_test_func(test_func=test_gemm_shapa_alpha_out_dtype_not_equal)
ut_case.add_cust_test_func(test_func=test_gemm_input_int8_error_out_dtype)
ut_case.add_cust_test_func(test_func=test_gemm_input_float16_error_out_dtype)
ut_case.add_cust_test_func(test_func=test_gemm_error_input_dtype)
ut_case.add_cust_test_func(test_func=test_gemm_error_shape_a)
ut_case.add_cust_test_func(test_func=test_gemm_error_shape_b)
ut_case.add_cust_test_func(test_func=test_gemm_error_k_shape_a_and_b)
ut_case.add_cust_test_func(test_func=test_gemm_error_dtype_bias_and_out)
ut_case.add_cust_test_func(test_func=test_gemm_error_flow)
ut_case.add_cust_test_func(test_func=test_gemm_error_bias_shape)
ut_case.add_cust_test_func(test_func=test_gemm_error_bias_shape_2)
ut_case.add_cust_test_func(test_func=test_gemm_error_bias_shape_3)
ut_case.add_cust_test_func(test_func=test_gemm_error_bias_shape_len)


if __name__ == '__main__':
    ut_case.run()
    exit(0)
