#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

from impl.dynamic.mat_mul import get_op_support_info
from impl.dynamic.mat_mul import op_select_format
CUBE_BLOCK = 16
ut_case = OpUT("MatMul", "impl.dynamic.mat_mul", "mat_mul")

# succ case
# m1_range, k1_range, n1_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, case_name, expect_result
matmul_case_succ = [
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_succcase0"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, False, True, "dynamic_matmul_succcase1"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, True, False, "dynamic_matmul_succcase2"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, True, True, "dynamic_matmul_succcase3"),
    ((1, 4), (1, 2, 1, None), (2, 4), "float16", "float16", "NZ", True, True, True, "dynamic_matmul_succcase4"),
]

matmul_ND_case_succ = [
    # m1_range, k1_range, n1_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, case_name, expect_result
    ((256,258), (16, 16), (256, 258), "float16", "float16", "ND", False, False, False, "dynamic_matmul_ND_succcase0"),
    ((256,284), (16, 16), (256, 284), "float16", "float16", "ND", True, True, False, "dynamic_matmul_ND_succcase1"),
    ((256,284), (16, 16), (256, 284), "float16", "float16", "ND", False, True, False, "dynamic_matmul_ND_succcase2"),
    ((256,284), (16, 16), (256, 284), "float16", "float16", "ND", True, False, False, "dynamic_matmul_ND_succcase3"),
]


matmul_case_error = [
    # dtype error
    ((1, 4), (1, 2), (2, 4), "float32", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase0", "dtype"),
    # orishape length is not 2
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase2", "x1_orishape_len"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase3", "x2_orishape_len"),
     # orishape is not with -1
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase4", "orishape_dynamic"),
    # range_relation
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase5", "range_relation"),
    # range lenth error
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase6", "x1_range_length"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase7", "x2_range_length"),
     # range error
    ((-1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase8", "range_error"),
    ((1, -1), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase9", "range_error"),
    ((5, 1), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase10", "range_error"),

]


def gen_matmul_dynamic_succecase(m_range, k_range, n_range, src_dtype, dst_dtype,
                                 format, trans_a, trans_b, bias_flag, case_name):
    """
    gen the case for ut test
    """
    if format == "NZ":
        format = "FRACTAL_NZ"

    shape = [-1, -1]

    if format == "FRACTAL_NZ":
        block_range = [[CUBE_BLOCK, CUBE_BLOCK], [CUBE_BLOCK, CUBE_BLOCK]]
        if len(k_range)  == 4:
            mk_range = k_range[:2]
            nk_range = k_range[2:]
        else:
            mk_range = nk_range = k_range
        x1_range = [m_range, mk_range] if trans_a else [mk_range, m_range]

        x2_range = [nk_range, n_range] if trans_b else [n_range, nk_range]
        y_range = [n_range, m_range]

        x1_range += block_range
        x2_range += block_range
        y_range += block_range
        shape += [CUBE_BLOCK, CUBE_BLOCK]
    elif format == "ND":
        x1_range = [k_range, m_range] if trans_a else [m_range, k_range]
        x2_range = [n_range, k_range] if trans_b else [k_range, n_range]
        y_range = [m_range, n_range]

    x1 = {"ori_shape": (-1, -1), "dtype": src_dtype, "shape": shape,
          "format": format , "ori_format": "ND", "range": x1_range
    }
    x2 = {"ori_shape": (-1, -1), "dtype": src_dtype, "shape": shape,
          "format": format , "ori_format": "ND", "range": x2_range
    }
    y = {"ori_shape": (-1, -1), "dtype": dst_dtype, "shape": shape,
         "format": format , "ori_format": "ND", "range": y_range
    }

    if bias_flag:
        bias_n_range = [16*i for i in n_range] if format == "FRACTAL_NZ" else n_range
        bias = {"ori_shape": (-1, ), "dtype": dst_dtype, "shape": (-1,),
                "format": "ND", "ori_format": "ND", "range": (bias_n_range,)}
    else:
        bias = None
    print([x1, x2, bias, None, y, trans_a, trans_b])

    return {
        "params": [x1, x2, bias, None, y, trans_a, trans_b],
        "case_name": case_name,
        "expect": "success",
        "format_expect":[],
        "support_expect":True
    }

def gen_matmul_dynamic_errorcase(m_range, k_range, n_range, src_dtype, dst_dtype,
                                 format, trans_a, trans_b, bias_flag, case_name, error_mode):
    """
    gen the error case for ut test
    """
    if format == "NZ":
        format = "FRACTAL_NZ"
    block_range = [[CUBE_BLOCK, CUBE_BLOCK], [CUBE_BLOCK, CUBE_BLOCK]]
    x1_range = [m_range, k_range] if trans_a else [k_range, m_range]
    x1_range += block_range
    x2_range = [k_range, n_range] if trans_b else [n_range, k_range]
    x2_range += block_range
    y_range = [n_range, m_range] +  block_range
    x1 = {"ori_shape": (-1, -1), "dtype": src_dtype, "shape": (-1, -1, CUBE_BLOCK, CUBE_BLOCK),
          "format": format , "ori_format": "ND", "range": x1_range
    }
    x2 = {"ori_shape": (-1, -1), "dtype": src_dtype, "shape": (-1, -1, CUBE_BLOCK, CUBE_BLOCK),
          "format": format , "ori_format": "ND", "range": x2_range
    }
    y = {"ori_shape": (-1, -1), "dtype": src_dtype, "shape": (-1, -1, CUBE_BLOCK, CUBE_BLOCK),
         "format": format , "ori_format": "ND", "range": y_range
    }
    if bias_flag:
        bias = {"ori_shape": (-1, ), "dtype": dst_dtype, "shape": (-1,),
                "format": "ND", "ori_format": "ND", "range": [n_range]
               }
    else:
        bias = None
    offset_x = 0
    offset_w = None
    if error_mode == "x1_orishape_len":
        x1["ori_shape"] = (-1, -1, -1)
    elif error_mode == "x2_orishape_len":
        x2["ori_shape"] = (-1, -1, -1)
    elif error_mode == "orishape_dynamic":
        x1["ori_shape"] = (1, 1)
        x2["ori_shape"] = (1, 1)
    elif error_mode == "range_relation":
        x1["range"][0], x1["range"][1] = x1["range"][1], x1["range"][0]
    elif error_mode == "x1_range_length":
        x1["range"] = x1["range"] + [[CUBE_BLOCK, CUBE_BLOCK]]
    elif error_mode == "x2_range_length":
        x2["range"] = x2["range"] + [[CUBE_BLOCK, CUBE_BLOCK]]

    return {
        "params": [x1, x2, bias, offset_w, y, trans_a, trans_b, offset_x],
        "case_name": case_name,
        "expect": RuntimeError
    }

# [shape_x1, range_x1, trans_a], [shape_x2, range_x2, trans_b], [shape_bias, range_bias] [shape_out, range_out]
common_cases = [
    [[[-2], [[1, None]], False], [[-2], [[1, None]], False], [[], []], [[-2], [[1, None]]]],
]

support_format = ['FRACTAL_NZ']
support_dtype = ['float16']

def shape_nd_to_Nz(shape, dtype):
    def helper(sp):
        if len(sp) != 2:
            raise RuntimeError("Not support len of shape != 2")
        a, b = sp
        if a < -2 or b < -2:
            raise RuntimeError("a, b must >= -2")

        block_reduce = {'int8': 32, 'float16': 16}[dtype]
        return [-1 if b == -1 else (b+block_reduce-1)//block_reduce,
                -1 if a == -1 else (a+15)//16,
                16,
                block_reduce]
    if list(shape) == [-2]:
        return (-2,)
    return shape[:-2] + helper(shape[-2:])

def gen_kernel_name(param):
    def str_list(lst):
        if not lst:
            return "None"
        return '_'.join(str(x) if x > 0 else f'm{x*-1}' for x in lst)

    def str_bool(val):
        return 'T' if val else 'F'

    x1, x2, bias, offset_w, y, trans_a, trans_b, offset_x = param
    return f'a_{str_list(x1["ori_shape"])}_b_{str_list(x2["ori_shape"])}_c_{str_list(bias["ori_shape"] if bias else None)}_trans_{str_bool(trans_a)}_{str_bool(trans_b)}'

def gen_cases_by_shape_and_range(case):
    [shape_x1, range_x1, trans_a], [shape_x2, range_x2, trans_b], [shape_bias, range_bias], [shape_out, range_out] = case
    params = []
    offset_w = None
    offset_x = 0

    from itertools import product
    for format, dtype in product(support_format, support_dtype):
        x1 = {"ori_shape": shape_x1, "dtype": dtype, "shape": shape_nd_to_Nz(shape_x1, dtype),
              "format": format, "ori_format": "ND", "range": range_x1
              }
        x2 = {"ori_shape": shape_x2, "dtype": dtype, "shape":  shape_nd_to_Nz(shape_x2, dtype),
              "format": format, "ori_format": "ND", "range": range_x2
              }

        if shape_bias != [] and range_bias != []:
            bias = {"ori_shape": shape_bias, "dtype": dtype, "shape":  shape_bias,
                    "format": "ND", "ori_format": "ND", "range": range_bias
                    }
        else:
            bias = None

        y = {"ori_shape": shape_out, "dtype": dtype, "shape": shape_nd_to_Nz(shape_out, dtype),
             "format": format, "ori_format": "ND", "range": range_out
             }
        param = [x1, x2, bias, offset_w, y, trans_a, trans_b, offset_x]
        params.append({"params": param, "case_name": f'dynamic_batchmatmul_{gen_kernel_name(param)}', "expect": "success"})

    return params

for case in common_cases:
    for param in gen_cases_by_shape_and_range(case):
        ut_case.add_case("Ascend910A", param)

for case in matmul_case_succ:
   ut_case.add_case("Ascend910A", gen_matmul_dynamic_succecase(*case))

for error_case in matmul_case_error:
    ut_case.add_case("Ascend910A", gen_matmul_dynamic_errorcase(*error_case))

def test_get_op_support_info_dynamic_matmul(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, 16, 16), "ori_shape": (-1, -1),
         "range": ((16, 48), (16, 48), (16, 16), (16, 16))}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, 16, 16), "ori_shape": (-1, -1),
         "range": ((16, 48), (16, 48), (16, 16), (16, 16))}
    get_op_support_info(x1, x2, None)

def test_op_select_format_dynamic_matmul(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, 16, 16), "ori_shape": (-1, -1),
         "range": ((16, 48), (16, 48), (16, 16), (16, 16))}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, 16, 16), "ori_shape": (-1, -1),
         "range": ((16, 48), (16, 48), (16, 16), (16, 16))}
    op_select_format(x1, x2)

for case in matmul_case_succ:
    ut_case.add_case("Ascend910A", gen_matmul_dynamic_succecase(*case))
for case in matmul_ND_case_succ:
    ut_case.add_case("Ascend910A", gen_matmul_dynamic_succecase(*case))
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_matmul)
ut_case.add_cust_test_func(test_func=test_op_select_format_dynamic_matmul)
print(ut_case)

if __name__ == "__main__":
    ut_case._case_info_map = {}
    ut_case.run(["Ascend910A"])
    exit(0)
