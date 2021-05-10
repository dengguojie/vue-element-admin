#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import flags
from math import ceil

from op_test_frame.ut import OpUT

CUBE_BLOCK = 16
ut_case = OpUT("BatchMatMul", "impl.dynamic.batch_matmul", "batch_matmul")


# batch_range, m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, batchb_flag, case_name
matmul_case = [
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, False, "dynamic_batch_matmul_v2_successcase0"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, True, True, True, "dynamic_batch_matmul_v2_successcase1"),
    # dtype error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float32", "NZ", False, False, False, True, "dynamic_matmul_errorcase0", "dtype"),
    # format error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "ND", False, False, False, True, "dynamic_matmul_errorcase1", "format"),
    # ori_shape error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_errorcase2", "x1_orishape"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_errorcase3", "x2_orishape"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_errorcase4", "dynamic_mode"),
    # range error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_errorcase5", "x1_range"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_errorcase6", "x2_range")
]


def gen_batch_matmul_dynamic(batch_range, m_range, k_range, n_range, src_dtype, dst_dtype,
                             format, trans_a, trans_b, bias_flag, batchb_flag, case_name, error_mode=None):
    """
    gen the error case for ut test
    """

    if format == "NZ":
        format = "FRACTAL_NZ"
    block_range = [[CUBE_BLOCK, CUBE_BLOCK], [CUBE_BLOCK, CUBE_BLOCK]]

    x1_range = [*batch_range, m_range, k_range] if trans_a else [*batch_range, k_range, m_range]
    x1_range += block_range
    x2_range = [k_range, n_range] if trans_b else [n_range, k_range]
    x2_range += block_range
    y_range = [*batch_range, n_range, m_range] + block_range

    if batchb_flag:
        x2_range = [*batch_range] + x2_range

    x1_ori_shape = (-1,) * (len(x1_range) - 2)
    x2_ori_shape = (-1,) * (len(x2_range) - 2)


    x1 = {"ori_shape": x1_ori_shape, "dtype": src_dtype, "shape": x1_ori_shape + (CUBE_BLOCK, CUBE_BLOCK),
          "format": format , "ori_format": "ND", "range": x1_range
    }
    x2 = {"ori_shape": x2_ori_shape, "dtype": src_dtype, "shape":  x2_ori_shape + (CUBE_BLOCK, CUBE_BLOCK),
          "format": format , "ori_format": "ND", "range": x2_range
    }
    y = {"ori_shape": x1_ori_shape, "dtype": dst_dtype, "shape": x1_ori_shape + (CUBE_BLOCK, CUBE_BLOCK),
         "format": format , "ori_format": "ND", "range": y_range
    }

    if bias_flag:
        bias_n_range = [16*i for i in n_range]
        bias = {"ori_shape": (-1, ), "dtype": dst_dtype, "shape": (-1,),
                "format": "ND", "ori_format": "ND", "range": (bias_n_range,)}
    else:
        bias = None

    if error_mode == "x1_orishape":
        x1["ori_shape"] = [-1, -1]
    elif error_mode == "x2_orishape":
        x2["ori_shape"] = [-1]
    elif error_mode == "dynamic_mode":
        x1["ori_shape"] = [1, 1, 1]
        x2["ori_shape"] = [1, 1, 1]
    elif error_mode == "x1_range":
        x1["shape"] = x1["shape"][:3]
        x1["range"] = x1["range"][:3]
    elif error_mode == "x2_range":
        x2["shape"] = x2["shape"][:3]
        x2["range"] = x2["range"][:3]

    if error_mode is None:
        expect = "success"
    else:
        expect = RuntimeError
    return {
        "params": [x1, x2, bias, y, trans_a, trans_b],
        "case_name": case_name,
        "expect": expect
    }

def test_op_select_format(test_arg):
    from impl.batch_matmul import op_select_format
    # static shape
    op_select_format({"shape": (3, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (3, 4, 5), "dtype": "float16", "format": "ND", "ori_shape": (4, 5), "ori_format": "ND"},
                     )
    op_select_format({"shape": (3, 2, 4), "dtype": "float", "format": "ND", "ori_shape": (3, 2, 4), "ori_format": "ND"},
                     {"shape": (1, 4, 5), "dtype": "float", "format": "ND", "ori_shape": (1, 4, 5), "ori_format": "ND"},
                     )
    # dynamic shape
    op_select_format({"shape": (-1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 2, 4), "ori_format": "ND"},
                     {"shape": (7, 4, 5), "dtype": "float16", "format": "ND", "ori_shape": (7, 4, 5), "ori_format": "ND"},
                     )

def test_op_check_supported(test_arg):
    from impl.batch_matmul import check_supported
    input_x1_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (2,3), (3,5)), "dtype": 'float16'}
    input_x2_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (3,5), (4,5)), "dtype": 'float16'}
    check_supported(input_x1_dynamic, input_x2_dynamic)

    input_x1 = {"ori_shape": (2, 16, 16), "shape": (2, 1, 1, 16, 16),"dtype": 'float16'}
    input_x2 = {"ori_shape": (2, 16, 16), "shape": (2, 1, 1, 16, 16),"dtype": 'float16'}
    check_supported(input_x1, input_x2)


def test_batch_matmul_generalization(test_arg):
    from impl.dynamic.batch_matmul import batch_matmul_generalization
    input_x1_dynamic = {"ori_shape": (5, 2, 3), "shape": (5, 1, 1, 16, 16), "range": ((4,7), (1,3), (1,3)), "dtype": 'float16', "format": "ND"}
    input_x2_dynamic = {"ori_shape": (5, 3, 5), "shape": (5, 1, 1, 16, 16), "range": ((1,3), (1,3), (1,3)), "dtype": 'float16', "format": "ND"}
    output_dynamic = {"ori_shape": (5, 2, 5), "shape": (5, 1, 1, 16, 16), "range": ((4,7), (1,3), (1,3)), "dtype": 'float16', "format": "ND"}
    bias_dynamic = {"ori_shape": (5, ), "dtype": 'float16', "shape": (5,), "format": "ND", "ori_format": "ND", "range": (1, 48)}
    batch_matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_z=output_dynamic,
                                trans_a=False, trans_b=False, kernel_name="batchmatmul_generalization",
                                generalize_config={"mode": "keep_rank"})


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

    x1, x2, bias, y, trans_a, trans_b = param
    return f'a_{str_list(x1["ori_shape"])}_b_{str_list(x2["ori_shape"])}_c_{str_list(bias["ori_shape"] if bias else None)}_trans_{str_bool(trans_a)}_{str_bool(trans_b)}'

def gen_cases_by_shape_and_range(case):
    [shape_x1, range_x1, trans_a], [shape_x2, range_x2, trans_b], [shape_bias, range_bias], [shape_out, range_out] = case
    params = []

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
        param = [x1, x2, bias, y, trans_a, trans_b]
        params.append({"params": param, "case_name": f'dynamic_batchmatmul_{gen_kernel_name(param)}', "expect": "success"})

    return params

normal_case = [
    # [((-1, -1, -1, -1), ((1, 196608), (1, 196608), (1, 196608), (1, 196608)), 'float16', 'FRACTAL_NZ', False),
    #  ((3, 8, -1, 1), ((3, 3), (8, 8), (1, 196608), (1, 1)), 'float16', 'FRACTAL_NZ', True),
    #  (None, None, None, None),
    #  ((3, 8, -1, 1), ((3, 3), (8, 8), (1, 196608), (1, 1)), 'float16', 'FRACTAL_NZ'),
    #  "dynamic_batch_matmul_succ_case0"],
    # IMPORTANT: default tiling
    [((-1, -1, -1, -1), ((1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 196608)), 'float16', 'FRACTAL_NZ', False),
     ((-1, -1, -1, -1), ((1, 196608), (1, 196608), (1, 196608), (1, 196608)), 'float16', 'FRACTAL_NZ', True),
     (None, None, None, None),
     ((-1, -1, -1, -1), ((1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 196608)), 'float16', 'FRACTAL_NZ'),
     "dynamic_batch_matmul_succ_case1"],
]

def cus_ceil(v, base):
    if v == -1:
        return v
    return ceil(v / base)

def c0(dtype):
    return {'float16': 16, 'float32': 8}[dtype]

def shape_from_nd_to(shape, dtype, format):
    if format == 'ND':
        return shape
    elif format == 'FRACTAL_NZ':
        batches = shape[:-2]
        a, b = shape[-2:]
        return list(batches) + [cus_ceil(b, c0(dtype)), cus_ceil(a, 16), 16, c0(dtype)]
    raise RuntimeError(f"not support shape from nd to {format}")

def range_from_nd_to(range, dtype, format):
    if format == 'ND':
        return range
    elif format == 'FRACTAL_NZ':
        batches = range[:-2]
        [a0, a1], [b0, b1] = range[-2:]
        return list(batches) + [[cus_ceil(b0, c0(dtype)), cus_ceil(b1, c0(dtype))], [cus_ceil(a0, 16), cus_ceil(a1, 16)], [16, 16], [c0(dtype), c0(dtype)]]
    raise RuntimeError(f"not support shape from nd to {format}")

def gen_batch_matmul_dynamic_normally(params):
    [x1_ori_shape, x1_ori_range, x1_dtype, x1_format, trans_a], [x2_ori_shape, x2_ori_range, x2_dtype, x2_format, trans_b], [bias_ori_shape, bias_ori_range, bias_dtype, bias_format], [y_ori_shape, y_ori_range, y_dtype, y_format], case_name = params

    x1 = {"ori_shape": x1_ori_shape, "dtype": x1_dtype, "shape": shape_from_nd_to(x1_ori_shape, x1_dtype, x1_format),
          "format": x1_format , "ori_format": "ND", "range": range_from_nd_to(x1_ori_range, x1_dtype, x1_format)
    }
    x2 = {"ori_shape": x2_ori_shape, "dtype": x2_dtype, "shape": shape_from_nd_to(x2_ori_shape, x2_dtype, x2_format),
          "format": x2_format , "ori_format": "ND", "range": range_from_nd_to(x2_ori_range, x2_dtype, x2_format),
    }
    y = {"ori_shape": y_ori_shape, "dtype": y_dtype, "shape": shape_from_nd_to(y_ori_shape, y_dtype, y_format),
         "format": y_format , "ori_format": "ND", "range": range_from_nd_to(y_ori_range, y_dtype, y_format),
    }

    if bias_ori_shape:
        bias = {"ori_shape": bias_ori_shape, "dtype": bias_dtype, "shape": shape_from_nd_to(bias_ori_shape, bias_dtype, bias_format),
                "format": bias_format, "ori_format": "ND", "range": range_from_nd_to(bias_ori_range, bias_dtype, bias_format)}
    else:
        bias = None

    return {
        "params": [x1, x2, bias, y, trans_a, trans_b],
        "case_name": case_name,
        "expect": "success"
    }
for case in normal_case:
    ut_case.add_case(case=gen_batch_matmul_dynamic_normally(case))

for case in matmul_case:
    ut_case.add_case("Ascend910A", gen_batch_matmul_dynamic(*case))

for case in common_cases:
    for param in gen_cases_by_shape_and_range(case):
        ut_case.add_case("Ascend910A", param)

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_batch_matmul_generalization)

if __name__ == "__main__":
    ut_case.run(["Ascend310", "Ascend910"])
