#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
from sys import flags
from op_test_frame.ut import OpUT
from math import ceil
from unittest.mock import MagicMock
from unittest.mock import patch

from te.platform.cce_conf import te_set_version
from impl.dynamic.batch_matmul_v2 import get_op_support_info
CUBE_BLOCK = 16
ut_case = OpUT("BatchMatMulV2", "impl.dynamic.batch_matmul_v2", "batch_matmul_v2")

vals = {("CORE_NUM", ): 48,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 196608,
        ("L0C_SIZE", ): 131072,
        ("Compiler_arch",): "dav-c220-cube",
        ("AICORE_TYPE",):"AiCore",
        ("Intrinsic_fix_pipe_l0c2out",):True,
        }
def side_effects(*args):
    return vals[args]


# batch_range, m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, batchb_flag, case_name
matmul_case = [
    #(((1, None), ), (1, None), (1, None), (1, None), "float16", "float16", "ND", False, False, False, False,"unrange_nd_dynamic_batch_matmul_v2"),
    (((1, None), ), (1, None), (1, None), (1, None), "float16", "float16", "NZ", False, False, False, False,"unrange_nz_dynamic_batch_matmul_v2"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, False, "dynamic_batch_matmul_v2_succcase0"),
    # TODO: temporarily block util the base package is updated, which is newer than 0713
    #(((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, True, True, True, "dynamic_batch_matmul_v2_succcase1"),
    # dtype error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float32", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase0", "dtype"),
    # ori_shape error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase2", "x1_orishape"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase3", "x2_orishape"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase4", "dynamic_mode"),
    # range error
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase5", "x1_range"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase6", "x2_range"),
    (((1, 2), ), (63, 66), (1, 2), (4, 7), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_errorcase7"),
    (((32, 2048), ), (1, 8), (1, 12), (1, 12), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_success1")
]
# batch_range, m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, batchb_flag, case_name
matmul_case_920 = [
    (((1, 1), ), (4, 4), (4, 4), (4, 4), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_case1"),
    (((32, 2048), ), (1, 8), (1, 12), (1, 12), "float16", "float16", "NZ", False, True, False, True, "dynamic_matmul_v2_case2"),
    (((2, 8), ), (8, 16), (8, 15), (4, 8), "float16", "float16", "NZ", True, False, False, False, "dynamic_matmul_v2_case3"),
    (((3, 9), ), (1, 4), (1, 2), (1, 12), "float16", "float16", "NZ", True, True, False, True, "dynamic_matmul_v2_case4"),
    (((1, 256), ), (16, 33), (1, 16), (4, 16), "float16", "float16", "NZ", False, False, False, True, "dynamic_matmul_v2_case5")
]

def gen_batch_matmul_dynamic(batch_range, m_range, k_range, n_range, src_dtype, dst_dtype,
                             format, trans_a, trans_b, bias_flag, batchb_flag, case_name, error_mode=None):
    """
    gen the error case for ut test
    """

    if format == "NZ":
        format = "FRACTAL_NZ"
    block_range = [] if format == "ND" else [[CUBE_BLOCK, CUBE_BLOCK], [CUBE_BLOCK, CUBE_BLOCK]]

    x1_range = [*batch_range, m_range, k_range] if trans_a else [*batch_range, k_range, m_range]
    x1_range += block_range
    x2_range = [k_range, n_range] if trans_b else [n_range, k_range]
    x2_range += block_range
    y_range = [*batch_range, n_range, m_range] + block_range

    if batchb_flag:
        x2_range = [*batch_range] + x2_range

    ori_shape_len_x1 = len(x1_range) if format == "ND" else len(x1_range) - 2
    ori_shape_len_x2 = len(x2_range) if format == "ND" else len(x2_range) - 2
    x1_ori_shape = (-1,) * ori_shape_len_x1
    x2_ori_shape = (-1,) * ori_shape_len_x2
    x1_shape = x1_ori_shape if format == "ND" else x1_ori_shape + (CUBE_BLOCK, CUBE_BLOCK)
    x2_shape = x2_ori_shape if format == "ND" else x2_ori_shape + (CUBE_BLOCK, CUBE_BLOCK)

    x1 = {"ori_shape": x1_ori_shape, "dtype": src_dtype, "shape": x1_shape,
          "format": format , "ori_format": "ND", "range": x1_range
    }
    x2 = {"ori_shape": x2_ori_shape, "dtype": src_dtype, "shape": x2_shape,
          "format": format , "ori_format": "ND", "range": x2_range
    }
    y = {"ori_shape": x1_ori_shape, "dtype": dst_dtype, "shape": x1_shape,
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
        "params": [x1, x2, bias, None, y, trans_a, trans_b],
        "case_name": case_name,
        "expect": expect
    }

def test_op_select_format(test_arg):
    from impl.dynamic.batch_matmul_v2 import op_select_format
    # dynamic shape
    op_select_format({"shape": (-1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 2, 4), "ori_format": "ND"},
                     {"shape": (7, 4, 5), "dtype": "float16", "format": "ND", "ori_shape": (7, 4, 5), "ori_format": "ND"},
                     )

def test_op_check_supported(test_arg):
    from impl.batch_matmul_v2 import check_supported
    input_x1_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (2,3), (3,5)), "dtype": 'float16'}
    input_x2_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (3,5), (4,5)), "dtype": 'float16'}
    check_supported(input_x1_dynamic, input_x2_dynamic)

    input_x1 = {"ori_shape": (2, 16, 16), "shape": (2, 1, 1, 16, 16),"dtype": 'float16'}
    input_x2 = {"ori_shape": (2, 16, 16), "shape": (2, 1, 1, 16, 16),"dtype": 'float16'}
    check_supported(input_x1, input_x2)

def test_op_check_supported_empty_range(test_arg):
    from impl.batch_matmul_v2 import check_supported
    input_x1_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (2,3), (3,5)), "dtype": 'float16'}
    input_x2_dynamic = {"ori_shape": (2, 16, 16), "shape": (2, 1, 1, 16, 16), "range": (), "dtype": 'float16'}
    check_supported(input_x1_dynamic, input_x2_dynamic)

def test_dynamic_batchmamtul_920_mock(test_args):
    for case in matmul_case_920:
        result = gen_batch_matmul_dynamic(*case)
        input_x1, input_x2, bias, _, output_z, trans_a, trans_b = result["params"]
        from impl.dynamic.batch_matmul_v2 import batch_matmul_compute
        from te.tvm.target import cce
        with tbe.common.context.op_context.OpContext("dynamic"):
            from impl.util.platform_adapter import tbe as tbe1
            with tbe1.compute():
                res = batch_matmul_compute(input_x1, input_x2, bias, None, output_z,
                      trans_a, trans_b, 0, "matmul")
            from impl.util.platform_adapter import tvm
            with tvm.target.cce():
                tbe1.auto_schedule(res.get("op_res"))

with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
    with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects)):
        ut_case.add_cust_test_func(test_func=test_dynamic_batchmamtul_920_mock)

for case in matmul_case:
    ut_case.add_case("Ascend910A", gen_batch_matmul_dynamic(*case))

normal_case = [
    [((-1, -1, -1, -1), ((1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 196608)), 'float16', 'FRACTAL_NZ', False),
     ((-1, -1, -1, -1), ((1, 196608), (1, 196608), (1, 196608), (1, 196608)), 'float16', 'FRACTAL_NZ', True),
     (None, None, None, None),
     ((-1, -1, -1, -1), ((1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 196608)), 'float16', 'FRACTAL_NZ'),
     "dynamic_batch_matmul_v2_succ_case1"],
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
        "params": [x1, x2, bias, None, y, trans_a, trans_b, 0],
        "case_name": case_name,
        "expect": "success"
    }

# TODO: temporarily block util the base package is updated, which is newer than 0713
for case in normal_case:
    ut_case.add_case(case=gen_batch_matmul_dynamic_normally(case))

ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_check_supported_empty_range)

def test_get_op_support_info_dynamic_batchmatmul_v2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, -1, 16, 16), "ori_shape": (-1, -1, -1),
          "range": ((1, 6), (16, 48), (16, 48), (16, 16), (16, 16))}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, -1, 16, 16), "ori_shape": (-1, -1, -1),
          "range": ((1, 6), (16, 48), (16, 48), (16, 16), (16, 16))}
    get_op_support_info(x1, x2, trans_b=True)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_batchmatmul_v2)


def test_batch_matmul_v2_fuzzy_generalization(test_arg):
    from impl.dynamic.batch_matmul_v2 import batch_matmul_v2_generalization
    input_x1_dynamic = {"ori_shape": (5, 2, 3), "shape": (5, 1, 1, 16, 16), "range": ((4,7), (1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    input_x2_dynamic = {"ori_shape": (5, 3, 5), "shape": (5, 1, 1, 16, 16), "range": ((1,3), (1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    output_dynamic = {"ori_shape": (5, 2, 5), "shape": (5, 1, 1, 16, 16), "range": ((4,7), (1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    bias_dynamic = {"ori_shape": (5, ), "dtype": 'float16', "shape": (5,), "format": "ND", "ori_format": "ND", "range": (1, 48)}
    batch_matmul_v2_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, offset_w={}, output_z=output_dynamic,
                                   trans_a=False, trans_b=False, offset_x=0, kernel_name="batchmatmul_generalization",
                                   generalize_config={"mode": "keep_rank"})
ut_case.add_cust_test_func(test_func=test_batch_matmul_v2_fuzzy_generalization)

if __name__ == "__main__":
    ut_case.run(["Ascend310", "Ascend910"])