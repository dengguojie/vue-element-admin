#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
from op_test_frame.ut import OpUT
from tbe.common.context import op_context

CUBE_BLOCK = 16
ut_case = OpUT("BatchMatMul", "impl.dynamic.batch_matmul", "batch_matmul")


#shape_a, shape_b, m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, case_name
matmul_case_succ = [
    ((-1, -1), (-1, -1), (1, 3), (4, 15), (4, 15), "float16", "float16", "FRACTAL_NZ", False, False, False, "fuzzy_compile_matmul_case0"),
]

#shape_a, shape_b, m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, case_name, expect_result
generalize_case = [
    ((2, 3), (3, 4), (1, 3), (1, 3), (1, 3), "float16", "float16", "FRACTAL_NZ", False, False, False, "generalize_matmul_case0")
]

def gen_matmul_fuzzy_case(shape_a, shape_b, m_range, k_range, n_range, src_dtype, dst_dtype,
                          format, trans_a, trans_b, bias_flag, case_name):
    """
    gen the case for ut test
    """
    block_range = [[CUBE_BLOCK, CUBE_BLOCK], [CUBE_BLOCK, CUBE_BLOCK]]
    if len(k_range)  == 4:
        mk_range = k_range[:2]
        nk_range = k_range[2:]
    else:
        mk_range = nk_range = k_range
    x1_range = [m_range, mk_range] if trans_a else [mk_range, m_range]
    x1_range += block_range
    x2_range = [nk_range, n_range] if trans_b else [n_range, nk_range]
    x2_range += block_range
    y_range = [n_range, m_range] + block_range
    shape_m = shape_a[1] if trans_a else shape_a[0]
    shape_n = shape_a[0] if trans_a else shape_b[1]
    x1 = {"ori_shape": shape_a, "dtype": src_dtype, "shape": (-1, -1, CUBE_BLOCK, CUBE_BLOCK),
          "format": format , "ori_format": "ND", "range": x1_range
    }
    x2 = {"ori_shape": shape_b, "dtype": src_dtype, "shape": (-1, -1, CUBE_BLOCK, CUBE_BLOCK),
          "format": format , "ori_format": "ND", "range": x2_range
    }
    y = {"ori_shape": (shape_m, shape_n), "dtype": src_dtype, "shape": (-1, -1, CUBE_BLOCK, CUBE_BLOCK),
         "format": format , "ori_format": "ND", "range": y_range
    }

    if bias_flag:
        bias_n_range = [16*i for i in n_range]
        bias = {"ori_shape": (-1, ), "dtype": dst_dtype, "shape": (-1,),
                "format": "ND", "ori_format": "ND", "range": (bias_n_range,)}
    else:
        bias = None

    return {
        "params": [x1, x2, bias, y, trans_a, trans_b],
        "case_name": case_name,
        "expect": "success" 
    }


def _generate_missing_support_info(range_m, range_k, range_n, batch_range):
    missing_support_info = [
        {
            "inputs": [
                {
                    "index": 0, 
                    "tensor": [
                        {
                            "shape": [-1, -1, -1, -1],
                            "range": [batch_range[0], batch_range[1], range_m, range_k]
                        }
                    ]
                },
                {
                    "index": 0, 
                    "tensor": [
                        {
                            "shape": [-1, -1],
                            "range": [range_k, range_n]
                        }
                    ]
                }
            ],
            "outputs": [
                {
                    "index": 0, 
                    "tensor": [
                        {
                            "shape": [-1, -1],
                            "range": [range_m, range_n]
                        }
                    ]
                }
            ]
        }
    ]
    return missing_support_info



for case in matmul_case_succ:
    ut_case.add_case("Ascend910A", gen_matmul_fuzzy_case(*case))


if __name__ == "__main__":
    with op_context.OpContext("dynamic"):
        context = op_context.get_context()
        context.set_build_type("fuzzily_build")
        batch_range = [[2, 3], [4, 7]]
        missing_support_info = _generate_missing_support_info([1, 3], [4, 15], [4, 15], batch_range)
        context.add_addition("missing_support_info", json.dumps(missing_support_info))
        ut_case.run(soc="Ascend910A")