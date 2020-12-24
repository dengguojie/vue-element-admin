#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import flags
from op_test_frame.ut import OpUT

CUBE_BLOCK = 16
ut_case = OpUT("BatchMatmul", "impl.dynamic.batch_matmul", "batch_matmul")


# batch_range, m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, batchb_flag, case_name
matmul_case = [
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, False, "dynamic_batch_matmul_v2_succcase0"),
    (((1, 5), ), (1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, True, True, True, "dynamic_batch_matmul_v2_succcase1"),
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
     input_x1 = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (2,3), (3,5)), "dtype": 'float16'}
     input_x2 = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,3), (3,5), (4,5)), "dtype": 'float16'}
     check_supported(input_x1, input_x2)

for case in matmul_case:
    ut_case.add_case("Ascend910A", gen_batch_matmul_dynamic(*case))
    ut_case.add_cust_test_func(test_func=test_op_select_format)
    ut_case.add_cust_test_func(test_func=test_op_check_supported)

if __name__ == "__main__":
    with te.op.dynamic():
        ut_case.run()