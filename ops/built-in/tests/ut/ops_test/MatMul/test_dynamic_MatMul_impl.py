#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

CUBE_BLOCK = 16
ut_case = OpUT("MatMul", "impl.dynamic.mat_mul", "mat_mul")

# succ case
# m_range, k_range, n_range, src_dtype, dst_dtype, format, trans_a, trans_b, bias_flag, case_name, expect_result
matmul_case_succ = [
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, False, False, "dynamic_matmul_succcase0"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, False, True, "dynamic_matmul_succcase1"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", False, True, False, "dynamic_matmul_succcase2"),
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "NZ", True, True, True, "dynamic_matmul_succcase3"),
    ((1, 4), (1, 2, 1, None), (2, 4), "float16", "float16", "NZ", True, True, True, "dynamic_matmul_succcase4"),
]

matmul_case_error = [
    # dtype error
    ((1, 4), (1, 2), (2, 4), "float32", "float16", "NZ", False, False, False, "dynamic_matmul_errorcase0", "dtype"),
    # format error
    ((1, 4), (1, 2), (2, 4), "float16", "float16", "ND", False, False, False, "dynamic_matmul_errorcase1", "format"),
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

    return {
        "params": [x1, x2, bias, None, y, trans_a, trans_b],
        "case_name": case_name,
        "expect": "success"   
    }

def gen_matmul_dynamic_errorcase(m_range, k_range, n_range, src_dtype, dst_dtype,
                                 format, trans_a, trans_b, bias_flag, case_name, error_mode):
    """
    gen the error case for ut test
    """

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

for case in matmul_case_succ:
    ut_case.add_case("Ascend910", gen_matmul_dynamic_succecase(*case))

for error_case in matmul_case_error:
    ut_case.add_case("Ascend910", gen_matmul_dynamic_errorcase(*error_case))

if __name__ == "__main__":
    with te.op.dynamic():
        ut_case.run()