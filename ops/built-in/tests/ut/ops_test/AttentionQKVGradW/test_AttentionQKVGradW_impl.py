#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AttentionQKVGradW", "impl.attention_qkv_grad_w", "attention_qkv_grad_w")


attention_qkv_gradx_testcases = [
    # m=1024, k=12288, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (1024,), (1024,), (1024,), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_qkv_gradw_1024_12288_1024_float16_NZ_NZ"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="ND", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (1024,), (1024,), (1024,), "float16", "ND", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradw_1024_12288_1024_float16_ND_NZ"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="ND"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (1024,), (1024,), (1024,), "float16", "FRACTAL_NZ", "ND", RuntimeError,
     "bert_large_attention_qkv_gradw_1024_12288_1024_float16_NZ_ND"),
    # m=12288, k=1024, n=1024, dtype="float32", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (1024,), (1024,), (1024,), "float32", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradw_1024_12288_1024_float32_NZ_NZ"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (48, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (48, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (1024,), (1024,), (1024,), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradw_1024_12288_1024_float16_NZ_NZ_kernel_inconsistant"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (32, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (1024,), (1024,), (1024,), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradw_1024_12288_1024_float16_NZ_NZ_dw_out_inconsistant"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (512,), (1024,), (1024,), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradw_1024_12288_1024_float16_NZ_NZ_dbias_out_inconsistant"),
]


def _gen_case(params):
    x_shape, query_dx_shape, key_dw_shape, value_dw_shape, dw_q_shape, dw_k_shape, dw_v_shape, dbias_q_shape, \
        dbias_k_shape, dbias_v_shape, dtype, x_format, kernel_format, expect_result, kernel_name = params
    x_ori_shape = (x_shape[1] * x_shape[2], x_shape[0] * x_shape[3])
    x = {"shape": x_shape, "dtype": dtype, "format": x_format, "ori_shape": x_ori_shape, "ori_format": "ND"}
    query_dx_ori_shape = (query_dx_shape[1] * query_dx_shape[2], query_dx_shape[0] * query_dx_shape[3])
    query_dx = {"shape": query_dx_shape, "dtype": dtype, "format": x_format, "ori_shape": query_dx_ori_shape, "ori_format": "ND"}
    key_dw_ori_shape = (key_dw_shape[1] * key_dw_shape[2], key_dw_shape[0] * key_dw_shape[3])
    key_dw = {"shape": key_dw_shape, "dtype": dtype, "format": x_format, "ori_shape": key_dw_ori_shape, "ori_format": "ND"}
    value_dw_ori_shape = (value_dw_shape[1] * value_dw_shape[2], value_dw_shape[0] * value_dw_shape[3])
    value_dw = {"shape": value_dw_shape, "dtype": dtype, "format": x_format, "ori_shape": value_dw_ori_shape, "ori_format": "ND"}
    dw_q_ori_shape = (dw_q_shape[1] * dw_q_shape[2], dw_q_shape[0] * dw_q_shape[3])
    dw_query = {"shape": dw_q_shape, "dtype": dtype, "format": kernel_format,
                    "ori_shape": dw_q_ori_shape, "ori_format": "ND"}
    dw_k_ori_shape = (dw_k_shape[1] * dw_k_shape[2], dw_k_shape[0] * dw_k_shape[3])
    dw_key = {"shape": dw_k_shape, "dtype": dtype, "format": kernel_format,
                  "ori_shape": dw_k_ori_shape, "ori_format": "ND"}
    dw_v_ori_shape = (dw_v_shape[1] * dw_v_shape[2], dw_v_shape[0] * dw_v_shape[3])
    dw_value = {"shape": dw_v_shape, "dtype": dtype, "format": kernel_format,
                    "ori_shape": dw_v_ori_shape, "ori_format": "ND"}
    
    dbias_query = {"shape": dbias_q_shape, "dtype": dtype, "format": "ND",
                    "ori_shape": dbias_q_shape, "ori_format": "ND"}
    dbias_key = {"shape": dbias_k_shape, "dtype": dtype, "format": "ND",
                    "ori_shape": dbias_k_shape, "ori_format": "ND"}
    dbias_value = {"shape": dbias_v_shape, "dtype": dtype, "format": "ND",
                    "ori_shape": dbias_v_shape, "ori_format": "ND"}

    testcase = {
        "params": [x, query_dx, key_dw, value_dw, dw_query, dw_key, dw_value, dbias_query, dbias_key, dbias_value],
        "case_name": kernel_name,
        "expect": expect_result,
        "support_expect": True,
    }
    return testcase

for case in attention_qkv_gradx_testcases:
    ut_case.add_case(["Ascend910A"], _gen_case(case))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
