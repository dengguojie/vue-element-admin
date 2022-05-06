#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AttentionQKVGradX", "impl.attention_qkv_grad_x", "attention_qkv_grad_x")


attention_qkv_gradx_testcases = [
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (64, 768, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_qkv_gradx_12288_1024_1024_float16_NZ_NZ"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="ND", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (64, 768, 16, 16), "float16", "ND", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradx_12288_1024_1024_float16_ND_NZ"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="ND"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (64, 768, 16, 16), "float16", "FRACTAL_NZ", "ND", RuntimeError,
     "bert_large_attention_qkv_gradx_12288_1024_1024_float16_NZ_ND"),
    # m=12288, k=1024, n=1024, dtype="float32", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (64, 768, 16, 16), "float32", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradx_12288_1024_1024_float32_NZ_NZ"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (48, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 48, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (64, 768, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradx_12288_1024_1024_float16_NZ_NZ_x_shape_inconsistant"),
    # m=12288, k=1024, n=1024, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 768, 16, 16), (64, 48, 16, 16), (64, 64, 16, 16),
     (64, 64, 16, 16), (64, 768, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_qkv_gradx_12288_1024_1024_float16_NZ_NZ_kernel_inconsistant"),
]


def _gen_case(params):
    ln_dx_shape, query_dx_shape, key_dw_shape, value_dw_shape, kernel_q_shape, kernel_k_shape, kernel_v_shape, \
        out_shape, dtype, x_format, kernel_format, expect_result, kernel_name = params
    ln_dx_ori_shape = (ln_dx_shape[1] * ln_dx_shape[2], ln_dx_shape[0] * ln_dx_shape[3])
    ln_dx = {"shape": ln_dx_shape, "dtype": dtype, "format": x_format, "ori_shape": ln_dx_ori_shape, "ori_format": "ND"}
    query_dx_ori_shape = (query_dx_shape[1] * query_dx_shape[2], query_dx_shape[0] * query_dx_shape[3])
    query_dx = {"shape": query_dx_shape, "dtype": dtype, "format": x_format, "ori_shape": query_dx_ori_shape, "ori_format": "ND"}
    key_dw_ori_shape = (key_dw_shape[1] * key_dw_shape[2], key_dw_shape[0] * key_dw_shape[3])
    key_dw = {"shape": key_dw_shape, "dtype": dtype, "format": x_format, "ori_shape": key_dw_ori_shape, "ori_format": "ND"}
    value_dw_ori_shape = (value_dw_shape[1] * value_dw_shape[2], value_dw_shape[0] * value_dw_shape[3])
    value_dw = {"shape": value_dw_shape, "dtype": dtype, "format": x_format, "ori_shape": value_dw_ori_shape, "ori_format": "ND"}
    kernel_q_ori_shape = (kernel_q_shape[1] * kernel_q_shape[2], kernel_q_shape[0] * kernel_q_shape[3])
    kernel_query = {"shape": kernel_q_shape, "dtype": dtype, "format": kernel_format,
                    "ori_shape": kernel_q_ori_shape, "ori_format": "ND"}
    kernel_k_ori_shape = (kernel_k_shape[1] * kernel_k_shape[2], kernel_k_shape[0] * kernel_k_shape[3])
    kernel_key = {"shape": kernel_k_shape, "dtype": dtype, "format": kernel_format,
                  "ori_shape": kernel_k_ori_shape, "ori_format": "ND"}
    kernel_v_ori_shape = (kernel_v_shape[1] * kernel_v_shape[2], kernel_v_shape[0] * kernel_v_shape[3])
    kernel_value = {"shape": kernel_v_shape, "dtype": dtype, "format": kernel_format,
                    "ori_shape": kernel_v_ori_shape, "ori_format": "ND"}

    out_shape_ori = (query_dx_ori_shape[0], kernel_q_ori_shape[1])
    dx = {"shape": out_shape, "dtype": dtype, "format": x_format, "ori_shape": out_shape_ori, "ori_format": "ND"}
    testcase = {
        "params": [ln_dx, query_dx, key_dw, value_dw, kernel_query, kernel_key, kernel_value, dx],
        "case_name": kernel_name,
        "expect": expect_result,
        "support_expect": True,
    }
    return testcase

for case in attention_qkv_gradx_testcases:
    ut_case.add_case(["Ascend910A"], _gen_case(case))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
