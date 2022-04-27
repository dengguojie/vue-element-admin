#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AttentionLnQKV", "impl.attention_ln_qkv", "attention_ln_qkv")


attention_ln_qkv_testcases = [
    # batch=24, seq_len=512, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (24, 16, 4, 32, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_ln_qkv_24_512_1024_16_float16_NZ_NZ"),
    # batch=24, seq_len=512, hidden_size=768, n=12, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((48, 768, 16, 16), (48, 48, 16, 16), (48, 48, 16, 16), (48, 48, 16, 16), (768,), (768,), (12, 64), (12, 64),
     (12, 64), (24, 12, 4, 32, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_ln_qkv_24_512_768_12_float16_NZ_NZ"),
    # batch=24, seq_len=256, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 384, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (24, 16, 4, 16, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_ln_qkv_24_256_1024_16_float16_NZ_NZ"),
    # batch=32, seq_len=384, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (32, 16, 4, 24, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_ln_qkv_32_384_1024_16_float16_NZ_NZ"),
    # batch=48, seq_len=32, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 96, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (48, 16, 4, 2, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_ln_qkv_48_32_1024_16_float16_NZ_NZ"),
    # batch=6, seq_len=256, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 96, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (6, 16, 4, 16, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", "success",
     "bert_large_attention_ln_qkv_6_256_1024_16_float16_NZ_NZ"),
    # batch=24, seq_len=512, hidden_size=1024, n=16, dtype="float16", x_format="ND", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (24, 16, 4, 32, 16, 16), "float16", "ND", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_ln_qkv_24_512_1024_16_float16_ND_NZ"),
    # batch=24, seq_len=512, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="ND"
    ((64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (24, 16, 4, 32, 16, 16), "float16", "FRACTAL_NZ", "ND", RuntimeError,
     "bert_large_attention_ln_qkv_24_512_1024_16_float16_NZ_ND"),
    # batch=24, seq_len=512, hidden_size=1024, n=16, dtype="float32", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (16, 64),
     (16, 64), (24, 16, 4, 32, 16, 16), "float32", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_ln_qkv_24_512_1024_16_float32_NZ_NZ"),
    # batch=24, seq_len=512, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (48, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (12, 64), (16, 64),
     (16, 64), (24, 16, 4, 32, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_ln_qkv_24_512_1024_16_float16_NZ_NZ_kernel_inconsistant"),
    # batch=24, seq_len=512, hidden_size=1024, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((64, 768, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (64, 64, 16, 16), (1024,), (1024,), (16, 64), (),
     (16, 64), (24, 16, 4, 32, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_ln_qkv_24_512_1024_16_float16_NZ_NZ_bias_flag_inconsistant"),
    # batch=24, seq_len=512, hidden_size=4096, n=16, dtype="float16", x_format="FRACTAL_NZ", kernel_format="FRACTAL_NZ"
    ((256, 768, 16, 16), (64, 256, 16, 16), (64, 256, 16, 16), (64, 256, 16, 16), (4096,), (4096,), (16, 64), (16, 64),
     (16, 64), (24, 64, 4, 32, 16, 16), "float16", "FRACTAL_NZ", "FRACTAL_NZ", RuntimeError,
     "bert_large_attention_ln_qkv_24_512_4096_16_float16_NZ_NZ_unsupported_k")
]


def _gen_case(params):
    x_shape, kernel_q_shape, kernel_k_shape, kernel_v_shape, gamma_shape, beta_shape, bias_q_shape, bias_k_shape, bias_v_shape, \
        out_shape, dtype, x_format, kernel_format, expect_result, kernel_name = params
    x_ori_shape = (x_shape[1] * x_shape[2], x_shape[0] * x_shape[3])
    x = {"shape": x_shape, "dtype": dtype, "format": x_format, "ori_shape": x_ori_shape, "ori_format": "ND"}
    kernel_q_ori_shape = (kernel_q_shape[1] * kernel_q_shape[2], kernel_q_shape[0] * kernel_q_shape[3])
    kernel_query = {"shape": kernel_q_shape, "dtype": dtype, "format": kernel_format,
                    "ori_shape": kernel_q_ori_shape, "ori_format": "ND"}
    kernel_k_ori_shape = (kernel_k_shape[1] * kernel_k_shape[2], kernel_k_shape[0] * kernel_k_shape[3])
    kernel_key = {"shape": kernel_k_shape, "dtype": dtype, "format": kernel_format,
                  "ori_shape": kernel_k_ori_shape, "ori_format": "ND"}
    kernel_v_ori_shape = (kernel_v_shape[1] * kernel_v_shape[2], kernel_v_shape[0] * kernel_v_shape[3])
    kernel_value = {"shape": kernel_v_shape, "dtype": dtype, "format": kernel_format,
                    "ori_shape": kernel_v_ori_shape, "ori_format": "ND"}
    gamma = {"shape": gamma_shape, "dtype": dtype, "format": "NHWC", "ori_shape": gamma_shape,
             "ori_format": "NHWC"}
    beta = {"shape": beta_shape, "dtype": dtype, "format": "NHWC", "ori_shape": beta_shape,
             "ori_format": "NHWC"}
    if len(bias_q_shape) != 0:
        bias_query = {"shape": bias_q_shape, "dtype": dtype, "format": "ND", "ori_shape": bias_q_shape,
                      "ori_format": "ND"}
    else:
        bias_query = None
    if len(bias_k_shape) != 0:
        bias_key = {"shape": bias_k_shape, "dtype": dtype, "format": "ND", "ori_shape": bias_k_shape,
                      "ori_format": "ND"}
    else:
        bias_key = None
    if len(bias_v_shape) != 0:
        bias_value = {"shape": bias_v_shape, "dtype": dtype, "format": "ND", "ori_shape": bias_v_shape,
                      "ori_format": "ND"}
    else:
        bias_value = None

    norm = {"shape": x_shape, "dtype": dtype, "format": x_format, "ori_shape": x_ori_shape, "ori_format": "ND"}
    out_shape_ori = (x_ori_shape[0], kernel_q_ori_shape[1])
    query_output = {"shape": out_shape, "dtype": dtype, "format": x_format, "ori_shape": out_shape_ori, "ori_format": "ND"}
    key_output = {"shape": out_shape, "dtype": dtype, "format": x_format, "ori_shape": out_shape_ori, "ori_format": "ND"}
    value_output = {"shape": out_shape, "dtype": dtype, "format": x_format, "ori_shape": out_shape_ori, "ori_format": "ND"}
    mean_shape = (x_ori_shape[0],)
    mean = {"shape": mean_shape, "dtype": dtype, "format": "NHWC", "ori_shape": mean_shape, "ori_format": "NHWC"}
    variance = {"shape": mean_shape, "dtype": dtype, "format": "NHWC", "ori_shape": mean_shape, "ori_format": "NHWC"}
    testcase = {
        "params": [x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value, norm, query_output,
                   key_output, value_output, mean, variance],
        "case_name": kernel_name,
        "expect": expect_result,
        "support_expect": True,
    }
    return testcase

for case in attention_ln_qkv_testcases:
    ut_case.add_case(["Ascend910A", "Ascend710"], _gen_case(case))

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])

