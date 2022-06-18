from op_test_frame.ut import OpUT
from tbe.common.platform.platform_info import set_current_compile_soc_info
from impl.swin_transformer_ln_qkv import swin_transformer_ln_qkv, check_supported
ut_case = OpUT("SwinTransformerLnQKV", "impl.swin_transformer_ln_qkv", "swin_transformer_ln_qkv")


def create_case(input_type, batch, m, k, n, head_num, head_dim, seq_length, shifts, epsilon):
    input_x = {"shape": (batch, k // 16, m // 16, 16, 16), "format": "FRACTAL_NZ", "dtype": input_type,
               "ori_shape": (batch, m, k), "ori_format": "ND"}
    input_gamma = {"shape": (k,), "format": "ND", "dtype": input_type,
                   "ori_shape": (k,), "ori_format": "ND"}
    input_beta = {"shape": (k,), "format": "ND", "dtype": input_type,
                  "ori_shape": (k,), "ori_format": "ND"}
    input_weight = {"shape": (k // 16, n // 16, 16, 16), "format": "ND", "dtype": input_type,
                    "ori_shape": (k, n), "ori_format": "ND"}
    input_bias = {"shape": (n,), "format": "ND", "dtype": input_type,
                  "ori_shape": (n,), "ori_format": "ND"}
    query_output = {"shape": (batch * m // seq_length, head_num, head_dim // 16, seq_length // 16, 16, 16),
                    "format": "FRACTAL_NZ", "dtype": input_type,
                    "ori_shape": (batch * m // seq_length, head_num, seq_length, head_dim), "ori_format": "ND"}
    key_output = {"shape": (batch * m // seq_length, head_num, head_dim // 16, seq_length // 16, 16, 16),
                  "format": "FRACTAL_NZ", "dtype": input_type,
                  "ori_shape": (batch * m // seq_length, head_num, seq_length, head_dim), "ori_format": "ND"}
    value_output = {"shape": (batch * m // seq_length, head_num, head_dim // 16, seq_length // 16, 16, 16),
                    "format": "FRACTAL_NZ", "dtype": input_type,
                    "ori_shape": (batch * m // seq_length, head_num, seq_length, head_dim), "ori_format": "ND"}
    return [input_x, input_gamma, input_beta, input_weight, input_bias,
            query_output, key_output, value_output,
            head_num, head_dim, seq_length, shifts, epsilon]


def add_case_310p3(test_arg):
    set_current_compile_soc_info("Ascend310P3")
    input_type = "float16"
    case_all = ((8, 9216, 128, 384, 4, 32, 144, (0, 0, 0, 0), 0.00001),
                (8, 9216, 128, 384, 4, 32, 144, (0, 6, 6, 0), 0.00001))
    for index, case_0 in enumerate(case_all):
        batch, m, k, n, head_num, head_dim, seq_length, shifts, epsilon = case_0
        params = create_case(input_type, batch, m, k, n, head_num, head_dim, seq_length, shifts, epsilon)
        swin_transformer_ln_qkv(*params)
        check_supported(*params)
    params = create_case(input_type, *case_all[0])
    params[11] = [0, 0, 0]
    check_supported(*params)
    params = create_case(input_type, *case_all[0])
    params[12] = -0.1
    check_supported(*params)
    params = create_case(input_type, *case_all[0])
    params[0]["ori_shape"] = (1, 1)
    check_supported(*params)
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(["Ascend910A"], test_func=add_case_310p3)


if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
