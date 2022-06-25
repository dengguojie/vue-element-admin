from op_test_frame.ut import OpUT
from tbe.common.platform.platform_info import set_current_compile_soc_info
from impl.swin_attention_ffn import swin_attention_ffn, check_supported
ut_case = OpUT("SwinAttentionFFN", "impl.swin_attention_ffn", "swin_attention_ffn")


def create_case(input_type, batch, seq_num, seq_len, head_len, shifts):
    input_x = {"shape": (batch * seq_num, head_len // 16, seq_len // 16, 16, 16), "format": "FRACTAL_NZ", "dtype": input_type,
               "ori_shape": (batch * seq_num, seq_len, head_len), "ori_format": "ND"}
    input_weight = {"shape": (head_len // 16, head_len // 16, 16, 16), "format": "ND", "dtype": input_type,
                    "ori_shape": (head_len, head_len), "ori_format": "ND"}
    input_bias = {"shape": (head_len,), "format": "ND", "dtype": input_type,
                  "ori_shape": (head_len,), "ori_format": "ND"}
    output_y = {"shape": (batch, head_len // 16, seq_num * seq_len // 16, 16, 16),
                "format": "FRACTAL_NZ", "dtype": input_type,
                "ori_shape": (batch, head_len, seq_num * seq_len), "ori_format": "ND"}

    return [input_x, input_weight, input_bias, output_y, shifts]


def add_case_310p3(test_arg):
    set_current_compile_soc_info("Ascend310P3")
    input_type = "float16"
    case_all = ((8, 64, 144, 128, (0, 0, 0, 0)),
                (8, 64, 144, 128, (0, 6, 6, 0)))
    for index, case_0 in enumerate(case_all):
        batch, seq_num, seq_len, head_len, shifts = case_0
        params = create_case(input_type, batch, seq_num, seq_len, head_len, shifts)
        swin_attention_ffn(*params)
        check_supported(*params)
    set_current_compile_soc_info(test_arg)


ut_case.add_cust_test_func(["Ascend910A"], test_func=add_case_310p3)


if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
