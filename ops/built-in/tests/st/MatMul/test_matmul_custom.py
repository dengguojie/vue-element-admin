import sys

from op_test_frame.ut import OpUT
from impl.mat_mul import op_select_format
from impl.dynamic.mat_mul import mat_mul

ut_case = OpUT("MatMul", None, None)

def test_op_select_format_matmul_1(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (2, 1, 16, 16), "ori_shape": (16, 32)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float16", "shape": (1, 2, 16, 16), "ori_shape": (32, 16)}
    bias = {"format": "ND","ori_format": "ND", "dtype": "float32", "shape": (16,), "ori_shape": (16,)}
    op_select_format(x1, x2, bias, impl_mode="keep_bias_fp32")

def test_op_select_format_matmul_2(test_arg):
    x1 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float32", "shape": (2, 1, 16, 16), "ori_shape": (16, 32)}
    x2 = {"format": "FRACTAL_NZ","ori_format": "ND", "dtype": "float32", "shape": (1, 2, 16, 16), "ori_shape": (32, 16)}
    bias = {"format": "ND","ori_format": "ND", "dtype": "float32", "shape": (16,), "ori_shape": (16,)}
    op_select_format(x1, x2, bias, impl_mode="keep_bias_fp32")

def test_op_select_format_matmul_3(test_arg):
    input_x1 = {"shape": (16, -1, 16, 16), "ori_shape": (-1, 256), "format": "FRACTAL_NZ", "ori_format": "ND",
                "dtype": "float16",  "range": ((16, 16), (1, 256), (16, 16), (16, 16)), "param_name": "input_x1"}
    input_x2 = {"shape": (-1, 16, 16, 16), "ori_shape": (-1, 256), "format": "FRACTAL_ZN_RNN", "ori_format": "ND",
                "dtype": "float16", "range": ((1, 256), (16, 16), (16, 16), (16, 16)), "param_name": "input_x2"}
    output_y = {"shape": (-1, -1, 16, 16), "ori_shape": (-1, -1), "format": "FRACTAL_NZ", "ori_format": "ND",
                "dtype": "float16", "range": ((1, 256), (1, 256), (16, 16), (16, 16)), "param_name": "output_y"}
    import tbe
    import tbe.common.context.op_info as operator_info
    with tbe.common.context.op_context.OpContext("dynamic"):
        op_info = operator_info.OpInfo("MatMul", "MatMul")
        tbe.common.context.op_context.get_context().add_op_info(op_info)
        mat_mul(input_x1, input_x2, bias=None, output_y=output_y, trans_a=False, trans_b=True)

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_1)

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_2)

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_3)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910A"])
    sys.exit(0)
