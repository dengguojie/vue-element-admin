import sys

from op_test_frame.ut import OpUT
from impl.mat_mul import op_select_format

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

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_1)
ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_2)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910A"])
    sys.exit(0)
