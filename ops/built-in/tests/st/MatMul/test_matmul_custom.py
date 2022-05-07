import sys

from op_test_frame.ut import OpUT
from impl.mat_mul import op_select_format
from impl.dynamic.mat_mul import mat_mul
from impl.dynamic.mat_mul import matmul_generalization

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

def test_matmul_fuzzy_single_op_generalization():
    input_x1_dynamic = {"ori_shape": (5, 2, 3), "shape": (5, 2, 3), "range": ((4,7), (1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    input_x2_dynamic = {"ori_shape": (5, 3, 5), "shape": (5, 3, 5), "range": ((1,3), (1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    output_dynamic = {"ori_shape": (5, 2, 5), "shape": (5, 2, 5), "range": ((4,7), (1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    bias_dynamic = None
    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, offset_w={}, output_y=output_dynamic,
                                   trans_a=False, trans_b=False, offset_x=0, kernel_name="batchmatmul_generalization",
                                   generalize_config={"mode": "keep_rank", "single_op": "true"})


def test_matmul_fuzzy_binary_generalization():
    input_x1_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,None), (1,None), (1,None)), "dtype": 'float16', "format": "FRACTAL_NZ", "ori_format" : "ND"}
    input_x2_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,None), (1,None), (1,None)), "dtype": 'float16', "format": "FRACTAL_NZ", "ori_format" : "ND"}
    output_dynamic = {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1, 16, 16), "range": ((1,None), (1,None), (1,None)), "dtype": 'float16', "format": "FRACTAL_NZ", "ori_format" : "ND"}
    bias_dynamic = None
    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, offset_w={}, output_y=output_dynamic,
                                   trans_a=False, trans_b=False, offset_x=0, kernel_name="batchmatmul_generalization",
                                   generalize_config={"mode": "all_shape", "single_op": "true"})

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_1)

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_2)

ut_case.add_cust_test_func(test_func=test_op_select_format_matmul_3)

if __name__ == '__main__':
    test_matmul_fuzzy_single_op_generalization()
    test_matmul_fuzzy_binary_generalization()
    ut_case.run(["Ascend310", "Ascend910A"])
    sys.exit(0)
