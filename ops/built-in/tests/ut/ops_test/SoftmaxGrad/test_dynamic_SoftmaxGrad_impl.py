#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SoftmaxGrad", "impl.dynamic.softmax_grad", "softmax_grad")

def gen_softmaxgrad_case(dynamic_input_shapes, ori_input_shapes, dtype,
                    case_name_val, impl_mode, expect, input_format="ND"):
    inputs = (
        {"shape": dynamic_input_shapes,
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * len(dynamic_input_shapes)},
    )
    outputs = (
        {"shape": [-1],
         "dtype": dtype,
         "ori_shape": ori_input_shapes,
         "ori_format": input_format,
         "format": input_format,
         'range': [[1, 100000]] * 1},
    )

    return {"params": [inputs[0],
                       inputs[0],
                       outputs[0]],
            'addition_params': {'impl_mode': impl_mode},
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}

ut_case.add_case(["Ascend910A"],
                 gen_softmaxgrad_case((-1, -1, -1),
                                      (16, 16, 16),
                                      "float16", "dynamic_softmax_grad_1", "high_precision", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_softmaxgrad_case((-1, -1, -1),
                                      (16, 16, 16),
                                      "float16", "dynamic_softmax_grad_2", "high_performance", "success"))

from impl.dynamic.softmax_grad import op_select_format
def test_dynamic_softmax_grad_op_select_format_001(test_arg):
    """
    test_dynamic_softmax_grad_op_select_format_001
    """
    op_select_format(
        {
            "shape": (32, 32, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        {
            "shape": (32, 32, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        {
            "shape": (32, 32, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        -1,
    )
ut_case.add_cust_test_func(test_func=test_dynamic_softmax_grad_op_select_format_001)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
