# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("ctc_loss_v2")

def generate_case(input1_shape, input2_shape, input3_shape, input4_shape, output1_shape, output2_shape):
    dtype_int = "int32"
    dtype_float = "float32"

    case = {}

    input1 = {
        "dtype":dtype_float,
        "ori_shape":input1_shape,
        "shape":input1_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    input2 = {
        "dtype":dtype_int,
        "ori_shape":input2_shape,
        "shape":input2_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    input3 = {
        "dtype":dtype_int,
        "ori_shape":input3_shape,
        "shape":input3_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    input4 = {
        "dtype":dtype_int,
        "ori_shape":input4_shape,
        "shape":input4_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    output1 = {
        "dtype":dtype_float,
        "ori_shape":output1_shape,
        "shape":output1_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"output"
    }
    output2 = {
        "dtype":dtype_float,
        "ori_shape":output2_shape,
        "shape":output2_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"output"
    }
    blank = 0
    reduction = "mean"
    zero_infinity = False

    case["params"] = [input1, input2, input3, input4, output1, output2, blank, reduction, zero_infinity]

    return case


ut_case.add_case("Ascend910A", case=generate_case((195, 1, 41), (1, 74), (1,), (1,), (1, 195, 149), (1,)))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)