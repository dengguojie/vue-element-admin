# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT("ctc_loss_v2_grad")

def generate_case(input0_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, input6_shape,
                  output0_shape):
    dtype_int = "int32"
    dtype_float = "float32"

    case = {}
    input0 = {
        "dtype":dtype_float,
        "ori_shape":input0_shape,
        "shape":input0_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
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
    input5 = {
        "dtype":dtype_float,
        "ori_shape":input5_shape,
        "shape":input5_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    input6 = {
        "dtype":dtype_float,
        "ori_shape":input6_shape,
        "shape":input6_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    output0 = {
        "dtype":dtype_float,
        "ori_shape":output0_shape,
        "shape":output0_shape,
        "ori_format":"ND",
        "format":"ND",
        "param_type":"input"
    }
    blank = 0
    reduction = "mean"
    zero_infinity = False

    case["params"] = [input0, input1, input2, input3, input4, input5, input6, output0, blank, reduction,
                      zero_infinity]

    return case


ut_case.add_case("Ascend910A",
                 case=generate_case((1,), (195, 1, 41), (1, 74), (1,), (1,), (1,), (1, 195, 149), (195, 1, 41)))
ut_case.add_case("Ascend910A",
                 case=generate_case((1,), (195, 1, 41), (74,), (1,), (1,), (1,), (1, 195, 149), (195, 1, 41)))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)