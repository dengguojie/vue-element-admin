#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np


def gen_prelu_case(shape, weight_shape, case_name, dtype="float32"):
    return {"params": [{"shape": shape, "dtype": dtype, "format": "ND",
                        "ori_shape": shape, "ori_format": "ND"},
                       {"shape": weight_shape, "dtype": dtype,
                        "format": "ND", "ori_shape": weight_shape,
                        "ori_format": "ND"},
                       {"shape": shape, "dtype": dtype, "format": "ND",
                        "ori_shape": shape, "ori_format": "ND"}],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


ut_case = OpUT("Prelu", "impl.prelu", None)
platform = ["Ascend310", "Ascend710", "Ascend910"]
ut_case.add_case(platform, gen_prelu_case((1, 2, 4), (1,), "prelu_1"))
ut_case.add_case(platform, gen_prelu_case((16, 16), (16,), "prelu_2"))
ut_case.add_case(platform, gen_prelu_case((32, 2, 4, 16), (1,), "prelu_3"))
ut_case.add_case(platform, gen_prelu_case((32, 2, 4, 16), (1, ), "prelu_4"))
ut_case.add_case(platform, gen_prelu_case((1, 2), (1, ), "prelu_5"))


def calc_expect_func(input_x, weight, output_y):
    data = input_x["value"]
    shape = input_x["shape"]
    print(data)
    print(weight["value"])
    if weight["shape"] == (1,):
        slop = weight["value"][0]
        output = np.where(data > 0, data, data * slop)
        return [output, ]
    elif input_x["format"] == "FRACTAL_NZ":
        # C1 NN0 C0 -> C1C0 xx
        weight_tmp = np.zeros(shape[0]*shape[-1], )
        idx = 0
        for i in weight["value"].flatten():
            weight_tmp[idx] = i
            idx += 1
        input_tmp = \
            np.transpose(data, (0, 3, 1, 2)).reshape(shape[0]*shape[-1], -1)
        output = np.zeros(input_tmp.shape, input_x["dtype"])
        for i in range(input_tmp.shape[0]):
            output[i] = np.where(input_tmp[i] > 0,
                                 input_tmp[i], input_tmp[i] * weight_tmp[i])
        # C1C0 xx -> C1 xx C0
        output = \
            np.reshape(output, (shape[0], shape[-1], shape[1], shape[2])
                       ).transpose((0, 2, 3, 1))
        print(output)
        return [output, ]


def gen_prelu_precision_case(input_shape, input_format, weight_shape,
                             dtype, expect="success", kernel_name="prelu"):
    return {"params": [{"shape": input_shape, "dtype": dtype,
                        "format": input_format, "ori_shape": input_shape,
                        "ori_format": "ND", "param_type": "input",
                        "value_range": [-10, 10]},
                       {"shape": weight_shape, "dtype": dtype,
                        "format": "ND", "ori_shape": weight_shape,
                        "ori_format": "ND", "param_type": "input",
                        "value_range": [0, 1]},
                       {"shape": input_shape, "dtype": dtype,
                        "format": input_format, "ori_shape": input_shape,
                        "ori_format": "ND", "param_type": "output"}],
            "case_name": kernel_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True,
            "calc_expect_func": calc_expect_func}


'''
# precision_case
ut_case.add_precision_case("all", gen_prelu_precision_case((1, 1, 16, 16),
   "FRACTAL_NZ", (1,), "float16"))
ut_case.add_precision_case("all", gen_prelu_precision_case((1, 1, 16, 16),
   "FRACTAL_NZ", (6,), "float16"))
ut_case.add_precision_case("all", gen_prelu_precision_case((2, 1, 16, 16),
   "FRACTAL_NZ", (17,), "float16"))
'''

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
