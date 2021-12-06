# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("ApplyAdamV2", None, None)


def get_params_adam(input_shape, data_type):
    data_format = "ND"
    input_shape_1 = (1, )
    input_0 = {"shape": input_shape, "format": data_format, "dtype": data_type,
               "ori_shape": input_shape, "ori_format": data_format}
    input_1 = {"shape": input_shape_1, "format": data_format, "dtype": data_type,
               "ori_shape": input_shape_1, "ori_format": data_format}

    params = [input_0, input_0, input_0, input_1,
              input_1, input_1, input_1, input_0, input_1, input_1, input_1,
              None, input_0, input_0, input_0, "adam"]
    return params

def get_params_mbart_adam(input_shape, data_type):
    data_format = "ND"
    input_shape_1 = (1, )
    input_0 = {"shape": input_shape, "format": data_format, "dtype": data_type,
               "ori_shape": input_shape, "ori_format": data_format}
    input_1 = {"shape": input_shape_1, "format": data_format, "dtype": data_type,
               "ori_shape": input_shape_1, "ori_format": data_format}

    params = [input_0, input_0, input_0, input_1,
              input_1, input_1, input_1, input_0, input_1, input_1, input_1,
              input_1, input_0, input_0, input_0, "mbart_adam"]
    return params

case1 = {
    "params": get_params_adam((1024, ), "float32"),
    "case_name": "bert_adam_0",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {
    "params": get_params_adam((1024, ), "float16"),
    "case_name": "bert_adam_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case3 = {
    "params": get_params_mbart_adam((1024, ), "float32"),
    "case_name": "mbart_adam_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case4 = {
    "params": get_params_mbart_adam((1024, ), "float16"),
    "case_name": "mbart_adam_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
