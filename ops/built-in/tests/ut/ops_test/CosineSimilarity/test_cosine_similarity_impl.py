# # -*- coding:utf-8 -*-
import sys
import torch
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("CosineSimilarity")


#pylint: disable=unused-argument
def calc_expect_func(input_x1, input_x2, output_y, dim=1):
    input_x1_trans = torch.from_numpy(input_x1["value"])
    input_x2_trans = torch.from_numpy(input_x2["value"])
    cos = torch.nn.CosineSimilarity(dim, eps=1e-8)
    res = cos(input_x1_trans, input_x2_trans)
    res = res.numpy()
    return res


ut_case.add_case("all", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 8), "shape": (32, 36, 5, 8),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5, 9), "shape": (32, 5, 9),
         "param_type": "output"}],
    "case_name": "test_cosine_similarity_1",
    "expect": RuntimeError
})

ut_case.add_case("all", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5, 9), "shape": (32, 5, 9),
         "param_type": "output"}, 8],
    "case_name": "test_cosine_similarity_2",
    "expect": RuntimeError
})

ut_case.add_precision_case("Ascend910A", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5, 9), "shape": (32, 5, 9),
         "param_type": "output"}],
    "case_name": "test_cosine_similarity_3",
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})

ut_case.add_precision_case("Ascend910A", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5), "shape": (32, 36, 5),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5), "shape": (32, 36, 5),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
         "param_type": "output"}],
    "case_name": "test_cosine_similarity_4",
    "calc_expect_func": calc_expect_func,
    "expect": "success"
})

ut_case.add_precision_case("Ascend310", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5, 9), "shape": (32, 36, 5, 9),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5, 9), "shape": (32, 5, 9),
         "param_type": "output"}],
    "case_name": "test_cosine_similarity_5",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005),
    "expect": "success"
})

ut_case.add_precision_case("Ascend310", {
    "params": [
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5), "shape": (32, 36, 5),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 36, 5), "shape": (32, 36, 5),
         "param_type": "input"},
        {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
         "param_type": "output"}],
    "case_name": "test_cosine_similarity_6",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005),
    "expect": "success"
})