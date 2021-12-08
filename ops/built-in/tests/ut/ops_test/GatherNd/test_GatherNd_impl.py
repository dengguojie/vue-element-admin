"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

GatherNd ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("GatherNd", None, None)


def gen_gather_nd_case(dict_params, dict_indices, dict_y, kernel_name_val, expect):
    return {"params": [dict_params, dict_indices, dict_y],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910"],
                 gen_gather_nd_case(
                     {"shape": (1024, 1024, 16), "dtype": "float32", "ori_shape": (1024, 1024, 16),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (2, 2), "dtype": "int32", "ori_shape": (2, 2),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (2, 16), "dtype": "float32", "ori_shape": (2, 16),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_01", "success"))

ut_case.add_case(["Ascend910"],
                 gen_gather_nd_case(
                     {"shape": (9, 7, 6, 5, 4, 2), "dtype": "float16", "ori_shape": (29, 7, 6, 5, 4, 2),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (3, 2, 3), "dtype": "int32", "ori_shape": (3, 2, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (3, 5, 4, 2), "dtype": "float16", "ori_shape": (3, 5, 4, 2),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_02", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (16, 3, 3), "dtype": "int32", "ori_shape": (16, 3, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (33, 2), "dtype": "int64", "ori_shape": (33, 2),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (33, 2), "dtype": "int32", "ori_shape": (33, 2),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_03", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (16, 3, 3), "dtype": "int32", "ori_shape": (16, 3, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (2, 9), "dtype": "int64", "ori_shape": (2, 9),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (2, 9), "dtype": "int32", "ori_shape": (2, 9),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_04", RuntimeError))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (16, 3, 3), "dtype": "int32", "ori_shape": (16, 3, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (33, 3), "dtype": "int32", "ori_shape": (33, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (33, 3), "dtype": "int32", "ori_shape": (33, 3),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_05", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (127, 4, 2, 2, 2, 2), "dtype": "int32", "ori_shape": (127, 4, 2, 2, 2, 2),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (190000, 6), "dtype": "int32", "ori_shape": (190000, 6),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (190000, 1), "dtype": "int32", "ori_shape": (190000, 1),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_06", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (20000,), "dtype": "float16", "ori_shape": (20000,),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (32, 28, 8, 5, 124, 7, 1), "dtype": "int32", "ori_shape": (32, 28, 8, 5, 124, 7, 1),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (32, 28, 8, 5, 124, 7, 20000), "dtype": "float16",
                      "ori_shape": (32, 28, 8, 5, 124, 7, 20000),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_07", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (2, 34000), "dtype": "float16", "ori_shape": (2, 34000),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (15000, 1), "dtype": "int32", "ori_shape": (15000, 1),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (15000, 34000), "dtype": "float16", "ori_shape": (15000, 34000),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_08", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (1, 7, 4564, 9973), "dtype": "float16", "ori_shape": (1, 7, 4564, 9973),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (1, 7, 4564, 9973), "dtype": "float16", "ori_shape": (1, 7, 4564, 9973),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_09", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (13, 5, 97304), "dtype": "float16", "ori_shape": (13, 5, 97304),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (17, 30, 16, 2), "dtype": "int64", "ori_shape": (17, 30, 16, 2),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (17, 30, 16, 97304), "dtype": "float16", "ori_shape": (17, 30, 16, 97304),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_10", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (8, 95, 32), "dtype": "float16", "ori_shape": (8, 95, 32),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (2, 3, 3, 13, 1), "dtype": "int64", "ori_shape": (2, 3, 3, 13, 1),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (2, 3, 3, 13, 95, 32), "dtype": "float16", "ori_shape": (2, 3, 3, 13, 95, 32),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_11", "success"))

ut_case.add_case("all",
                 gen_gather_nd_case(
                     {"shape": (8, 11, 13, 11, 3), "dtype": "int8", "ori_shape": (8, 11, 13, 11, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (5, 7, 16, 7, 3), "dtype": "int32", "ori_shape": (5, 7, 16, 7, 3),
                      "format": "ND", "ori_format": "ND"},
                     {"shape": (5, 7, 16, 7, 11, 3), "dtype": "int8", "ori_shape": (5, 7, 16, 7, 11, 3),
                      "format": "ND", "ori_format": "ND"},
                     "gather_nd_12", "success"))


def calc_expect_func(dict_data, dict_indices, dict_out):
    shape_indices = dict_indices["value"].shape
    shape_indices_ele = list(shape_indices[:-1]) + list(dict_data["value"].shape[shape_indices[-1]:])
    indices = dict_indices["value"].reshape(np.prod(shape_indices[:-1]), shape_indices[-1])
    res = []
    for i in range(indices.shape[0]):
        aa = tuple(indices[i])
        tmp = dict_data["value"]
        for j in aa:
            tmp = tmp[j]
        res.append(tmp)
    res = np.array(res).astype(dict_data["dtype"]).reshape(shape_indices_ele)
    return res


ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 3, 3), "shape": (16, 3, 3),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 2), "shape": (33, 2),
                "param_type": "input", "value_range": [0, min((16, 3, 3))]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (33, 3), "shape": (33, 3),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9, 7, 6, 5, 4, 2),
                "shape": (9, 7, 6, 5, 4, 2), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 3), "shape": (3, 2, 3),
                "param_type": "input", "value_range": [0, min((9, 7, 6, 5, 4, 2))]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 5, 4, 2),
                "shape": (3, 2, 5, 4, 2), "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 1024, 16),
                "shape": (1024, 1024, 16), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 2), "shape": (3, 2, 2),
                "param_type": "input", "value_range": [0, min((1024, 1024, 16))]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 2, 16), "shape": (3, 2, 16),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (9, 7, 6, 5), "shape": (9, 7, 6, 5),
         "param_type": "input"},
        {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1), "shape": (3, 1),
         "param_type": "input", "value_range": [0, min((9, 7, 6, 5))]},
        {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 7, 6, 5), "shape": (3, 7, 6, 5),
         "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
