#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("NormalizeBBox", "impl.normalize_bbox", "normalize_bbox")

def gen_normalize_bbox_case(boxes, shape_hw, dtype, reversed_box,
                            case_name_val, expect):
    return {"params": [{"shape": boxes, "dtype": dtype, "ori_shape": boxes,
                        "ori_format": "ND", "format": "ND"},
                       {"shape": shape_hw, "dtype": "int32", "ori_shape": shape_hw,
                        "ori_format": "ND", "format": "ND"},
                       {"shape": boxes, "dtype": dtype, "ori_shape": boxes,
                        "ori_format": "ND", "format": "ND"},
                       reversed_box],
            "case_name": case_name_val,
            "expect": expect,
            "support_expect": True}


def gen_normalize_bbox_precision_case(boxes, shape_hw, dtype, reversed_box, precision):
    return {"params": [{"shape": boxes, "dtype": dtype, "format": "ND", "ori_shape": boxes,
                        "ori_format": "ND", "param_type": "input", "value_range": [1.0, 1.0]},
                       {"shape": shape_hw, "dtype": "int32", "format": "ND", "ori_shape": shape_hw,
                        "ori_format": "ND", "param_type": "input", "value_range": [1.0, 10.0]},
                       {"shape": boxes, "dtype": dtype, "format": "ND", "ori_shape": boxes,
                        "ori_format": "ND", "param_type": "output"},
                        reversed_box],
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(precision, precision)}

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 16), (25, 3), "float16",
                 True, "case_1", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600000), (1, 3), "float16",
                 True, "case_2", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 15), (25, 3), "float16",
                 True, "case_3", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600001), (1, 3), "float16",
                 True, "case_4", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 16), (25, 3), "float32",
                 True, "case_5", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600000), (1, 3), "float32",
                 True, "case_6", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 15), (25, 3), "float32",
                 True, "case_7", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 1600001), (1, 3), "float32",
                 True, "case_8", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 4), (25, 3), "float16",
                 False, "case_9", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600000, 4), (1, 3), "float16",
                 False, "case_10", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 5, 4), (25, 3), "float16",
                 False, "case_11", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600001, 4), (1, 3), "float16",
                 False, "case_12", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 4, 4), (25, 3), "float32",
                 False, "case_13", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600000, 4), (1, 3), "float32",
                 False, "case_14", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((25, 5, 4), (25, 3), "float32",
                 False, "case_15", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 1600001, 4), (1, 3), "float32",
                 False, "case_16", "success"))

ut_case.add_case(["Ascend910"],
                 gen_normalize_bbox_case((25, 4, 15), (25, 3), "float32",
                 True, "case_17", "success"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 16001, 5, 4), (1, 3), "float32",
                                         False, "err_1", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 16001, 4), (1, 2, 3), "float16",
                                         False, "err_2", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 16001, 5), (1, 3), "float32",
                                         False, "err_3", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 6, 5), (1, 3), "float32",
                                         True, "err_4", "RuntimeError"))

ut_case.add_case(["Ascend310"],
                 gen_normalize_bbox_case((1, 4, 5), (1, 2), "float32",
                                         True, "err_5", "RuntimeError"))


def calc_expect_func(boxes, shape_hw, y, reversed_box):
    x1_shape = boxes["shape"]
    x2_shape = shape_hw["shape"]
    x1_dtype = boxes["dtype"]

    if reversed_box is True:
        shape_data = x1_shape[-1]
    else:
        shape_data = x1_shape[1]
    data_a = []
    data_c = []
    data_b = shape_hw["value"]
    if reversed_box is True:
        for i in range(0, x1_shape[0]):
            data_a_1 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_h = data_b[i][0]
            data_c_1 = data_a_1 / data_b_h

            data_a_2 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_w = data_b[i][1]
            data_c_2 = data_a_2 / data_b_w

            data_a_3 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_h = data_b[i][0]
            data_c_3 = data_a_3 / data_b_h

            data_a_4 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_w = data_b[i][1]
            data_c_4 = data_a_4 / data_b_w

            data_a_temp = np.concatenate((data_a_1, data_a_2, data_a_3, data_a_4),
                                         axis=0)

            data_a = np.concatenate((data_a, data_a_temp), axis=0)
            data_c_temp = np.concatenate((data_c_1, data_c_2, data_c_3, data_c_4),
                                         axis=0)
            data_c = np.concatenate((data_c, data_c_temp), axis=0)
    else:
        for i in range(0, x1_shape[0]):
            data_a_1 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_h = data_b[i][0]
            data_c_1 = data_a_1 / data_b_h

            data_a_2 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_w = data_b[i][1]
            data_c_2 = data_a_2 / data_b_w

            data_a_3 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_h = data_b[i][0]
            data_c_3 = data_a_3 / data_b_h

            data_a_4 = np.random.uniform(1, 1, shape_data).astype(x1_dtype)
            data_b_w = data_b[i][1]
            data_c_4 = data_a_4 / data_b_w

            data_a_temp = np.concatenate((data_a_1, data_a_2, data_a_3, data_a_4),
                                         axis=0)
            data_a_temp = data_a_temp.reshape(x1_shape[2], x1_shape[1]).T
            data_a_temp = data_a_temp.reshape(-1)

            data_a = np.concatenate((data_a, data_a_temp), axis=0)
            data_c_temp = np.concatenate((data_c_1, data_c_2, data_c_3, data_c_4),
                                         axis=0)
            data_c_temp = data_c_temp.reshape(x1_shape[2], x1_shape[1]).T
            data_c_temp = data_c_temp.reshape(-1)
            data_c = np.concatenate((data_c, data_c_temp), axis=0)

    result = data_c.astype(x1_dtype).reshape(x1_shape)

    return result


ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((25, 4, 16), (25, 3), "float16", True, 0.001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((3, 4, 4), (3, 3), "float16", True, 0.001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((3, 4, 80010), (3, 3), "float32", True, 0.0001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((22, 4, 140), (22, 3), "float32", True, 0.0001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((25, 4, 16), (25, 3), "float16", False, 0.001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((3, 4, 4), (3, 3), "float16", False, 0.001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((3, 4, 80010), (3, 3), "float32", False, 0.0001))

ut_case.add_precision_case("all", gen_normalize_bbox_precision_case((22, 4, 140), (22, 3), "float32", False, 0.0001))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
