# -*- coding:utf-8 -*-
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("roi_extractor")


def calc_expect_func():
    pass


def generate_case(x_shape, roi_shape, y_shape, dtype, output_size=7, vrange=None, precision_standard=None,
                  iscal=False, iserror=False, calc_func=calc_expect_func):
    if vrange is None:
        vrange = [-10.0, 10.0]

    def shape_to_5hd(shape, dtype):
        n, c, h, w = shape
        c0 = 8 if dtype == "float32" else 16
        c1 = (c + c0 - 1) // c0
        return (n, c1, h, w, c0)

    x_shape_5hd = [shape_to_5hd(x_sp, dtype) for x_sp in x_shape]
    output_shape_5hd = (x_shape_5hd[0][0], x_shape_5hd[0][1], output_size, output_size, x_shape_5hd[0][-1])

    num_lvls = len(x_shape)
    case = {}
    x_data = [None for _ in range(num_lvls)]
    roi_data = []
    if iscal:
        case["calc_expect_func"] = calc_func
        for i in range(num_lvls):
            x1 = np.zeros(x_shape_5hd[i])
            x_data.append(x1)
        roi_data = np.zeros(roi_shape)

    x = []
    for i in range(num_lvls):
        x.append(
            {
                "dtype": dtype,
                "ori_shape": x_shape[i],
                "shape": x_shape_5hd[i],
                "ori_format": "NCHW",
                "format": "NC1HWC0",
                "value": x_data[i],
                "param_type": "input",
            })
    rois = {
        "dtype": dtype,
        "ori_shape": roi_shape,
        "shape": roi_shape,
        "ori_format": "ND",
        "format": "ND",
        "value": roi_data,
        "param_type": "input",
    }
    y = {
        "dtype": dtype,
        "ori_shape": y_shape,
        "shape": output_shape_5hd,
        "ori_format": "NCHW",
        "format": "NC1HWC0",
        "param_type": "output",
    }
    case["params"] = [x, rois, None, y, 56, 0, (1./4, 1./8, 1./16, 1./32), output_size, output_size, 0, "avg", True]
    if iserror:
        case["expect"] = RuntimeError
    if precision_standard is None:
        case["precision_standard"] = precision_info.PrecisionStandard(0.005, 0.005)
    else:
        case["precision_standard"] = precision_info.PrecisionStandard(precision_standard[0], precision_standard[1])
    return case


# 'pylint: disable=unused-argument
def test_710(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.roi_extractor import roi_extractor

    te_set_version("Ascend710")

    x_shape = [(1, 16, 16, 16), (1, 16, 9, 9), (1, 16, 8, 8), (1, 16, 4, 5)]
    roi_shape = [1000, 5]
    y_shape = (1000, 16, 7, 7)
    dtype = "float16"
    output_size = 7

    def shape_to_5hd(shape):
        n, c, h, w = shape
        c0 = 16
        c1 = (c + c0 - 1) // c0
        return n, c1, h, w, c0

    x_shape_5hd = [shape_to_5hd(x_sp) for x_sp in x_shape]
    output_shape_5hd = (x_shape_5hd[0][0], x_shape_5hd[0][1], output_size, output_size, x_shape_5hd[0][-1])

    num_lvls = len(x_shape)

    x = []
    for i in range(num_lvls):
        x.append(
            {
                "dtype": dtype,
                "ori_shape": x_shape[i],
                "shape": x_shape_5hd[i],
                "ori_format": "NCHW",
                "format": "NC1HWC0",
                "param_type": "input",
            })
    rois = {
        "dtype": dtype,
        "ori_shape": roi_shape,
        "shape": roi_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "input",
    }
    y = {
        "dtype": dtype,
        "ori_shape": y_shape,
        "shape": output_shape_5hd,
        "ori_format": "NCHW",
        "format": "NC1HWC0",
        "param_type": "output",
    }
    params = [x, rois, None, y, 56, 0, (1./4, 1./8, 1./16, 1./32), 7, 7, 0, "avg", True, "roi_extractor"]
    roi_extractor(*params)
    te_set_version("Ascend310")


ut_case.add_case("Ascend310", generate_case([(1, 16, 78, 87), (1, 16, 19, 39), (1, 16, 18, 58), (1, 16, 24, 35)],
                                            (100, 5), (100, 16, 7, 7), "float16", 7))
ut_case.add_case("Ascend310", generate_case([(1, 16, 21, 65), (1, 16, 21, 22), (1, 16, 10, 18), (1, 16, 24, 15)],
                                            (1000, 5), (1000, 16, 7, 7), "float16", 7))
ut_case.add_case("Ascend310", generate_case([(1, 16, 21, 65), (1, 16, 21, 22), (1, 16, 10, 18), (1, 16, 24, 15)],
                                            (1000, 5), (1000, 16, 7, 7), "float32", 7))
ut_case.add_case("Ascend310", generate_case([(1, 16, 16, 16), (1, 16, 9, 9), (1, 16, 8, 8), (1, 16, 4, 5)], (1000, 5),
                                            (1000, 16, 7, 7), "float32", 7))
ut_case.add_case("Ascend310", generate_case([(1, 16, 16, 16), (1, 16, 9, 9), (1, 16, 8, 8), (1, 16, 4, 5)], (1000, 4),
                                            (1000, 16, 7, 7), "float32", 7, iserror=True))
ut_case.add_cust_test_func("Ascend310", test_710)
ut_case.run("Ascend310")
