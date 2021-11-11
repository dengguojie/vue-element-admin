#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from te import tvm
ut_case = OpUT("PriorBoxD", None, None)

def prior_box_cce(feature_shape, img_shape, data_h_shape, data_w_shape, box_shape, res_shape, dtype, dformat, min_size, max_size, img_h = 0, img_w = 0, step_h = 0.0, step_w = 0.0, flip = True, clip = False, offset = 0.5, variance = [0.1], expect="success",case_name = "prior_box"):

    return {"params": [{"shape": feature_shape, "dtype":dtype, "format":dformat,"ori_shape": feature_shape,"ori_format":dformat},
                       {"shape":img_shape, "dtype":dtype, "format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": data_h_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": data_w_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": box_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": box_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": res_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       min_size, max_size, img_h, img_w, step_h, step_w, flip, clip, offset, variance],
            "case_name": case_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


case1 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, False, 0.5, [0.1, 0.1, 0.2, 0.2],"success","prior_box_1")
case2 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2],"success", "prior_box_2")
case3 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2],"success", "prior_box_3")

case4 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [-162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2],RuntimeError, "prior_box_4")
case5 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [113.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2],RuntimeError, "prior_box_5")
case6 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2],RuntimeError, "prior_box_6")
case7 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, -64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2],RuntimeError, "prior_box_3")
case8 = prior_box_cce((3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, False, 0.5, [0.1, 0.1, 0.2, 0.2], RuntimeError, "prior_box_8")
case9 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1],"success", "prior_box_9")
case10 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], RuntimeError, "prior_box_10")
case11 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0, 213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], RuntimeError, "prior_box_11")
case12 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], -300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], RuntimeError, "prior_box_12")
case13 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, -300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], RuntimeError, "prior_box_13")
case14 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 0, 0, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], "success", "prior_box_14")
case15 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, -64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], RuntimeError, "prior_box_15")
case16 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, 0, 0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], "success", "prior_box_16")
case17 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, -0.1, 0.2, 0.2], RuntimeError, "prior_box_17")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend910A"], case14)
ut_case.add_case(["Ascend310", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend910A"], case14)
ut_case.add_case(["Ascend310", "Ascend910A"], case15)
ut_case.add_case(["Ascend310", "Ascend910A"], case16)
ut_case.add_case(["Ascend310", "Ascend910A"], case17)

op_list = []
shape = (5, 1, 1, 1)
tensor_in = tvm.placeholder(shape, name='tensor_in', dtype="float16")
tensor_in_list = tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_list")
for i in range(4):
    op_list.append(tensor_in_list)


def test_get_compute_axis(test_arg):
    from impl.prior_box_d import get_compute_axis
    get_compute_axis(
        op_list, {
            "y": 0,
            "top_data_xmin": 0,
            "top_data_ymin": 0,
            "top_data_xmax": 1,
            "top_data_ymax": 1,
            "top_data_res1": 2,
            "top_data_res2": 2,
            "top_data_res3": 2,
            "top_data_res4": 2,
            "top_data": 3
        })


def test_multicore_factor_calculate(test_arg):
    from impl.prior_box_d import _multicore_factor_calculate
    _multicore_factor_calculate([4, 2, 5, 5, 6, 4], 8)
    _multicore_factor_calculate([1, 4, 5, 5, 6, 4], 8)
    _multicore_factor_calculate([1, 1, 1, 1, 6, 4], 8)
    _multicore_factor_calculate([1, 1, 1, 1, 1, 4], 8)
    _multicore_factor_calculate([3, 3, 5, 5, 3, 3], 9)
    _multicore_factor_calculate([3, 3, 5, 5, 3, 3], 27)
    _multicore_factor_calculate([3, 3, 1, 5, 3, 3], 135)


def test_tiling_factor_calculate(test_arg):
    from impl.prior_box_d import _tiling_factor_calculate
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 0, 1, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 0, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 0, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 128, 1, 1, 1, 1], 0, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 128, 128, 1, 1, 1], 0, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 128, 128, 128, 1, 1], 0, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 128, 128, 128, 128, 1], 0, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 0, 1, "float16", 128, 1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 0, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 0, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 1, 1, 1, 1], 0, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 128, 1, 1, 1], 0, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 128, 128, 1, 1], 0, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 128, 128, 128, 1], 0, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 1, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 1, 1, 1, 1], 1, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 128, 1, 1, 1], 1, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 128, 128, 1, 1], 1, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 128, 128, 128, 1], 1, 128, "float16", 128, 1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 2, 1, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 2, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 128, 1, 1], 2, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 128, 128, 1], 2, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 128, 1, 1, 1, 1], 2, 1, "float16", 128, 3)
    _tiling_factor_calculate([1, 128, 1, 1, 128, 1], 2, 128, "float16", 128, 3)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 2, 1, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 1, 1, 1, 1], 2, 1, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 1, 128, 1, 1], 2, 1, "float16", 128, 1)
    _tiling_factor_calculate([1, 128, 1, 128, 128, 1], 2, 1, "float16", 128, 1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 3, 1, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 3, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 128, 1], 3, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 4, 1, "float16", 128, -1)
    _tiling_factor_calculate([1, 1, 1, 1, 1, 1], 4, 128, "float16", 128, -1)
    _tiling_factor_calculate([1, 2, 5, 5, 6, 4], 5, 1, "float16", 128, -1)


ut_case.add_cust_test_func(test_func=test_get_compute_axis)
ut_case.add_cust_test_func(test_func=test_multicore_factor_calculate)
ut_case.add_cust_test_func(test_func=test_tiling_factor_calculate)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)