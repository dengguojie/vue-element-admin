#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
import te
from te import tvm
from op_test_frame.ut import OpUT
from impl.trans_data import trans_data_compute
from impl.trans_data import get_op_support_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")

def test_transdata_1(test_arg):
    try:
        input_tensor = tvm.placeholder([100, 1, 112, 16], name='input_tensor', dtype="float16",
                                       attrs={'ori_format': 'NCHW'})
        output_tensor = {'dtype': 'float16', 'format': 'NCHW', 'ori_format': 'NCHW', 'ori_shape': [100, 1, 7, 16], 'shape': [100, 1, 7, 16]}
        trans_data_compute(input_tensor, output_tensor, "NC1HWC0", "NCHW")
    except RuntimeError:
        pass

ut_case.add_cust_test_func(test_func=test_transdata_1)

def test_get_op_support_info_nhwc_nc1hwc0_1(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 8, 12, 24), "dtype": "float16", "format": "NCHW", "ori_shape": (4, 8, 12, 24),
         "ori_format": "NCHW"},
        {"shape": (4, 2, 8, 12, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 2, 8, 12, 16),
         "ori_format": "NC1HWC0"},
        "NCHW", "NC1HWC0")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 0

def test_get_op_support_info_nc1hwc0_nhwc_2(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 2, 8, 12, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 2, 8, 12, 16),
         "ori_format": "NC1HWC0"},
        {"shape": (4, 8, 12, 24), "dtype": "float16", "format": "ND", "ori_shape": (4, 8, 12, 24),
         "ori_format": "ND"},
        "NC1HWC0", "ND")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 0

def test_get_op_support_info_nd_nz_3(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (23, 33), "dtype": "float16", "format": "ND", "ori_shape": (23, 33),
         "ori_format": "ND"},
        {"shape": (3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (3, 2, 16, 16),
         "ori_format": "FRACTAL_NZ"},
        "ND", "FRACTAL_NZ")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 1

def test_get_op_support_info_nd_nz_4(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (7, 23, 33), "dtype": "float16", "format": "ND", "ori_shape": (7, 23, 33),
         "ori_format": "ND"},
        {"shape": (7, 3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (7, 3, 2, 16, 16),
         "ori_format": "FRACTAL_NZ"},
        "ND", "FRACTAL_NZ")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 0

def test_get_op_support_info_nz_nd_5(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (3, 2, 16, 16),
         "ori_format": "FRACTAL_NZ"},
        {"shape": (23, 33), "dtype": "float16", "format": "ND", "ori_shape": (23, 33),
         "ori_format": "ND"},
        "FRACTAL_NZ", "ND")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 1

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 0

def test_get_op_support_info_nz_nd_6(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (7, 3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (7, 3, 2, 16, 16),
         "ori_format": "FRACTAL_NZ"},
        {"shape": (7, 23, 33), "dtype": "float16", "format": "ND", "ori_shape": (7, 23, 33),
         "ori_format": "ND"},
        "FRACTAL_NZ", "ND")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 0

def test_get_op_support_info_ndhwc_ndc1hwc0_7(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 6, 8, 12, 24), "dtype": "float16", "format": "NDHWC", "ori_shape": (4, 6, 8, 12, 24),
         "ori_format": "NDHWC"},
        {"shape": (4, 2, 8, 12, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (4, 2, 8, 12, 16),
         "ori_format": "NDC1HWC0"},
         "NDHWC", "NDC1HWC0")
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    assert len(split_maps) == 0

ut_case.add_cust_test_func(test_func=test_get_op_support_info_nhwc_nc1hwc0_1)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nc1hwc0_nhwc_2)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_nz_3)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_nz_4)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_nd_5)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_nd_6)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_ndhwc_ndc1hwc0_7)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)