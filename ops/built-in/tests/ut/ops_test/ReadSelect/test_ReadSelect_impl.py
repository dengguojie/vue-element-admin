#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
from op_test_frame.ut import OpUT
from impl.read_select import get_op_support_info

ut_case = OpUT("ReadSelect", None, None)

case1 = {"params": [{"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0],"src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"},
                    {"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0], "src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0],"src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"},
                    {"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0], "src_in_flag":"DDR", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0],"src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"},
                    {"shape": (1,16,13,56,16), "dtype": "float16", "valid_shape":(1,16,11,56,16), "slice_offset":[0,0,1,1,0], "src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(1,16,13,56,16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0],"src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"},
                    {"shape": (2, 4, 10, 5, 16), "dtype": "float16", "valid_shape":(), "slice_offset":[0,0,1,1,0], "src_in_flag":"L1", "format":"NC1HWC0", "ori_shape":(2, 4, 10, 5, 16), "ori_format":"NC1HWC0"}],
         "case_name": "read_select_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)


def test_get_op_support_info(test_arg):
    input_x = {
        "shape": (2, 4, 10, 5, 16),
        "dtype": "float16", "valid_shape": (),
        "src_in_flag": "L1", "format": "NC1HWC0", "ori_shape": (2, 64, 10, 5),
        "ori_format": "NCHW"}
    output_x = {
        "shape": (2, 4, 9, 4, 16),
        "dtype": "float16", "valid_shape": (),
        "src_in_flag": "L1", "format": "NC1HWC0", "ori_shape": (2, 64, 9, 4),
        "ori_format": "NCHW"}
    support_info = get_op_support_info(input_x, output_x, [], 4)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 2
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 1
        axis = input_list[0].get("axis")
        assert len(axis) == 1
        assert axis[0] in (0, 1)


    input_x = {
        "shape": (2, 4, 10, 5, 16),
        "dtype": "float16", "valid_shape": (),
        "slice_offset": [0, 0, 1, 1, 0],
        "src_in_flag": "L1", "format": "ND", "ori_shape": (2, 4, 10, 5, 16),
        "ori_format": "ND"}
    output_x = {
        "shape": (2, 4, 9, 4, 16),
        "dtype": "float16", "valid_shape": (),
        "slice_offset": [0, 0, 1, 1, 0],
        "src_in_flag": "L1", "format": "ND", "ori_shape": (2, 4, 9, 4, 16),
        "ori_format": "ND"}
    support_info = get_op_support_info(input_x, output_x, [], 4)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 3
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 1
        axis = input_list[0].get("axis")
        assert len(axis) == 1
        assert axis[0] in (0, 1, 4)

    input_x = {
        "shape": (2, 4, 10, 5, 16),
        "dtype": "float16", "valid_shape": (),
        "slice_offset": [0, 0, 1, 1, 0],
        "src_in_flag": "L1", "format": "NC1HWC0", "ori_shape": (2, 4, 10, 5, 16),
        "ori_format": "NC1HWC0"}
    output_x = {
        "shape": (1, 3, 9, 4, 15),
        "dtype": "float16", "valid_shape": (),
        "slice_offset": [1, 1, 1, 1, 1],
        "src_in_flag": "L1", "format": "NC1HWC0", "ori_shape": (1, 3, 9, 4, 15),
        "ori_format": "NC1HWC0"}
    support_info = get_op_support_info(input_x, output_x, [], 4)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 0

ut_case.add_cust_test_func(test_func=test_get_op_support_info)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
