#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import json

from impl.confusion_transpose_d import get_op_support_info
from op_test_frame.ut import OpUT
ut_case = OpUT("ConfusionTransposeD", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    [0,1,2], [1,2,4], True],
         "case_name": "confusion_transpose_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1], [16,16], True],
         "case_name": "confusion_transpose_d_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [0,1,2,3], [32, 2, 4, 16], True],
         "case_name": "confusion_transpose_d_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [0,1,2,3], [32, 2, 4, 16], True],
         "case_name": "confusion_transpose_d_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [0,1], [1,2], True],
         "case_name": "confusion_transpose_d_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)

def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.confusion_transpose_d import op_select_format
    support_all_dtype = {'float', 'int32', 'int64', 'uint32', 'uint64', 'float16', 'bool', 'uint8', 'int16', 'int8', 'uint16'}
    def check_format(support_dtype, support_format, format_json):
        import json
        obj = json.loads(format_json)

        def check_param_format(param_name):
            result_dtype = set(obj.get(param_name).get("dtype").split(","))
            if result_dtype != support_dtype:
                raise RuntimeError("dtype of {} expected:{} actual:{}".format(param_name, support_dtype, result_dtype))

            result_format = set(obj.get(param_name).get("format").split(","))
            if result_format != support_format:
                raise RuntimeError(
                    "format of {} expected:{} actual:{}".format(param_name, support_format, result_format))

        check_param_format("input0")
        check_param_format("output0")

    result = op_select_format({'shape': (64, 64, 64, 64), 'ori_shape': (64, 64, 64, 64), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              {'shape': (64, 64, 64, 64), 'ori_shape': (64, 64, 64, 64), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              (0, 2, 1, 3), (64, 4096, 64), True, "test_dropout_do_mask_v3_d_op_select_format_4")
    check_format(support_all_dtype,{"FRACTAL_NZ","ND"},result)

ut_case.add_cust_test_func(test_func=test_op_select_format)

def test_get_op_support_info_nd_reshape_transpose_1(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 8, 12, 24, 14), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 8, 12, 24, 14),
         "ori_format": "NHWC"},
        {"shape": (7, 6, 3, 32, 4, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (7, 6, 3, 32, 4, 8),
         "ori_format": "NHWC"},
        [2, 0, 1, 5, 3, 4], [6, 3, 7, 4, 8, 32], False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    assert len(split_maps) == 0

def test_get_op_support_info_nd_reshape_transpose_2(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 8, 12, 24, 14), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 8, 12, 24, 14),
         "ori_format": "NHWC"},
        {"shape": (1, 56, 18, 2, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 56, 2, 9, 2, 64),
         "ori_format": "NHWC"},
        [2, 4, 1, 0, 3], [2, 18, 1, 64, 56], False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    assert len(split_maps) == 1

    input_list = split_maps[0].get("inputList")
    assert len(input_list) == 1
    axis = input_list[0].get("axis")
    assert len(axis) == 1
    assert axis[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 3

def test_get_op_support_info_nd_transpose_reshape_3(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 8, 12, 24, 14), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 8, 12, 24, 14),
         "ori_format": "NHWC"},
        {"shape": (7, 6, 3, 32, 4, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (7, 6, 3, 32, 4, 8),
         "ori_format": "NHWC"},
        [2, 0, 1, 3, 4], [6, 3, 7, 4, 8, 32], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    assert len(split_maps) == 0

def test_get_op_support_info_nd_transpose_reshape_4(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 8, 12, 24, 14), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 8, 12, 24, 14),
         "ori_format": "NHWC"},
        {"shape": (7, 6, 3, 32, 4, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (7, 6, 3, 32, 4, 8),
         "ori_format": "NHWC"},
        [2, 3, 0, 1, 4], [8, 9, 4, 1, 2, 14, 16], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 4

def test_get_op_support_info_nz_3_4_reshape_transpose_5(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (32, 4, 48, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 768, 64),
         "ori_format": "NHWC"},
        {"shape": (16, 32, 4, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 32, 48, 64),
         "ori_format": "NHWC"},
        [2, 0, 1, 3], [32, 48, 16, 64], False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 1

def test_get_op_support_info_nz_3_4_reshape_transpose_6(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (32, 288, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64, 4608),
         "ori_format": "NHWC"},
        {"shape": (64, 96, 3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96, 32, 48),
         "ori_format": "NHWC"},
        [1, 2, 0, 3], [32, 64, 96, 48], False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 3

def test_get_op_support_info_nz_2_4_reshape_transpose_7(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (288, 96, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1536, 4608),
         "ori_format": "NHWC"},
        {"shape": (48, 16, 18, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (48, 16, 32, 288),
         "ori_format": "NHWC"},
        [1, 2, 0, 3], [32, 48, 16, 288], False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 1

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 3

def test_get_op_support_info_nz_4_2_transpose_reshape_8(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (48, 16, 18, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (48, 16, 32, 288),
         "ori_format": "NHWC"},
        {"shape": (864, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 13824),
        "ori_format": "NHWC"},
        [1, 2, 0, 3], [512, 13824], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 0

def test_get_op_support_info_nz_4_3_transpose_reshape_9(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (16, 32, 4, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 32, 48, 64),
         "ori_format": "NHWC"},
        {"shape": (32, 4, 48, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 768, 64),
        "ori_format": "NHWC"},
        [1, 2, 0, 3], [32, 768, 64], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    assert len(split_maps) == 0

def test_get_op_support_info_nz_4_3_transpose_reshape_10(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (64, 32, 4, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 32, 48, 64),
         "ori_format": "NHWC"},
        {"shape": (32, 4, 192, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 3072, 64),
        "ori_format": "NHWC"},
        [1, 0, 2, 3], [32, 3072, 64], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 2

def test_get_op_support_info_nz_4_3_transpose_reshape_11(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (64, 96, 3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96, 32, 48),
         "ori_format": "NHWC"},
        {"shape": (32, 288, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 64, 4608),
         "ori_format": "NHWC"},
        [2, 0, 1, 3], [32, 64, 4608], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 2

def test_get_op_support_info_nd_reshape_transpose_12(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (4, 8, 12, 24, 14), "dtype": "float16", "format": "NHWC", "ori_shape": (4, 8, 12, 24, 14),
         "ori_format": "NHWC"},
        {"shape": (56, 2, 64, 1, 1, 2, 9), "dtype": "float16", "format": "NHWC", "ori_shape": (56, 2, 64, 1, 1, 2, 9),
         "ori_format": "NHWC"},
        [6, 2, 5, 0, 3, 1, 4], [1, 2, 2, 1, 9, 64, 56], False)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 3

def test_get_op_support_info_nz_4_4_transpose_reshape_13(test_arg):
    """
    test for get_op_support_info
    """
    support_info = get_op_support_info(
        {"shape": (64, 96, 3, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (64, 96, 32, 48),
         "ori_format": "NHWC"},
        {"shape": (96, 384, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (6144, 1536),
        "ori_format": "NHWC"},
        [0, 1, 2, 3], [6144, 1536], True)
    split_maps = json.loads(support_info).get("_op_slice_info").get("splitMaps")

    input_list = split_maps[0].get("inputList")
    assert input_list[0].get("axis")[0] == 0

    output_list = split_maps[0].get("outputList")
    assert output_list[0].get("axis")[0] == 1

ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_reshape_transpose_1)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_reshape_transpose_2)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_transpose_reshape_3)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_transpose_reshape_4)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_3_4_reshape_transpose_5)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_3_4_reshape_transpose_6)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_2_4_reshape_transpose_7)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_4_2_transpose_reshape_8)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_4_3_transpose_reshape_9)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_4_3_transpose_reshape_10)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_4_3_transpose_reshape_11)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nd_reshape_transpose_12)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_nz_4_4_transpose_reshape_13)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
