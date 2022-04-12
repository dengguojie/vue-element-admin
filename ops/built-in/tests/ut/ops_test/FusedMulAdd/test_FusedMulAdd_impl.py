#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
import json
from op_test_frame.common import precision_info
ut_case = OpUT("FusedMulAdd", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (1,1,16,16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (16,16),"ori_format": "ND"}],
         "case_name": "FusedMulAdd_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)

def calc_expect_func(x1, x2, x3, y):
    mul = x1['value'] * x2['value']
    res = mul + x3['value']
    res = res.astype(y['dtype'])
    return res

precision_case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "param_type":"input"},
                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case2 = {"params": [{"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ), "param_type":"input"},
                              {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case3 = {"params": [{"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32), "param_type":"input"},
                              {"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case4 = {"params": [{"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16), "param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case1)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case2)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case3)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case4)

def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.fused_mul_add import op_select_format
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_1")

def test_op_support_info(test_arg):
    """
    test_op_support_info
    """
    from impl.fused_mul_add import get_op_support_info
    res = get_op_support_info({"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 16, 384, 384), "ori_format": "ND"},
                     {"shape": [], "dtype": "float16", "format": "ND", "ori_shape": [], "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 384, 384), "ori_format": "ND"},
                     "test_fused_mul_add_op_support_info_1")
    split_maps = json.loads(res).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 1
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 2
        idx = input_list[0].get("idx")
        assert idx == 0
        idx = input_list[1].get("idx")
        assert idx == 2
    
    res_2 = get_op_support_info({"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     "test_fused_mul_add_op_support_info_2")

    from impl.util.util_common import update_shape_for_other_format
    #update_shape_for_other_format(src_shape, src_format, dst_format)
    update_shape_for_other_format((16, 16, 16, 16, 16), "NC1HWC0", (16, 256, 16, 16), "NCHW")
    update_shape_for_other_format((16, 16, 16, 16, 16), "ND", (16, 256, 16, 16), "FRACTAL_NZ")
    update_shape_for_other_format((16, 16, 16, 16, 16), "NDCHW", (16, 16, 16, 16, 16), "NDC1HWC0")
    update_shape_for_other_format((16, 16, 16, 16), "NCHW", (16, 16, 16, 16), "FRACTAL_Z")
    update_shape_for_other_format((16, 16, 16, 16), "NCHW", (16, 16, 16, 16), "NC1HWC0")
    update_shape_for_other_format((16, 16, 16, 16), "NCHW", (16, 16, 16, 16), "NCHW", "int8")
    update_shape_for_other_format((16,), "ND", (16,), "FRACTAL_NZ")
    update_shape_for_other_format((16, 16, 16, 16, 16), "NDCHW", (16, 16, 16, 16, 16), "FRACTAL_Z_3D")


ut_case.add_cust_test_func(test_func=test_op_support_info)

