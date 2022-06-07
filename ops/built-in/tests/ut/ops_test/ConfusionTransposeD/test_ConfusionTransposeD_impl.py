#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

    result = op_select_format({'shape': (64, 1024, 384), 'ori_shape': (64, 1024, 384), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              {'shape': (64, 1024, 384), 'ori_shape': (64, 1024, 384), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              (0, 1, 2), (65536, 384), True, "test_confusion_transpose_d_op_format_1")
    check_format(support_all_dtype,{"FRACTAL_NZ","ND"},result)
    result = op_select_format({'shape': (65536, 1536), 'ori_shape': (65536, 1536), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              {'shape': (65536, 1536), 'ori_shape': (65536, 1536), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              (0, 1, 2), (64, 1024, 1536), False, "test_dropout_do_mask_v3_d_op_select_format_2")
    check_format(support_all_dtype,{"FRACTAL_NZ","ND"},result)
    result =  op_select_format({'shape': (64, 1024, 384), 'ori_shape': (64, 1024, 384), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                               {'shape': (64, 1024, 384), 'ori_shape': (64, 1024, 384), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                               (0, 2, 1, 3), (64, 1024, 6, 64), False, "test_dropout_do_mask_v3_d_op_select_format_3")
    check_format(support_all_dtype,{"FRACTAL_NZ","ND"},result)
    result = op_select_format({'shape': (64, 6, 1024, 64), 'ori_shape': (64, 6, 1024, 64), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              {'shape': (64, 6, 1024, 64), 'ori_shape': (64, 6, 1024, 64), 'format': 'ND', 'ori_format': 'ND', 'dtype': 'float32'},
                              (0, 2, 1, 3), (64, 1024, 384), True, "test_dropout_do_mask_v3_d_op_select_format_4")
    check_format(support_all_dtype,{"FRACTAL_NZ","ND"},result)

ut_case.add_cust_test_func(test_func=test_op_select_format)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
