#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.dynamic.slice import op_select_format

ut_case = OpUT("Slice", "impl.dynamic.slice", "slice")

# format case
case_fz_001 = {"params": [
    {"shape": (1 * 16 * 16, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 16, 16),
     "ori_format": "NCHW"},
    {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW",
     "const_value": (0, 0, 0, 0)},
    {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,), "ori_format": "NCHW",
     "const_value": (16, 16, 16, 16)},
    {"shape": (1 * 16 * 16, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (16, 16, 16, 16),
     "ori_format": "NCHW"}
],
    "case_name": "Slice_FRACTAL_Z_001",
    "expect": "success",
    "support_expect": True}

# when FRACTAL_Z, mean: (C1HW)NiNoC0
case_fz_002 = {"params": [
    {"shape": [4 * 23 * 45, 2, 16, 16], "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": [32, 23, 45, 65],
     "ori_format": "NHWC", "sub_format": 1},
    {"shape": (4,), "dtype": "int64", "format": "NHWC", "ori_shape": (4,), "ori_format": "NHWC",
     "const_value": (0, 1, 2, 0)},
    {"shape": (4,), "dtype": "int64", "format": "ND", "ori_shape": (4,), "ori_format": "ND",
     "const_value": (-1, 6, 5, 32)},
    {"shape": [2 * 5 * 3, 2, 16, 16], "dtype": "float32", "format": "FRACTAL_Z",
     "ori_shape": [32, 5, 3, 32],"ori_format": "NHWC"}
],
    "case_name": "Slice_FRACTAL_Z_002",
    "expect": "success",
    "support_expect": True}

case_5hd = {"params": [
    {"shape": (34, 5, 23, 45, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (34, 23, 45, 65),
     "ori_format": "NHWC"},
    {"shape": (4,), "dtype": "int64", "format": "NHWC", "ori_shape": (4,),
     "ori_format": "NHWC", "const_value": (3, 1, 2, 16)},
    {"shape": (4,), "dtype": "int64", "format": "NHWC", "ori_shape": (4,),
     "ori_format": "NHWC", "const_value": (17, 6, 5, -1)},
    {"shape": (14, 4, 5, 3, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (14, 15, 3, 49),
     "ori_format": "NHWC"}
],
    "case_name": "Slice_NC1HWC0",
    "expect": "success",
    "support_expect": True}

case_6hd = {"params": [
    {"shape": (34, 23, 5, 45, 65, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (34, 23, 45, 65, 79),
     "ori_format": "NDHWC"},
    {"shape": (5,), "dtype": "int64", "format": "NHWC", "ori_shape": (4,), "ori_format": "NHWC",
     "const_value": (3, 1, 2, 0, 16)},
    {"shape": (5,), "dtype": "int64", "format": "NHWC", "ori_shape": (4,), "ori_format": "NHWC",
     "const_value": (17, 6, 5, 7, -1)},
    {"shape": (14, 5, 4, 3, 7, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (14, 5, 3, 7, 63),
     "ori_format": "NDHWC"}
],
    "case_name": "Slice_NDC1HWC0",
    "expect": "success",
    "support_expect": True}

# when FRACTAL_NZ, mean: [A, B, C, D] -> [A, B, ceil(D//16), ceil(C//16), 16, 16]
case_nz = {"params": [
    {"shape": [23, 4, 3, 16, 16], "dtype": "float32", "format": "FRACTAL_NZ",
     "ori_shape": [23, 45, 65], "ori_format": "NHWC"},
    {"shape": (3,), "dtype": "int64", "format": "NHWC", "ori_shape": (3,),
     "ori_format": "NHWC", "const_value": (3, 0, 16)},
    {"shape": (3,), "dtype": "int64", "format": "NHWC", "ori_shape": (3,),
     "ori_format": "NHWC", "const_value": (14, -1, 32)},
    {"shape": [14, 1, 3, 16, 16], "dtype": "float32", "format": "FRACTAL_NZ",
     "ori_shape": [11, 45, 16],"ori_format": "NHWC"}
],
    "case_name": "Slice_FRACTAL_NZ",
    "expect": "success",
    "support_expect": True}

ut_case.add_case(["Ascend910A", "Ascend710"], case_fz_001)
ut_case.add_case(["Ascend910A", "Ascend710"], case_fz_002)
ut_case.add_case(["Ascend910A", "Ascend710"], case_5hd)
ut_case.add_case(["Ascend910A", "Ascend710"], case_6hd)
ut_case.add_case(["Ascend910A", "Ascend710"], case_nz)

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

# op_select_format case
def test_op_select_format_001(test_arg):
    result = op_select_format({"shape": [34, 23, 45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [34, 23, 45, 65], "ori_format": "NHWC"},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (3, 1, 2, 0)},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (17, 6, 5, 16)},
                              {"shape": [17, 6, 5, 16], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [17, 6, 5, 16],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "NC1HWC0"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_002(test_arg):
    result = op_select_format({"shape": [34, 23, 45, 65, 79], "dtype": "float32", "format": "NDHWC",
                               "ori_shape": [34, 23, 45, 65, 79], "ori_format": "NDHWC"},
                              {"shape": (5,), "dtype": "int64", "format": "NCHW", "ori_shape": (5,),
                               "ori_format": "NCHW", "const_value": (3, 1, 2, 0, 16)},
                              {"shape": (5,), "dtype": "int64", "format": "NCHW", "ori_shape": (5,),
                               "ori_format": "NCHW", "const_value": (17, 6, 5, 7, -1)},
                              {"shape": [14, 5, 3, 7, 63], "dtype": "float32", "format": "NDHWC",
                               "ori_shape": [14, 5, 3, 7, 63],"ori_format": "NDHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "NDC1HWC0"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_003(test_arg):
    result = op_select_format({"shape": [34, 23, 45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [34, 23, 45, 65], "ori_format": "NHWC"},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW",
                               "ori_shape": (4,), "ori_format": "NCHW", "const_value": (3, 1, 2, 3)},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (17, 6, 5, 16)},
                              {"shape": [17, 6, 5, 13], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [17, 6, 5, 13],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_004(test_arg):
    result = op_select_format({"shape": [32, 23, 45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [32, 23, 45, 65], "ori_format": "NHWC", "sub_format": 1},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (0, 1, 2, 0)},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (-1, 6, 5, 65)},
                              {"shape": [32, 5, 3, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [32, 5, 3, 65],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "NC1HWC0", "FRACTAL_Z"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_005(test_arg):
    result = op_select_format({"shape": [32, 23, 45, 65, 79], "dtype": "float32", "format": "NDHWC",
                               "ori_shape": [32, 23, 45, 65, 79], "ori_format": "NDHWC", "sub_format": 1},
                     {"shape": (5,), "dtype": "int64", "format": "NCHW", "ori_shape": (5,), "ori_format": "NCHW", "const_value": (16, 1, 2, 0, 16)},
                     {"shape": (5,), "dtype": "int64", "format": "NCHW", "ori_shape": (5,), "ori_format": "NCHW", "const_value": (32, 6, 5, 7, 32)},
                     {"shape": [16, 5, 3, 7, 16], "dtype": "float32", "format": "NHHWC", "ori_shape": [16, 5, 3, 7, 16],"ori_format": "NDHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "NDC1HWC0", "FRACTAL_Z_3D"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_006(test_arg):
    result = op_select_format({"shape": [32, 23, 45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [32, 23, 45, 65], "ori_format": "NHWC"},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (0, 1, 2, 0)},
                              {"shape": (4,), "dtype": "int64", "format": "NCHW", "ori_shape": (4,),
                               "ori_format": "NCHW", "const_value": (-1, 6, 5, 65)},
                              {"shape": [32, 5, 3, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [32, 5, 3, 65],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "NC1HWC0"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_007(test_arg):
    result = op_select_format({"shape": [23, 45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [23, 45, 65], "ori_format": "NHWC"},
                              {"shape": (3,), "dtype": "int64", "format": "NCHW", "ori_shape": (3,),
                               "ori_format": "NCHW", "const_value": (3, 0, 16)},
                              {"shape": (3,), "dtype": "int64", "format": "NCHW", "ori_shape": (3,),
                               "ori_format": "NCHW", "const_value": (14, 45, 32)},
                              {"shape": [11, 45, 16], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [11, 45, 16],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "FRACTAL_NZ"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_008(test_arg):
    result = op_select_format({"shape": [45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [45, 65], "ori_format": "NHWC"},
                              {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,),
                               "ori_format": "NCHW", "const_value": (0, 16)},
                              {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,),
                               "ori_format": "NCHW", "const_value": (-1, 32)},
                              {"shape": [45, 16], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [45, 16],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "FRACTAL_NZ"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_009(test_arg):
    result = op_select_format({"shape": [45, 65], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [45, 65], "ori_format": "NHWC"},
                              {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,),
                               "ori_format": "NCHW", "const_value": (3, 16)},
                              {"shape": (2,), "dtype": "int64", "format": "NCHW", "ori_shape": (2,),
                               "ori_format": "NCHW", "const_value": (-1, 32)},
                              {"shape": [42, 16], "dtype": "float32", "format": "NHWC",
                               "ori_shape": [42, 16],"ori_format": "NHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND"}
    check_format(expect_dtype, expect_format, result)

ut_case.add_cust_test_func(test_func=test_op_select_format_001)
ut_case.add_cust_test_func(test_func=test_op_select_format_002)
ut_case.add_cust_test_func(test_func=test_op_select_format_003)
ut_case.add_cust_test_func(test_func=test_op_select_format_004)
ut_case.add_cust_test_func(test_func=test_op_select_format_006)
ut_case.add_cust_test_func(test_func=test_op_select_format_007)
ut_case.add_cust_test_func(test_func=test_op_select_format_008)
ut_case.add_cust_test_func(test_func=test_op_select_format_009)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    ut_case.run("Ascend310")
    ut_case.run("Ascend710")
