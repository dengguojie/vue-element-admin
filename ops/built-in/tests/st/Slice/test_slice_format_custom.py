"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic Pad ut case
"""

# from op_test_frame.ut import OpUT
import json
from impl.dynamic.slice import op_select_format


def check_format(support_dtype, support_format, format_json):
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
def test_op_select_format_001():
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

def test_op_select_format_002():
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

def test_op_select_format_003():
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

def test_op_select_format_004():
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

def test_op_select_format_005():
    result = op_select_format({"shape": [32, 23, 45, 65, 79], "dtype": "float32", "format": "NDHWC",
                               "ori_shape": [32, 23, 45, 65, 79], "ori_format": "NDHWC", "sub_format": 1},
                     {"shape": (5,), "dtype": "int64", "format": "NCHW", "ori_shape": (5,), "ori_format": "NCHW", "const_value": (16, 1, 2, 0, 16)},
                     {"shape": (5,), "dtype": "int64", "format": "NCHW", "ori_shape": (5,), "ori_format": "NCHW", "const_value": (32, 6, 5, 7, 32)},
                     {"shape": [16, 5, 3, 7, 16], "dtype": "float32", "format": "NHHWC", "ori_shape": [16, 5, 3, 7, 16],"ori_format": "NDHWC"})
    expect_dtype = {'uint64', 'int8', 'uint16', 'bool', 'int16', 'int64', 'uint32', 'int32', 'float', 'float16', 'uint8'}
    expect_format = {"ND", "NDC1HWC0", "FRACTAL_Z_3D"}
    check_format(expect_dtype, expect_format, result)

def test_op_select_format_006():
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

def test_op_select_format_007():
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

def test_op_select_format_008():
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

def test_op_select_format_009():
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

if __name__ == '__main__':
    test_op_select_format_001()
    test_op_select_format_002()
    test_op_select_format_003()
    test_op_select_format_004()
    test_op_select_format_006()
    test_op_select_format_007()
    test_op_select_format_008()
    test_op_select_format_009()
