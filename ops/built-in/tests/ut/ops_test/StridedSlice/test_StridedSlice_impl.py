"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

SpaceToBatch ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("StridedSlice", None, None)

# mode 7
case1 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 8, 16), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "const_value": [1]},
                    {"shape": (8, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 16), "ori_format": "ND"},
                     0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSlice_1",
         "expect": "success",
         "support_expect": True}

# mode 1
case2 = {"params": [{"shape": (8, 8, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 8, 16), "ori_format": "ND"},
                    {"shape": (3,), "dtype": "int32", "format": "ND",
                     "ori_shape": (3,), "ori_format": "ND"},
                    {"shape": (3,), "dtype": "int32", "format": "ND",
                     "ori_shape": (3,), "ori_format": "ND"},
                    {"shape": (3,), "dtype": "int32", "format": "ND",
                     "ori_shape": (3,), "ori_format": "ND", "const_value": [1, 1, 1]},
                    {"shape": (8, 16), "dtype": "float16", "format": "ND",
                     "ori_shape": (8, 16), "ori_format": "ND"},
                     1, 1, 0, 0, 2
                    ],
         "case_name": "StridedSlice_2",
         "expect": "success",
         "support_expect": True}

# mode 7
case3 = {"params": [{"shape": (8, 8, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (8, 8, 16), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "int32", "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND", "const_value": [1]},
                    {"shape": (8, 16), "dtype": "float32", "format": "ND",
                     "ori_shape": (8, 16), "ori_format": "ND"},
                     0, 0, 0, 0, 1
                    ],
         "case_name": "StridedSlice_3",
         "expect": "success",
         "support_expect": True}

# mode 1
case4 = {"params": [{"shape": (4, 4, 4, 4), "dtype": "float16", "format": "ND",
                     "ori_shape": (4, 4, 4, 4), "ori_format": "ND"},
                    {"shape": (4,), "dtype": "int32", "format": "ND",
                     "ori_shape": (4,), "ori_format": "ND"},
                    {"shape": (4,), "dtype": "int32", "format": "ND",
                     "ori_shape": (4,), "ori_format": "ND"},
                    {"shape": (4,), "dtype": "int32", "format": "ND",
                     "ori_shape": (4,), "ori_format": "ND", "const_value": [1, 1, 1, 1]},
                    {"shape": (2, 2, 2, 2), "dtype": "float16", "format": "ND",
                     "ori_shape": (2, 2, 2, 2), "ori_format": "ND"},
                     0, 0, 0, 0, 0
                    ],
         "case_name": "StridedSlice_4",
         "expect": "success",
         "support_expect": True}

# mode 3
case5 = {"params": [{"shape": (1, 5, 5, 5, 424, 35), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 5, 5, 5, 424, 35), "ori_format": "ND"},
                    {"shape": (6,), "dtype": "int32", "format": "ND",
                     "ori_shape": (6,), "ori_format": "ND"},
                    {"shape": (6,), "dtype": "int32", "format": "ND",
                     "ori_shape": (6,), "ori_format": "ND"},
                    {"shape": (6,), "dtype": "int32", "format": "ND",
                     "ori_shape": (6,), "ori_format": "ND", "const_value": [1, 1, 1, 1, 1, 1]},
                    {"shape": (1, 2 ,4, 4, 190, 5), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, 2 ,4, 4, 190, 5), "ori_format": "ND"},
                     0, 0, 0, 0, 0
                    ],
         "case_name": "StridedSlice_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (64, 256), "dtype": "float16", "format": "ND",
                     "ori_shape": (64, 256), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND",
                     "ori_shape": (2,), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND",
                     "ori_shape": (2,), "ori_format": "ND"},
                    {"shape": (2,), "dtype": "int32", "format": "ND",
                     "ori_shape": (2,), "ori_format": "ND", "const_value": [1, 1]},
                    {"shape": (64, 128), "dtype": "float16", "format": "ND",
                     "ori_shape": (64, 128), "ori_format": "ND"},
                     3, 1, 0, 0, 0
                    ],
         "case_name": "StridedSlice_6",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["all"], case1)
ut_case.add_case(["all"], case2)
ut_case.add_case(["all"], case3)
ut_case.add_case(["all"], case4)
ut_case.add_case(["all"], case5)
ut_case.add_case(["all"], case6)

def test_op_check_supported_1(test_arg):
    from impl.strided_slice import check_supported
    shape = {'ori_shape': (8, 8, 16), 'shape': (8, 8, 16), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    begin = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    end = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    strides = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64', "const_value": (1, 1, 3)}
    output = {'ori_shape': (8, 16), 'shape': (8, 16), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, output, 0, 0, 0, 0, 1) == True:
        raise Exception("Failed to call check_supported in stridedslice.")
        
def test_op_check_supported_2(test_arg):
    from impl.strided_slice import check_supported
    shape = {'ori_shape': (8, 8, 16), 'shape': (8, 8, 16), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    begin = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    end = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    strides = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    output = {'ori_shape': (8, 16), 'shape': (8, 16), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, output, 0, 0, 0, 0, 1) == True:
        raise Exception("Failed to call check_supported in stridedslice.")

def test_op_check_supported_3(test_arg):
    from impl.strided_slice import check_supported
    shape = {'ori_shape': (8, 8, 16), 'shape': (8, 8, 16), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    begin = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    end = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64'}
    strides = {'ori_shape': (3,), 'shape': (3,), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'int64', "const_value": (1, 1, 1)}
    output = {'ori_shape': (8, 16), 'shape': (8, 16), 'ori_format': 'NHWC', 'format': 'NHWC', 'dtype': 'float16'}
    if check_supported(shape, begin, end, strides, output, 0, 0, 0, 0, 1) == False:
        raise Exception("Failed to call check_supported in stridedslice.")

ut_case.add_cust_test_func(test_func=test_op_check_supported_1)
ut_case.add_cust_test_func(test_func=test_op_check_supported_2)
ut_case.add_cust_test_func(test_func=test_op_check_supported_3)


if __name__ == "__main__":
    ut_case.run("Ascend910A")
