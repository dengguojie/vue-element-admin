"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ImageProjectiveTransform ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("ImageProjectiveTransform", "impl.dynamic.image_projective_transform", "image_projective_transform")

case1 = {
    "params": [
        {"shape": (2, 5, 3, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 5, 3, 3),"ori_format": "NHWC"},
        {"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (2, 5, 3, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 5, 3, 3),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [
        {"shape": (34, 5, 3, 3), "dtype": "float16", "format": "NHWC", "ori_shape": (34, 5, 3, 3),"ori_format": "NHWC"},
        {"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (34, 5, 3, 3), "dtype": "float16", "format": "NHWC", "ori_shape": (34, 5, 3, 3),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        {"shape": (2, 5, 3, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 5, 3, 3),"ori_format": "NHWC"},
        {"shape": (2, 8), "dtype": "float32", "format": "ND", "ori_shape": (2, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (2, 5, 3, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 5, 3, 3),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_3",
    "expect": "success",
    "support_expect": True
}
case4 = {
    "params": [
        {"shape": (34, 5, 3, 3), "dtype": "float16", "format": "NHWC", "ori_shape": (34, 5, 3, 3),"ori_format": "NHWC"},
        {"shape": (34, 8), "dtype": "float32", "format": "ND", "ori_shape": (34, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (34, 5, 3, 3), "dtype": "float16", "format": "NHWC", "ori_shape": (34, 5, 3, 3),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_4",
    "expect": "success",
    "support_expect": True
}
case5 = {
    "params": [
        {"shape": (4, 100, 100, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 100, 100, 3),"ori_format": "NHWC"},
        {"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (4, 100, 100, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 100, 100, 3),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_5",
    "expect": "success",
    "support_expect": True
}
case6 = {
    "params": [
        {"shape": (34, 100, 100, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (34, 100, 100, 3),"ori_format": "NHWC"},
        {"shape": (34, 8), "dtype": "float32", "format": "ND", "ori_shape": (34, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (34, 100, 100, 3), "dtype": "float32", "format": "NHWC", "ori_shape": (34, 100, 100, 3),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_6",
    "expect": "success",
    "support_expect": True
}
case7 = {
    "params": [
        {"shape": (4, 2, 2, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        {"shape": (1, 8), "dtype": "float32", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (4, 2, 2, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_7",
    "expect": "success",
    "support_expect": True
}
case8 = {
    "params": [
        {"shape": (4, 2, 2, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        {"shape": (4, 8), "dtype": "float32", "format": "ND", "ori_shape": (4, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (4, 2, 2, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_8",
    "expect": "success",
    "support_expect": True
}
case9 = {
    "params": [
        {"shape": (4, 2, 2, 1), "dtype": "int32", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        {"shape": (4, 8), "dtype": "float32", "format": "ND", "ori_shape": (4, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (4, 2, 2, 1), "dtype": "int32", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_9",
    "expect": "success",
    "support_expect": True
}
case10 = {
    "params": [
        {"shape": (4, 2, 2, 1), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        {"shape": (4, 8), "dtype": "float32", "format": "ND", "ori_shape": (4, 8),"ori_format": "ND"},
        {"shape": (2, ), "dtype": "int32", "format": "ND", "ori_shape": (2, ),"ori_format": "ND"},
        {"shape": (4, 2, 2, 1), "dtype": "uint8", "format": "NHWC", "ori_shape": (4, 2, 2, 1),"ori_format": "NHWC"},
        "NEAREST"
    ],
    "case_name": "ImageProjectiveTransform_9",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
