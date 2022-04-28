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

NonMaxSuppression dynamic ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("NonMaxSuppressionBucketize", "impl.dynamic.non_max_suppression_bucketize", "non_max_suppression_bucketize")

ut_case.add_case(
    ["Ascend910A"], {
        "params": [{
            'shape': (4, 100, 4),
            'dtype': "float16",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (4, 100, 4)
        }, {
            'shape': (4, 100),
            'dtype': "float16",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (4, 100)
        }, {
            'shape': (4, 100),
            'dtype': "float16",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (4, 100)
        }, {
            'shape': (4, ),
            'dtype': "int32",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (4, )
        }, {
            'shape': (4, -1, 4),
            'dtype': "uint8",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (-1, ),
            "range": [(4, 4), (0, 100), (4, 4)]
        }, {
            'shape': (4, -1),
            'dtype': "uint8",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (4, -1),
            "range": [(4, 4), (0, 100)]
        }, {
            'shape': (4, -1),
            'dtype': "uint8",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (4, -1),
            "range": [(4, 4), (0, 100)]
        }],
        "expect":
        "success",
        "support_expect":
        True,
        "case_name":
        "test_nonmaxsuppressionbucketize_dynamic_001"
    })

ut_case.run("Ascend910A")