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

YoloxBoundingBoxDecode ut case
"""
import tbe
from unittest.mock import patch
from unittest.mock import MagicMock
from op_test_frame.ut import OpUT

ut_case = OpUT("YoloxBoundingBoxDecode", "impl.dynamic.yolox_bounding_box_decode", "yolox_bounding_box_decode")

case1 = {"params": [{"shape": (8400, 4), "dtype": "float16", "format": "ND", "ori_shape": (8400, 4),"ori_format": "ND",
                     "range":[(8400, 8400),(4,4)]},             # priors
                    {"shape": (-1, 8400, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 8400, 4),"ori_format": "ND",
                     "range":[(1, None),(8400, 8400),(4, 4)]},   # bboxes
                    {"shape": (-1, 8400, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 8400, 4),"ori_format": "ND",
                     "range":[(1, None),(8400, 8400),(4, 4)]}    # decoded_bboxes
                    ],
         "case_name": "YoloxBoundingBoxDecode_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (8400, 4),"ori_format": "ND",
                     "range":[(1, None),(4,4)]},                # priors
                    {"shape": (-1, -1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 8400, 4),"ori_format": "ND",
                     "range":[(1, None),(1, None),(4, 4)]},     # bboxes
                    {"shape": (-1, -1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 8400, 4),"ori_format": "ND",
                     "range":[(1, None),(1, None),(4, 4)]}      # decoded_bboxes
                    ],
         "case_name": "YoloxBoundingBoxDecode_dynamic_2",
         "expect": "success",
         "support_expect": True}


case3 = {"params": [{"shape": (-1, 3), "dtype": "float16", "format": "ND", "ori_shape": (8400, 3),"ori_format": "ND",
                     "range":[(1, None),(1, None)]},            # priors
                    {"shape": (-1, -1, 2), "dtype": "float16", "format": "ND", "ori_shape": (-1, 8400, 2),"ori_format": "ND",
                     "range":[(1, None),(1, None),(1, None)]},  # bboxes
                    {"shape": (-1, -1, 2), "dtype": "float16", "format": "ND", "ori_shape": (-1, 8400, 2),"ori_format": "ND",
                     "range":[(1, None),(1, None),(1, None)]}   # decoded_bboxes
                    ],
         "case_name": "YoloxBoundingBoxDecode_dynamic_3",
         "expect": RuntimeError,
         "support_expect": True}


case4 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (8400, 4),"ori_format": "ND",
                     "range":[(1, None),(4, 4)]},               # priors
                    {"shape": (-1, -1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4800, 4),"ori_format": "ND",
                     "range":[(1, None),(1, None),(4, 4)]},     # bboxes
                    {"shape": (-1, -1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4800, 4),"ori_format": "ND",
                     "range":[(1, None),(1, None),(4, 4)]}      # decoded_bboxes
                    ],
         "case_name": "YoloxBoundingBoxDecode_dynamic_4",
         "expect": RuntimeError,
         "support_expect": True}


case5 = {"params": [{"shape": (8400, 4), "dtype": "float32", "format": "ND", "ori_shape": (8400, 4),"ori_format": "ND",
                     "range":[(8400, 8400),(4, 4)]},             # priors
                    {"shape": (-1, 8400, 4), "dtype": "float32", "format": "ND", "ori_shape": (-1, 8400, 4),"ori_format": "ND",
                     "range":[(1, None),(8400, 8400),(4, 4)]},   # bboxes
                    {"shape": (-1, 8400, 4), "dtype": "float32", "format": "ND", "ori_shape": (-1, 8400, 4),"ori_format": "ND",
                     "range":[(1, None),(8400, 8400),(4, 4)]}    # decoded_bboxes
                    ],
         "case_name": "YoloxBoundingBoxDecode_dynamic_5",
         "expect": "success",
         "support_expect": True}



ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case4)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case5)


if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310","Ascend710"])
    exit(0)
