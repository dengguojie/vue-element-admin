"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

DecodeCornerpointsTargetWrtCenterV1 ut case
"""
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info
ut_case = OpUT("DecodeCornerpointsTargetWrtCenterV1", None, None)

case1 = {"params": [{"shape": (8,8), "dtype": "float16", "format": "ND", "ori_shape": (8,8),"ori_format": "ND"}, #x
                    {"shape": (8,4), "dtype": "float16", "format": "ND", "ori_shape": (8,4),"ori_format": "ND"},
                    {"shape": (8,8), "dtype": "float16", "format": "ND", "ori_shape": (8,8),"ori_format": "ND"},
                    ],
         "case_name": "DecodeCornerpointsTargetWrtCenterV1_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (65400,8), "dtype": "float16", "format": "ND", "ori_shape": (65400,8),"ori_format": "ND"}, #x
                    {"shape": (65400,4), "dtype": "float16", "format": "ND", "ori_shape": (65400,4),"ori_format": "ND"},
                    {"shape": (65400,8), "dtype": "float16", "format": "ND", "ori_shape": (65400,8),"ori_format": "ND"},
                    ],
         "case_name": "DecodeCornerpointsTargetWrtCenterV1_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (8,4), "dtype": "float16", "format": "ND", "ori_shape": (8,4),"ori_format": "ND"}, #x
                    {"shape": (8,4), "dtype": "float16", "format": "ND", "ori_shape": (8,4),"ori_format": "ND"},
                    {"shape": (8,4), "dtype": "float16", "format": "ND", "ori_shape": (8,4),"ori_format": "ND"},
                    ],
         "case_name": "DecodeCornerpointsTargetWrtCenterV1_3",
         "expect": RuntimeError,
         "support_expect": True}


# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case1)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case2)
ut_case.add_case(["Ascend910","Ascend310","Ascend710"], case3)

def calc_expect_func(inputA, inputB, output):
    input_shape_x = inputA['shape']
    data_a = inputA['value']
    data_b = inputB['value']
    keypoints_predictions = np.reshape(data_a, (-1, 8))
    anchor_x1, anchor_y1, anchor_x2, anchor_y2 = np.split(data_b, 4 , axis=1)
    anchor_center_x = 0.5 * (anchor_x1 + anchor_x2)
    anchor_center_y = 0.5 * (anchor_y1 + anchor_y2)
    anchors_p1p2p3p4 = np.concatenate((anchor_center_x, anchor_y1, anchor_x2, anchor_center_y, anchor_center_x, anchor_y2, anchor_x1, anchor_center_y), axis=1)
    anchor_w = anchor_x2 - anchor_x1
    anchor_h = anchor_y2 - anchor_y1
    anchor_whwhwhwh = np.concatenate((anchor_w, anchor_h, anchor_w, anchor_h,anchor_w, anchor_h,anchor_w, anchor_h), axis=1)

    decode_keypoint = keypoints_predictions * anchor_whwhwhwh  +  anchors_p1p2p3p4
    c = np.reshape(decode_keypoint, (-1, 1, 8))

    data_c = np.reshape(c, input_shape_x)
    return data_c

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 8), "dtype": "float16", "format": "ND", "ori_shape": (1, 8),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (8, 8), "dtype": "float16", "format": "ND", "ori_shape": (8, 8),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (8, 4), "dtype": "float16", "format": "ND", "ori_shape": (8, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (8, 8), "dtype": "float16", "format": "ND", "ori_shape": (8, 8),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16, 8), "dtype": "float16", "format": "ND", "ori_shape": (16, 8),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 4), "dtype": "float16", "format": "ND", "ori_shape": (16, 4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16, 8), "dtype": "float16", "format": "ND", "ori_shape": (16, 8),"ori_format": "ND", "param_type": "output"},
                                                    ],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
