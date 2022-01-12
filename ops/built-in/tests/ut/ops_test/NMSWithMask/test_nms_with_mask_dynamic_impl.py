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

NMSWithMask dynamic ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("NMSWithMask", "impl.dynamic.nms_with_mask", "nms_with_mask")

ut_case.add_case(
    ["Ascend910A"], {
        "params": [{
            'shape': (-1, 8),
            'dtype': "float16",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (-1, 8),
            "range": [(16, 16), (8, 8)]
        }, {
            'shape': (-1, 5),
            'dtype': "float16",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (-1, 5),
            "range": [(16, 16), (5, 5)]
        }, {
            'shape': (-1, ),
            'dtype': "int32",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (-1, ),
            "range": [(16, 16)]
        }, {
            'shape': (-1, ),
            'dtype': "uint8",
            "format": "ND",
            "ori_format": "ND",
            "ori_shape": (-1, ),
            "range": [(16, 16)]
        }, 0.7],
        "expect":
        "success",
        "support_expect":
        True,
        "case_name":
        "test_nms_with_mask_dynamic_001"
    })


def test_op_mask_generalization_1(test_arg):
    from impl.dynamic.nms_with_mask import nms_with_mask_generalization

    box_scores = {
        'shape': (-1, -1),
        'dtype': "float16",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, 8),
        "range": [(16, 16), (8, 8)]
    }
    selected_boxes = {
        'shape': (-1, 5),
        'dtype': "float16",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, 5),
        "range": [(16, 16), (5, 5)]
    }
    selected_idx = {
        'shape': (-1, ),
        'dtype': "int32",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, ),
        "range": [(16, 16)]
    }
    selected_mask = {
        'shape': (-1, ),
        'dtype': "uint8",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, ),
        "range": [(16, 16)]
    }
    iou_thr = 0.7
    generalize_config = {"mode" : "keep_rank"}

    if not nms_with_mask_generalization(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr,
                                        generalize_config):
        raise Exception("Failed to call nms_with_mask_generalization in nms_with_mask.")


ut_case.add_cust_test_func(test_func=test_op_mask_generalization_1)
ut_case.run("Ascend910A")
