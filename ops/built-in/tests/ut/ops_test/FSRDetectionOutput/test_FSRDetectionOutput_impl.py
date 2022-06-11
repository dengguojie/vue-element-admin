#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("FsrDetectionOutput", None, None)

def get_param_input(total_num, num_classes, batch_rois, max_rois_num, input_dtype):
    score_dic = {
        "shape": [total_num, (num_classes + 15) // 16, 1, 1, 16],
        "dtype": input_dtype,
        "ori_shape": [total_num, (num_classes + 15) // 16, 1, 1, 16],
        "format": "NC1HWC0",
        "ori_format": "NC1HWC0",
    }
    bbox_delta_dic = {
        "shape": [total_num, (num_classes * 4 + 15) // 16, 1, 1, 16],
        "dtype": input_dtype,
        "ori_shape": [total_num, (num_classes * 4 + 15) // 16, 1, 1, 16],
        "format": "NC1HWC0",
        "ori_format": "NC1HWC0",
    }
    rois_dic = {
        "shape": [batch_rois, 5, max_rois_num],
        "dtype": input_dtype,
        "ori_shape": [batch_rois, 5, max_rois_num],
        "format": "NCHW",
        "ori_format": "NCHW",
    }

    actual_rois_num_dic = {
        "shape": [batch_rois, 8],
        "dtype": "int32",
        "ori_shape": [batch_rois, 8],
        "format": "NCHW",
        "ori_format": "NCHW",
    }

    actual_bbox_num_dic = {
        "shape": [batch_rois, num_classes, 8],
        "dtype": "int32",
        "ori_shape": [batch_rois, num_classes, 8],
        "format": "NCHW",
        "ori_format": "NCHW",
    }

    box_dic = {
        "shape": [batch_rois, num_classes, max_rois_num, 8],
        "dtype": input_dtype,
        "ori_shape": [batch_rois, num_classes, max_rois_num, 8],
        "format": "NCHW",
        "ori_format": "NCHW",
    }

    if max_rois_num > 1024:
        box_dic = {
            "shape": [batch_rois, num_classes, 1024, 8],
            "dtype": input_dtype,
            "ori_shape": [batch_rois, num_classes, 1024, 8],
            "format": "NCHW",
            "ori_format": "NCHW",
        }

    im_info_dic = {
        "shape": [batch_rois, 16],
        "dtype": input_dtype,
        "ori_shape": [batch_rois, 16],
        "format": "NCHW",
        "ori_format": "NCHW",
    }
    return [
        rois_dic, bbox_delta_dic, score_dic, im_info_dic, actual_rois_num_dic, actual_bbox_num_dic, box_dic, 5, 0.1, 0.2
    ]


case1 = {
    "params": get_param_input(10000, 5, 5, 3200, "float16"),
    "case_name": "fsr_detection_output_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case2 = {
    "params": get_param_input(10000, 5, 5, 1200, "float16"),
    "case_name": "fsr_detection_output_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case3 = {
    "params": get_param_input(10000, 5, 5, 4000, "float32"),
    "case_name": "fsr_detection_output_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

def test_get_op_support_info(test_arg):
    from impl.fsr_detection_output import get_op_support_info
    get_op_support_info(*(get_param_input(10000, 5, 5, 3200, "float16")))


ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case2)
ut_case.add_case(["Ascend310P3"], case3)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend310P3", "Ascend910A"])
