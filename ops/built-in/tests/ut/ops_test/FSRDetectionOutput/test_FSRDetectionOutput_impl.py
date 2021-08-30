#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("FsrDetectionOutput", None, None)

total_num = 10000
num_classes = 5
batch_rois = 5
max_rois_num = 3200
input_dtype = "float16"
score_dic = {
    "shape": [total_num,(num_classes+15)//16,1,1,16],
    "dtype": input_dtype,
    "ori_shape": [total_num,(num_classes+15)//16,1,1,16],
    "format":"NC1HWC0",
    "ori_format":"NC1HWC0",
}
bbox_delta_dic = {
    "shape": [total_num,(num_classes*4+15)//16,1,1,16],
    "dtype": input_dtype,
    "ori_shape": [total_num,(num_classes*4+15)//16,1,1,16],
    "format":"NC1HWC0",
    "ori_format":"NC1HWC0",
}
rois_dic = {
    "shape": [batch_rois, 5, max_rois_num],
    "dtype": input_dtype,
    "ori_shape": [batch_rois, 5, max_rois_num],
    "format":"NCHW",
    "ori_format":"NCHW",
}

actual_rois_num_dic = {
    "shape": [batch_rois,8],
    "dtype": "int32",
    "ori_shape": [batch_rois,8],
    "format":"NCHW",
    "ori_format":"NCHW",
}

actual_bbox_num_dic = {
    "shape": [batch_rois, num_classes, 8],
    "dtype": "int32",
    "ori_shape": [batch_rois, num_classes, 8],
    "format":"NCHW",
    "ori_format":"NCHW",
}

box_dic = {
    "shape": [batch_rois,num_classes,max_rois_num, 8],
    "dtype": input_dtype,
    "ori_shape": [batch_rois,num_classes,max_rois_num, 8],
    "format":"NCHW",
    "ori_format":"NCHW",
}

if max_rois_num > 1024:
    box_dic = {
        "shape": [batch_rois,num_classes,1024, 8],
        "dtype": input_dtype,
        "ori_shape":  [batch_rois,num_classes,1024, 8],
        "format":"NCHW",
        "ori_format":"NCHW",
    }

im_info_dic = {
    "shape": [batch_rois, 16],
    "dtype": input_dtype,
    "ori_shape": [batch_rois, 16],
    "format":"NCHW",
    "ori_format":"NCHW",
}

case1 = {"params": [rois_dic, bbox_delta_dic, score_dic, im_info_dic, 
                    actual_rois_num_dic, actual_bbox_num_dic, box_dic, 
                    5, 0.1, 0.2],
         "case_name": "fsr_detection_output_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


def test_get_op_support_info(test_arg):
    from impl.fsr_detection_output import get_op_support_info
    get_op_support_info(rois_dic, bbox_delta_dic, score_dic, im_info_dic,
                        actual_rois_num_dic, actual_bbox_num_dic, box_dic,
                        5, 0.1, 0.2)


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
