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

YoloPreDetection ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("YoloPreDetection", "impl.yolo_pre_detection", "yolo_pre_detection")

def ceil_x_yolo_pre_detection(total_len, align_value):
    """
    ceil align
    Parameters
    ----------
    total_len: len before align
    align_value: align byte
    Returns
    -------
    len after ceil align
    """
    align_len = (total_len + align_value - 1) // align_value * align_value
    return align_len

def gen_yolo_pre_detection_data(batch_val, grid_h_val, grid_w_val, dtype_val,
                                boxes_val, coords_val, classes_val, yolo_version, softmax,
                                background, softmaxtree, case_name):
    """
    feature_dic, boxes, coords, classes, yolo_version="V5",
    softmax=False, background=False, kernel_name="yolo_pre_detection"
    feature_dic{'shape':shape, 'format':"NCHW", 'dtype':dtype}
    """
    dtype_size = 2 if dtype_val == "float16" else 4
    shape_val = (batch_val, 1, grid_h_val, grid_w_val)
    input_dic = {'shape': shape_val, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":shape_val}
    coord_shape = (batch_val, boxes_val*coords_val,
                   ceil_x_yolo_pre_detection(grid_h_val*grid_w_val*dtype_size, 32)//dtype_size)
    obj_shape = (batch_val, ceil_x_yolo_pre_detection(boxes_val * grid_h_val * grid_w_val *
                                   dtype_size + 32, 32) // dtype_size)
    class_shape = (batch_val, classes_val,
                   ceil_x_yolo_pre_detection(boxes_val * grid_h_val * grid_w_val *
                          dtype_size + 32, 32) // dtype_size)
    coords_dic = {'shape': coord_shape, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":coord_shape}
    obj_dic = {'shape': obj_shape, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":obj_shape}
    class_dic = {'shape': class_shape, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":class_shape}

    return {"params": [input_dic, coords_dic, obj_dic, class_dic, boxes_val,
                       coords_val, classes_val, yolo_version, softmax,
                       background, softmaxtree],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_yolo_pre_detection_data(1, 13, 13, "float16", 5, 4, 80, "V2", False, False, False, case_name="yolo_pre_detection_1")
case2 = gen_yolo_pre_detection_data(1, 13, 13, "float16", 5, 4, 80, "V2", True,  False, False, case_name="yolo_pre_detection_2")
case3 = gen_yolo_pre_detection_data(1, 13, 13, "float16", 5, 4, 80, "V2", False, True, False, case_name="yolo_pre_detection_3")
case4 = gen_yolo_pre_detection_data(1, 13, 13, "float16", 5, 4, 80, "V2", True, True, False, case_name="yolo_pre_detection_4")
case5 = gen_yolo_pre_detection_data(1, 1024, 1024, "float16", 5, 4, 80, "V3", False, False, False, case_name="yolo_pre_detection_5")
case6 = gen_yolo_pre_detection_data(1, 1024, 1024, "float16", 5, 4, 80, "V5", False, False, False, case_name="yolo_pre_detection_6")

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], case4)
ut_case.add_case(["Ascend310"], case5)
ut_case.add_case(["Ascend310"], case6)