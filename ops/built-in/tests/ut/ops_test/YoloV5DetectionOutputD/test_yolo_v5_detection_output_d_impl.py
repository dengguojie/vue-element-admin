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

YoloV5DetectionOutputD ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("YoloV5DetectionOutputD", "impl.yolo_v5_detection_output_d", "yolo_v5_detection_output_d")

TEST_BIASES = [116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119,
               10, 13, 16, 30, 33, 23]

def common_cce(batch, box_info, dtype, boxes, classes, relative, obj_threshold,
               classes_threshold, nms_threshold, biases, resize_origin_img_to_net,
               kernel_name_val, product, cords=4, pre_nms_topn=1024, post_top_k=1024, 
               case_name="case1", expect="success"):

    coord_data = []
    obj_data = []
    classes_data = []
    windex = []
    hindex = []

    for info in box_info:
        h1 = info[0]
        w1 = info[1]
        coord_data.append({"shape": (batch, 4, h1*w1), "dtype": dtype,
                           "format": "NCHW", "ori_shape": (batch, 4, h1*w1),
                           "ori_format": "NCHW"})
        obj_data.append({"shape": (), "dtype": dtype, "format": "NCHW",
                         "ori_shape": (), "ori_format": "NCHW"})
        classes_data.append({"shape": (), "dtype": dtype, "format": "NCHW",
                             "ori_shape": (), "ori_format": "NCHW"})
        windex.append({"shape": (h1, w1), "dtype": dtype, "format": "NCHW",
                       "ori_shape": (h1, w1), "ori_format": "NCHW"})
        hindex.append({"shape": (h1, w1), "dtype": dtype, "format": "NCHW",
                       "ori_shape": (h1, w1), "ori_format": "NCHW"})

    img_info_dict = {"shape": (), "dtype": dtype, "format": "NCHW",
                     "ori_shape": (), "ori_format": "NCHW"}
    box_out_dict = {"shape": (), "dtype": dtype, "format": "NCHW",
                    "ori_shape": (), "ori_format": "NCHW"}
    box_out_num_dict = {"shape": (), "dtype": dtype, "format": "NCHW",
                        "ori_shape": (), "ori_format": "NCHW"}

    x = coord_data + obj_data + classes_data + [img_info_dict] + windex + hindex

    return {"params": [x, box_out_dict, box_out_num_dict, biases],
            "case_name": case_name,
            "expect": expect,
            "support_expect": True}

case1 = common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float32", "Ascend310", case_name="test_yolo_v5_d_1",
                   expect="success")

case2 = common_cce(2, [[90, 90], [90, 90], [90, 90]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float32", "Ascend310", case_name="test_yolo_v5_d_2",
                   expect="success")

case3 = common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float32", "Ascend710", case_name="test_yolo_v5_d_3",
                   expect="faild")

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend710"], case3)