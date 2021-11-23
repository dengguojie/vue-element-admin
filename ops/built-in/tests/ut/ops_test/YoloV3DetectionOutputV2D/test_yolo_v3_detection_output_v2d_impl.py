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

YoloV3DetectionOutputV2 ut case
"""
from op_test_frame.ut import OpUT
from impl.yolo_v3_detection_output_v2d import get_op_support_info
ut_case = OpUT("YoloV3DetectionOutputV2D", "impl.yolo_v3_detection_output_v2d", "yolo_v3_detection_output_v2d")

TEST_BIASES = [116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119,
               10, 13, 16, 30, 33, 23]

def common_cce(batch, box_info, dtype, boxes, classes, relative, obj_threshold,
               classes_threshold, nms_threshold, biases, resize_origin_img_to_net,
               kernel_name_val, product, cords=4,
               pre_nms_topn=1024, post_top_k=1024):
    # tbe_platform.cce_conf.te_set_version(product)

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
    print("588888", x, box_out_dict, box_out_num_dict, biases)
    split_info = get_op_support_info(x, box_out_dict, box_out_num_dict, biases)
    print(split_info)
    return{"params": [x,
                      box_out_dict, box_out_num_dict, biases
                      ],
           "case_name": "case1",
           "expect": "success",
           "support_expect": True}


case1 = common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float32", "Ascend910")

ut_case.add_case(["Ascend910A"], case1)

def test_yolo_v3_detection_output_v2d_001(test_arg):
    from impl.yolo_v3_detection_output_v2d import yolo_v3_detection_output_v2d
    from tbe.common.platform.platform_info import set_current_compile_soc_info
    set_current_compile_soc_info("Ascend710")
    yolo_v3_detection_output_v2d(*(common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float16", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(64, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float16", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(60, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float16", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(2, [[32, 40], [6, 6], [6, 6]], "float32", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float32", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(2, [[6, 6], [32, 42], [6, 6]], "float32", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float32", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(2, [[33, 41], [32, 42], [33, 41]], "float32", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float32", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(2, [[61, 63], [61, 63], [61, 63]], "float32", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float32", "Ascend710").get("params")))

    yolo_v3_detection_output_v2d(*(common_cce(2, [[60, 64], [60, 64], [60, 64]], "float32", 1, 2, True, 0.5, 0.5, 0.45,
                                              TEST_BIASES, True, "test_yolo_v3_float32", "Ascend710").get("params")))


ut_case.add_cust_test_func(test_func=test_yolo_v3_detection_output_v2d_001)

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)





