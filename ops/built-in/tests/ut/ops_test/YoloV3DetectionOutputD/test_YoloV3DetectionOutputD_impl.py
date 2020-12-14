#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("YoloV3DetectionOutputD", None, None)

def gen_data(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
             obj_threshold, classes_threshold, nms_threshold, biases1,
             biases2, biases3, post_top_k, pre_nms_topn,
             max_box_number_per_batch, kernel_name_val,cords=4):
    coord_data1_dict = {"shape":(batch,4,h1*w1),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,h1*w1),"ori_format": "NCHW"}
    coord_data2_dict = {"shape":(batch,4,h2*w2),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,h2*w2),"ori_format": "NCHW"}
    coord_data3_dict = {"shape":(batch,4,h3*w3),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,h3*w3),"ori_format": "NCHW"}
    obj_data1_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    obj_data2_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    obj_data3_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_data1_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_data2_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_data3_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    img_info_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    windex1_dict = {"shape":(h1, w1), "dtype": dtype,"format": "NCHW","ori_shape":(h1,w1),"ori_format": "NCHW"}
    windex2_dict = {"shape":(h2, w2), "dtype": dtype,"format": "NCHW","ori_shape":(h2,w2),"ori_format": "NCHW"}
    windex3_dict = {"shape":(h3, w3), "dtype": dtype,"format": "NCHW","ori_shape":(h3,w3),"ori_format": "NCHW"}

    hindex1_dict = {"shape":(h1, w1), "dtype": dtype,"format": "NCHW","ori_shape":(h1,w1),"ori_format": "NCHW"}
    hindex2_dict = {"shape":(h2, w2), "dtype": dtype,"format": "NCHW","ori_shape":(h2,w2),"ori_format": "NCHW"}
    hindex3_dict = {"shape":(h3, w3), "dtype": dtype,"format": "NCHW","ori_shape":(h3,w3),"ori_format": "NCHW"}
    box_out_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    box_out_num_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}


    return {"params": [coord_data1_dict, coord_data2_dict, coord_data3_dict, obj_data1_dict,
                       obj_data2_dict, obj_data3_dict, classes_data1_dict, classes_data2_dict,
                       classes_data3_dict, img_info_dict, windex1_dict, windex2_dict,
                       windex3_dict, hindex1_dict, hindex2_dict, hindex3_dict, box_out_dict,
                       box_out_num_dict, biases1, biases2, biases3],
            "case_name": "yolo_v3_detection_output_d_1",
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

def gen_data_err(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
             obj_threshold, classes_threshold, nms_threshold, biases1,
             biases2, biases3, post_top_k, pre_nms_topn,
             max_box_number_per_batch, kernel_name_val,cords=4):
    coord_data1_dict = {"shape":(batch,4,h1*w1),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,h1*w1),"ori_format": "NCHW"}
    coord_data2_dict = {"shape":(batch,4,h2*w2),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,h2*w2),"ori_format": "NCHW"}
    coord_data3_dict = {"shape":(batch,4,h3*w3),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,h3*w3),"ori_format": "NCHW"}
    obj_data1_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    obj_data2_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    obj_data3_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_data1_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_data2_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_data3_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    img_info_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    windex1_dict = {"shape":(h1, w1), "dtype": dtype,"format": "NCHW","ori_shape":(h1,w1),"ori_format": "NCHW"}
    windex2_dict = {"shape":(h2, w2), "dtype": dtype,"format": "NCHW","ori_shape":(h2,w2),"ori_format": "NCHW"}
    windex3_dict = {"shape":(h3, w3), "dtype": dtype,"format": "NCHW","ori_shape":(h3,w3),"ori_format": "NCHW"}

    hindex1_dict = {"shape":(h1, w1), "dtype": dtype,"format": "NCHW","ori_shape":(h1,w1),"ori_format": "NCHW"}
    hindex2_dict = {"shape":(h2, w2), "dtype": dtype,"format": "NCHW","ori_shape":(h2,w2),"ori_format": "NCHW"}
    hindex3_dict = {"shape":(h3, w3), "dtype": dtype,"format": "NCHW","ori_shape":(h3,w3),"ori_format": "NCHW"}
    box_out_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    box_out_num_dict = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}


    return {"params": [coord_data1_dict, coord_data2_dict, coord_data3_dict, obj_data1_dict,
                       obj_data2_dict, obj_data3_dict, classes_data1_dict, classes_data2_dict,
                       classes_data3_dict, img_info_dict, windex1_dict, windex2_dict,
                       windex3_dict, hindex1_dict, hindex2_dict, hindex3_dict, box_out_dict,
                       box_out_num_dict, biases1, biases2, biases3],
            "case_name": "yolo_v3_detection_output_d_2",
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

biases1 = (116.0,90.0,156.0,198.0,373.0,326.0)
biases2 = (30.0,61.0,62.0,45.0,59.0,119.0)
biases3 = (10.0,13.0,16.0,30.0,33.0,23.0)
case1 = gen_data(2, 6, 6, 6, 6, 6, 6, "float16", 1, 2, True, 0.5, 0.5, 0.45,
                 biases1, biases2, biases3, 1024, 512, 1024, 4)
case2 = gen_data_err(2, 6, 1, 6, 6, 6, 6, "float16", 1, 2, True, 0.5, 0.5, 0.45,
                 biases1, biases2, biases3, 1024, 512, 1024, 4)
case3 = gen_data_err(2, 6, 6, 6, 1, 6, 6, "float16", 1, 2, True, 0.5, 0.5, 0.45,
                 biases1, biases2, biases3, 1024, 512, 1024, 4)
case4 = gen_data_err(2, 6, 6, 6, 6, 6, 1, "float16", 1, 2, True, 0.5, 0.5, 0.45,
                 biases1, biases2, biases3, 1024, 512, 1024, 4)


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend910"], case2)
ut_case.add_case(["Ascend910"], case3)
ut_case.add_case(["Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)