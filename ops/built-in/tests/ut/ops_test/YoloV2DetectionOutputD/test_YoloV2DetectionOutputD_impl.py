#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("YoloV2DetectionOutputD", None, None)

def gen_data(batch, h, w, dtype, boxes, classes, relative, case_name):
    coord_data_dic = {"shape":(batch,4,1),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,1),"ori_format": "NCHW"}
    obj_prob_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_prob_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    img_info_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    windex_dic = {"shape":(h, w),"dtype": dtype,"format": "NCHW","ori_shape":(h, w),"ori_format": "NCHW"}
    hindex_dic = {"shape":(h, w),"dtype": dtype,"format": "NCHW","ori_shape":(h, w),"ori_format": "NCHW"}
    box_out_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    box_out_num_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    biases = (0.572730,0.677385,1.874460,2.062530,3.338430,5.474340,7.882820,3.527780,9.770520,9.168280)

    return {"params": [coord_data_dic, obj_prob_dic, classes_prob_dic, img_info_dic,
                       windex_dic, hindex_dic, box_out_dic, box_out_num_dic, biases],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

def gen_data_err(batch, h, w, dtype, boxes, classes, relative, case_name):
    coord_data_dic = {"shape":(batch,4,1),"dtype": dtype,"format": "NCHW","ori_shape":(batch,4,1),"ori_format": "NCHW"}
    obj_prob_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    classes_prob_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    img_info_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    windex_dic = {"shape":(h, w),"dtype": dtype,"format": "NCHW","ori_shape":(h, w),"ori_format": "NCHW"}
    hindex_dic = {"shape":(h, w),"dtype": dtype,"format": "NCHW","ori_shape":(h, w),"ori_format": "NCHW"}
    box_out_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    box_out_num_dic = {"shape":(),"dtype": dtype,"format": "NCHW","ori_shape":(),"ori_format": "NCHW"}
    biases = (0.572730,0.677385,1.874460,2.062530,3.338430,5.474340,7.882820,3.527780,9.770520,9.168280)

    return {"params": [coord_data_dic, obj_prob_dic, classes_prob_dic, img_info_dic,
                       windex_dic, hindex_dic, box_out_dic, box_out_num_dic, biases],
            "case_name": case_name,
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

case2 = gen_data(2, 19, 19, "float16", 5, 5, True, "yolo_v2_detection_output_d_2")
case3 = gen_data(34, 200, 200, "float16", 5, 5, True, "yolo_v2_detection_output_d_3")
case1 = gen_data(34, 6, 7, "float16", 5, 5, True, "yolo_v2_detection_output_d_1")
case4 = gen_data_err(34, 1, 7, "float16", 5, 5, True, "yolo_v2_detection_output_d_1")


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case4)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

def test_yolo_v2_detection_output_d_001(test_arg):
    from impl.yolo_v2_detection_output_d import yolo_v2_detection_output_d
    from tbe.common.platform.platform_info import set_current_compile_soc_info
    set_current_compile_soc_info("Ascend710")
    yolo_v2_detection_output_d(*(
        gen_data(2, 19, 19, "float16", 5, 5, True, "yolo_v2_detection_output_d_1").get("params")))
    yolo_v2_detection_output_d(*(
        gen_data(34, 6, 7, "float16", 5, 5, True, "yolo_v2_detection_output_d_2").get("params")))
    yolo_v2_detection_output_d(*(
        gen_data(34, 200, 200, "float16", 5, 5, True, "yolo_v2_detection_output_d_3").get("params")))
    yolo_v2_detection_output_d(*(
        gen_data(34, 201, 201, "float16", 5, 5, True, "yolo_v2_detection_output_d_4").get("params")))


ut_case.add_cust_test_func(test_func=test_yolo_v2_detection_output_d_001)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
    exit(0)
