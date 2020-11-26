# -*- coding:utf-8 -*-
import unittest
import sys
#sys.path.append("./llt/ops/st_all/cce_all/testcase_python")
from impl.yolo_v3_detection_output_v2 import yolo_v3_detection_output_v2
import os
import shutil
from run_testcase import run_testcase,get_path_val,print_func_name
from te import tik
from te import platform as tbe_platform

TEST_BIASES = [116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119,
               10, 13, 16, 30, 33, 23]
TEST_BIASES_YOLOV2 = [0.572730, 0.677385, 1.874460, 2.062530, 3.338430,
                       5.474340, 7.882820, 3.527780, 9.770520, 9.168280]
testcases = {
    "op_name": "yolo_v3_detection_output_v2",
    # batch,[[h1,w1],[h2,w2],[h3,w3]],dtype,boxes,classes,relative,
    # obj_threshold,classes_threshold,nms_threshold,biases,
    # resize_origin_img_to_net, N
    "all": {
            "yolo_v3_detection_output_v2_16_16_16": (
                1, [[4, 4], [4, 4], [4, 4]], "float16", 3, 2, True, 0.5,
                0.5, 0.45, TEST_BIASES, True, 10,
                "yolo_v3_detection_output_v2_16_16_16", "Ascend310"),
            "yolo_v3_detection_output_v2_16": (
                1, [[4, 4]], "float16", 5, 2, True, 0.5,
                0.5, 0.45, TEST_BIASES_YOLOV2, True, 4,
                "yolo_v3_detection_output_v2_16", "Ascend310"),
            "yolo_v3_detection_output_v2_169_169_169": (
                1, [[13, 13], [13, 13], [13, 13]], "float16", 3, 1, True, 0.5,
                0.5, 0.45, TEST_BIASES, False, 10,
                "yolo_v3_detection_output_v2_169_169_169", "Ascend310"),
            "yolo_v3_detection_output_v2_169": (
                1, [[13, 13]], "float16", 5, 2, True, 0.5,
                0.5, 0.45, TEST_BIASES_YOLOV2, False, 4,
                "yolo_v3_detection_output_v2_169", "Ascend310"),
            "yolo_v3_detection_output_v2_361": (
                1, [[19, 19]], "float16", 5, 10, True, 0.5,
                0.5, 0.45, TEST_BIASES_YOLOV2, True, 4,
                "yolo_v3_detection_output_v2_361", "Ascend310"),
            "yolo_v3_detection_output_v2_169_676_2704": (
                1, [[13, 13], [26, 26], [52, 52]], "float16", 3, 20, True, 0.5,
                0.5, 0.45, TEST_BIASES, True, 10,
                "yolo_v3_detection_output_v2_169_676_2704", "Ascend310")
    },
    "mini": {},
    "lite": {},
    "cloud": {},
    "tiny": {},
}

bin_path_val = get_path_val(testcases)


def test_yolo_v3_detection_output_v2(
        batch, box_info, dtype, boxes, classes, relative, obj_threshold,
        classes_threshold, nms_threshold, biases, resize_origin_img_to_net, N,
        kernel_name_val, product):
    tbe_platform.cce_conf.te_set_version(product)
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

    post_top_k = pre_nms_topn = 1024

    yolo_v3_detection_output_v2(coord_data + obj_data +
                                classes_data + [img_info_dict] +
                                windex + hindex,
                                box_out_dict, box_out_num_dict,
                                biases,
                                boxes, coords=4, classes=classes,
                                relative=relative,
                                obj_threshold=obj_threshold,
                                score_threshold=classes_threshold,
                                post_nms_topn=post_top_k,
                                iou_threshold=nms_threshold,
                                pre_nms_topn=pre_nms_topn,
                                N=N,
                                resize_origin_img_to_net=resize_origin_img_to_net,
                                kernel_name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"
    if (os.path.isfile(kernel_meta_path + lib_kernel_name)):
        shutil.move(kernel_meta_path + lib_kernel_name,
                    bin_path_val + "/" + lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o",
                    bin_path_val + "/" + kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json",
                    bin_path_val + "/" + kernel_name_val + ".json")


class TestYoloDetectionOutputV2Cce(unittest.TestCase):
    def tearDown(self):
        pass

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def setUpClass(self):
        pass

    @print_func_name
    def test_cce_yolo_v3_detection_output_v2(self):
        run_testcase(testcases, test_yolo_v3_detection_output_v2)


def main():
    unittest.main()
    exit(0)


if __name__ == "__main__":
    main()
