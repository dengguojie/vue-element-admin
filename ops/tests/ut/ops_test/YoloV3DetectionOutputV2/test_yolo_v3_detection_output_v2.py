"""
Copyright (C) 2016. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

yolo_v3_test ut case
"""
import unittest
import time
import sys
from te import tvm
from te import tik
from impl.yolo_v3_detection_output_v2 import yolo_v3_detection_output_v2
from impl.yolo_v3_cls_prob_v2 import ClsProbComputer
from te import platform as tbe_platform

TEST_BIASES = [116, 90, 156, 198, 373, 326, 30, 61, 62, 45, 59, 119,
               10, 13, 16, 30, 33, 23]


def print_func_name(func):

    def wrapper(*args, **kwargs):
        print("[ RUN        ] %s" %func.__name__)
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        msecs = (end_time - start_time) * 1000
        print("[ END        ] %s -> elapsed time: %f ms" %(func.__name__, msecs))
    return wrapper


def common_cce(batch, box_info, dtype, boxes, classes, relative, obj_threshold,
               classes_threshold, nms_threshold, biases, resize_origin_img_to_net,
               kernel_name_val, product, cords=4,
               pre_nms_topn=1024, post_top_k=1024):
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



    yolo_v3_detection_output_v2(coord_data + obj_data +
                                classes_data + [img_info_dict] +
                                windex + hindex,
                                box_out_dict, box_out_num_dict,
                                biases,
                                boxes, coords=cords, classes=classes,
                                relative=relative,
                                obj_threshold=obj_threshold,
                                score_threshold=classes_threshold,
                                post_nms_topn=post_top_k,
                                iou_threshold=nms_threshold,
                                pre_nms_topn=pre_nms_topn,
                                N=10,
                                resize_origin_img_to_net=resize_origin_img_to_net,
                                kernel_name=kernel_name_val)


class TestYoloDetectionOutput(unittest.TestCase):
    """Configuration scope to test tf_round_cce option.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use build_config instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    def tearDown(self):
        # Do this after each test case is executed.
        pass

    def setUp(self):
        # Do this before each test case is executed.
        self.ci_group_info_dict = {"krin": {},
                                   "balong_gux": {}}
        self.project_info_dict = {
            "kirin": [{"status": u"NEW", "files": [u"ci_scm/scm_common.py"]}]}

    @classmethod
    def tearDownClass(self):
        # Must use the @classmethod decorator, run once after all tests have run
        print("")
        print("---------------------------------------------------")

    @classmethod
    def setUpClass(self):
        # Must use the @classmethod decorator, run once before all tests have run
        print("---------------------------------------------------")

    @print_func_name
    def test_yolo_v3_common_123(self):
        infos = [[100, 10], [100, 10], [780, 4]]
        batch = 1
        coords = 4
        boxes = 1
        classes = 1
        relative = True
        N = 10
        dtype = "float32"
        yolo_num = int((N - 1) / 3)
        obj_threshold = 0.5
        score_threshold = 0.5
        post_nms_topn = 1024
        iou_threshold = 0.45
        pre_nms_topn = 512
        box_info = []
        kernel_name = "common"
        resize_origin_img_to_net = True
        biases_list = []
        for i in range(yolo_num):
            info = infos[i]
            h = info[0]
            w = info[1]
            box_info.append({"shape": (batch, boxes * (coords + 1 + classes),
                                       h, w),
                             "dtype": dtype, "format": "NCHW"})
            biases_list.append(TEST_BIASES[i * 2 * boxes: i * 2 * boxes +
                                                          2 * boxes])
        input_dict = {
            "box_info": box_info,
            "biases": biases_list,
            "coords": coords,
            "boxes": boxes,
            "classes": classes,
            "relative": relative,
            "obj_threshold": obj_threshold,
            "classes_threshold": score_threshold,
            "post_top_k": post_nms_topn,
            "nms_threshold": iou_threshold,
            "pre_nms_topn": pre_nms_topn,
            "max_box_number_per_batch": post_nms_topn,
            "resize_origin_img_to_net": resize_origin_img_to_net,
            "kernel_name": kernel_name,
        }

        detection_output = ClsProbComputer(input_dict)
        detection_output.set_dsize(2)
        assert(detection_output.get_dtype() == dtype)
        detection_output.set_pre_nms_topn(pre_nms_topn)
        detection_output.get_shape((1,), need_low_dim=False)
        detection_output.get_shape((3, 4), need_low_dim=True)
        detection_output.get_shape((4, 4), need_low_dim=False)

    @print_func_name
    def test_yolo_v3_check_param(self):
        try:
            common_cce(1, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                       0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_check_param", "Ascend910",
                       pre_nms_topn=1025)
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as error:
            print("error:%s" % error)
        try:
            common_cce(1, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                       0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_check_param", "Ascend910",
                       pre_nms_topn=1024, post_top_k=1025)
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as error:
            print("error:%s" % error)
        common_cce(1, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_check_param", "Ascend910",
                   pre_nms_topn=1024, post_top_k=15)

    @print_func_name
    def test_yolo_v3_float32(self):
        try:
            common_cce(1, [[6, 6], [6, 6], [6, 6]], "float32", 1, 2, True,
                       0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float32", "Ascend910")
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)

    @print_func_name
    def test_yolo_v3_float32(self):
        # TODO support float32
        common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float32", "Ascend910")

    @print_func_name
    def test_yolo_v3_float16(self):
        common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float16", "Ascend910")

    @print_func_name
    def test_yolo_v3_float16_coords_invalid(self):
        try:
            common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                       0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float16", "Ascend910", cords=5)
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)

    @print_func_name
    def test_yolo_v3_last_obj_class(self):
        common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                   0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float16", "Ascend710",
                   cords=4, pre_nms_topn=100)

#    @print_func_name
#    def test_yolo_v3_float16_bigshape(self):
#        tik.api.tik_conf.set_product_version("1.60.xx.xx")
#        batch = 2
#        h1 = 100
#        w1 = 100
#        h2 = 200
#        w2 = 200
#        h3 = 300
#        w3 = 400
#        n = boxes = 1
#        classes = 2
#        dtype = "float16"
#        relative = True
#        cords = 4
#        obj_threshold = 0.5
#        classes_threshold = 0.5
#        post_top_k = 1024
#        nms_threshold = 0.45
#        pre_nms_topn = 512
#        max_box_number_per_batch = 1024
#        biases1 = (116,90,156,198,373,326)
#        biases2 = (30,61,62,45,59,119)
#        biases3 = (10,13,16,30,33,23)
#        kernel_name_val="test_yolo_v3_float16_bigshape"
#        common_cce(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
#                   obj_threshold, classes_threshold, nms_threshold, biases1,
#                   biases2, biases3, post_top_k, pre_nms_topn,
#                   max_box_number_per_batch, kernel_name_val)
#
#    @print_func_name
#    def test_yolo_v3_float32_bigshape(self):
#        tbe_platform.cce_conf.te_set_version("Ascend610")
#        batch = 33
#        h1 = 101
#        w1 = 101
#        h2 = 201
#        w2 = 201
#        h3 = 301
#        w3 = 401
#        n = boxes = 1
#        classes = 2
#        dtype = "float32"
#        relative = True
#        cords = 4
#        obj_threshold = 0.5
#        classes_threshold = 0.5
#        post_top_k = 1024
#        nms_threshold = 0.45
#        pre_nms_topn = 512
#        max_box_number_per_batch = 1024
#        biases1 = (116,90,156,198,373,326)
#        biases2 = (30,61,62,45,59,119)
#        biases3 = (10,13,16,30,33,23)
#        kernel_name_val="test_yolo_v3_float32_bigshape"
#        common_cce(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
#                   obj_threshold, classes_threshold, nms_threshold, biases1,
#                   biases2, biases3, post_top_k, pre_nms_topn,
#                   max_box_number_per_batch, kernel_name_val)

#    @print_func_name
#    def test_yolo_v3_float16_bigshape_test2(self):
#        tbe_platform.cce_conf.te_set_version("Ascend610")
#        batch = 33
#        h1 = 101
#        w1 = 101
#        h2 = 201
#        w2 = 201
#        h3 = 301
#        w3 = 401
#        n = boxes = 1
#        classes = 1
#        dtype = "float16"
#        relative = True
#        cords = 4
#        obj_threshold = 0.5
#        classes_threshold = 0.5
#        post_top_k = 1024
#        nms_threshold = 0.45
#        pre_nms_topn = 512
#        max_box_number_per_batch = 1024
#        biases1 = (116,90,156,198,373,326)
#        biases2 = (30,61,62,45,59,119)
#        biases3 = (10,13,16,30,33,23)
#        kernel_name_val="test_yolo_v3_float16_bigshape"
#        common_cce(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
#                   obj_threshold, classes_threshold, nms_threshold, biases1,
#                   biases2, biases3, post_top_k, pre_nms_topn,
#                   max_box_number_per_batch, kernel_name_val)

    @print_func_name
    def test_yolo_v3_float16_pre_nms_topn_invalid(self):
        try:
            common_cce(2, [[6, 6], [6, 6], [6, 6]], "float16", 1, 2, True,
                       0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float16_pre_nms_topn_invalid",
                       "Ascend910", pre_nms_topn=1028)
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)

    @print_func_name
    def test_yolo_v3_float32_pre_nms_topn_invalid(self):
        # TODO not support float32
        try:
            common_cce(2, [[6, 6], [6, 6], [6, 6]], "float32", 1, 2, True,
                       0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float32_pre_nms_topn_invalid",
                       "Ascend910", pre_nms_topn=1028)
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)



    # @print_func_name
    # def test_yolo_v3_float32_max_box_number_per_batch_invalid(self):
    #     tbe_platform.cce_conf.te_set_version("Ascend610")
    #     batch = 2
    #     h1 = 6
    #     w1 = 6
    #     h2 = 6
    #     w2 = 6
    #     h3 = 6
    #     w3 = 6
    #     n = boxes = 1
    #     classes = 2
    #     dtype = "float32"
    #     relative = True
    #     cords = 4
    #     obj_threshold = 0.5
    #     classes_threshold = 0.5
    #     post_top_k = 1024
    #     nms_threshold = 0.45
    #     pre_nms_topn = 512
    #     max_box_number_per_batch = 1029
    #     biases1 = (116,90,156,198,373,326)
    #     biases2 = (30,61,62,45,59,119)
    #     biases3 = (10,13,16,30,33,23)
    #     kernel_name_val="test_yolo_v3_float32_max_box_number_per_batch_invalid"
    #     try:
    #         common_cce(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
    #                    obj_threshold, classes_threshold, nms_threshold, biases1,
    #                    biases2, biases3, post_top_k, pre_nms_topn,
    #                    max_box_number_per_batch, kernel_name_val,cords)
    #         raise AssertionError("%s.%s should throw an exception!" % (
    #             self.__class__.__name__, sys._getframe().f_code.co_name))
    #     except RuntimeError as e:
    #         print("error:%s" % e)

#    @print_func_name
#    def test_yolo_v3_float32_bigshape32(self):
#        tbe_platform.cce_conf.te_set_version("Ascend610")
#        batch = 1
#        h1 = 100
#        w1 = 10
#        h2 = 100
#        w2 = 10
#        h3 = 780
#        w3 = 4
#        n = boxes = 1
#        classes = 1
#        dtype = "float32"
#        relative = True
#        cords = 4
#        obj_threshold = 0.5
#        classes_threshold = 0.5
#        post_top_k = 1024
#        nms_threshold = 0.45
#        pre_nms_topn = 512
#        max_box_number_per_batch = 1024
#        biases1 = (116,90,156,198,373,326)
#        biases2 = (30,61,62,45,59,119)
#        biases3 = (10,13,16,30,33,23)
#        kernel_name_val="test_yolo_v3_float32_bigshape32"
#        common_cce(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
#                   obj_threshold, classes_threshold, nms_threshold, biases1,
#                   biases2, biases3, post_top_k, pre_nms_topn,
#                   max_box_number_per_batch, kernel_name_val)

#    @print_func_name
#    def test_yolo_v3_float16_his_es(self):
#        tik.api.tik_conf.set_product_version("5.10.xx.xx")
#        batch = 2
#        h1 = 6
#        w1 = 6
#        h2 = 6
#        w2 = 6
#        h3 = 6
#        w3 = 6
#        n = boxes = 1
#        classes = 2
#        dtype = "float16"
#        relative = False
#        cords = 4
#        obj_threshold = 0.5
#        classes_threshold = 0.5
#        post_top_k = 1024
#        nms_threshold = 0.45
#        pre_nms_topn = 512
#        max_box_number_per_batch = 1024
#        biases1 = (116,90,156,198,373,326)
#        biases2 = (30,61,62,45,59,119)
#        biases3 = (10,13,16,30,33,23)
#        kernel_name_val="test_yolo_v3_float16"
#        common_cce(batch, h1, w1, h2, w2, h3, w3, dtype, boxes, classes, relative,
#                   obj_threshold, classes_threshold, nms_threshold, biases1,
#                   biases2, biases3, post_top_k, pre_nms_topn,
#                   max_box_number_per_batch, kernel_name_val)

    @print_func_name
    def test_yolo_v3_float16_Aicore(self):
        common_cce(2, [[600, 600], [100, 200], [400, 500]], "float16",
                   1, 2, False, 0.5, 0.5, 0.45, TEST_BIASES, True,
                   "test_yolo_v3_float16", "Ascend910")

    @print_func_name
    def test_yolo_v3_float32_box1_height_weight_invalid(self):
        # TODO support fp32
        try:
            common_cce(2, [[1, 1], [6, 6], [6, 6]], "float16",
                       1, 2, True, 0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float32_box1_height_weight_invalid",
                       "Ascend910")
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)


    @print_func_name
    def test_yolo_v3_float32_box2_height_weight_invalid(self):
        # TODO support fp32
        try:
            common_cce(2, [[6, 6], [1, 2], [6, 6]], "float16",
                       1, 2, True, 0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float32_box2_height_weight_invalid",
                       "Ascend910")
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)

    @print_func_name
    def test_yolo_v3_float32_box3_height_weight_invalid(self):
        # TODO support fp32
        try:
            common_cce(2, [[6, 6], [6, 6], [3, 2]], "float16",
                       1, 2, True, 0.5, 0.5, 0.45, TEST_BIASES, True,
                       "test_yolo_v3_float32_box3_height_weight_invalid",
                       "Ascend910")
            raise AssertionError("%s.%s should throw an exception!" % (
                self.__class__.__name__, sys._getframe().f_code.co_name))
        except RuntimeError as e:
            print("error:%s" % e)


def main():
    unittest.main()
    exit(0)


if __name__ == "__main__":
    main()
