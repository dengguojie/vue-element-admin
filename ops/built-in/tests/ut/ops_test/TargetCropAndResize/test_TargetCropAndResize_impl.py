#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from te import tvm
from op_test_frame.ut import OpUT
ut_case = OpUT("TargetCropAndResize", "impl.target_crop_and_resize", "target_crop_and_resize")


def gen_target_crop_and_resize_case(
    input_shape, boxes_shape, box_index_shape, output_shape, output_h, output_w,
    input_format, case_name_val, expect):
    return {"params": [{"shape": input_shape, "dtype": "uint8", "ori_shape": input_shape, "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": boxes_shape, "dtype": "int32", "ori_shape": boxes_shape, "ori_format": "ND", "format": "ND"},
                       {"shape": box_index_shape, "dtype": "int32", "ori_shape": box_index_shape, "ori_format": "ND", "format": "ND"},
                       {"shape": output_shape, "dtype": "uint8", "ori_shape": output_shape, "ori_format": "NCHW", "format": "NC1HWC0_C04"},
                       output_h, output_w, input_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case(["Hi3796CV300CS"],
                 gen_target_crop_and_resize_case(
                     (1,3,224,224), (5,4), [5], (5,1,100,120,4), 100, 120,
                     "YUV420SP_U8", "target_crop_and_resize_1", "success"))
ut_case.add_case(["Ascend910A"],
                 gen_target_crop_and_resize_case(
                     (1,3,224,224), (5,4), [5], (5,1,100,120,4), 100, 120,
                     "YUV420SP_U8", "target_crop_and_resize_2", RuntimeError))


from impl.target_crop_and_resize import target_crop_and_resize_compute
def test_ihisi_target_crop_and_resize_compute(test_arg):
    x_shape = (1,3,224,224)
    x_shape2 = (1,3,1920,1088)
    boxes_shape = (5,4)
    box_index_shape = (5,)
    x_dic = {"shape": x_shape, "dtype": "uint8", "ori_shape": x_shape, "ori_format": "NCHW", "format": "NCHW"}
    x_dic2 = {"shape": x_shape2, "dtype": "uint8", "ori_shape": x_shape2, "ori_format": "NCHW", "format": "NCHW"}
    boxes_dic = {"shape": boxes_shape, "dtype": "int32", "ori_shape": boxes_shape, "ori_format": "ND", "format": "ND"}
    box_index_dic = {"shape": box_index_shape, "dtype": "int32", "ori_shape": box_index_shape, "ori_format": "ND", "format": "ND"}
    y_dic = {"shape": (5,1,100,120,4), "dtype": "uint8", "ori_shape": (1,4,100,120), "ori_format": "NCHW", "format": "NC1HWC0_C04"}
    y_dic2 = {"shape": (5,1,640,480,4), "dtype": "uint8", "ori_shape": (1,4,640,480), "ori_format": "NCHW", "format": "NC1HWC0_C04"}
    output_h = 100
    output_w = 120
    input_format = "YUV420SP_U8"

    boxes = tvm.placeholder(boxes_shape, name='boxes', dtype="int32")
    box_index = tvm.placeholder(box_index_shape, name='box_index', dtype="int32")

    x = tvm.placeholder(x_shape, name='x', dtype="uint8")
    target_crop_and_resize_compute(x, boxes, box_index, x_dic, y_dic, input_format)

    x2 = tvm.placeholder(x_shape2, name='x', dtype="uint8")
    target_crop_and_resize_compute(x2, boxes, box_index, x_dic2, y_dic2, input_format)

ut_case.add_cust_test_func(test_func=test_ihisi_target_crop_and_resize_compute)


if __name__ == '__main__':
    ut_case.run()

