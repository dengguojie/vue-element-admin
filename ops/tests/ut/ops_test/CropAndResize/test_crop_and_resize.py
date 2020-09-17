#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from common.ut import OpUT

ut_case = OpUT("CropAndResizeD", "impl.crop_and_resize", "crop_and_resize")


def get_dict(_shape, dtype="float16", _format="ND"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": _format, "ori_shape": _shape, "ori_format": "NHWC"}


def get_impl_list(image_shape, image_dtype, boxes_num, crop_size, method="bilinear"):
    N, H, W, C = image_shape
    C0 = 16
    C1 = (C + C0 - 1) // C0
    boxes_shape = [boxes_num ,4]
    boxes_index_shape = [boxes_num]
    input_list = [get_dict([N, C1, H, W, C0], image_dtype, "NC1HWC0"),
                  get_dict(boxes_shape, "float32", "ND"),
                  get_dict(boxes_index_shape, "int32", "ND")
                 ]

    output_shape = [N, C1] + crop_size + [C0]
    output_list = [get_dict(output_shape, "float32", "NC1HWC0")]
    par_list = [crop_size, 0, method]
    return input_list + output_list + par_list


case1 = {"params": get_impl_list([1, 38, 64, 1024], "float16", 100, [14, 14]),
         "case_name": "faster_rcnn_case_1",
         "expect": "success",
         "support_expect": True}
case2 = {"params": get_impl_list([1, 38, 64, 1024], "float32", 100, [14, 14]),
         "case_name": "case_2",
         "expect": "success",
         "support_expect": True}
case3 = {"params": get_impl_list([1, 38, 64, 2048], "float16", 523, [16, 16]),
         "case_name": "case_3",
         "expect": "success",
         "support_expect": True}

case4 = {"params": get_impl_list([1, 38, 64, 2048], "float32", 523, [16, 16]),
         "case_name": "case_4",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], case4)

if __name__ == '__main__':
    ut_case.run()
    exit(0)
