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

dynamic BatchMultiClassNonMaxSuppression ut case
"""
from op_test_frame.ut import OpUT
ut_case = OpUT("BatchMultiClassNonMaxSuppression", "impl.dynamic.batch_multi_class_non_max_suppression", "batch_multi_class_non_max_suppression")


def get_dict(_shape, _range, dtype="float16"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND", "range": _range}


def get_impl_list(batch_size, num_boxes, num_class, num_class_boxes,
                  score_threshold, iou_threshold, max_size_per_class,
                  max_total_size, change_coordinate_frame, is_need_clip,
                  is_need_valid):
    if num_class_boxes == 0:
        boxes_shape = [batch_size, 4, num_boxes]
        boxes_range = [(1, None), (4,4), (1, None)]
    else:
        boxes_shape = [batch_size, num_class_boxes, 4, num_boxes]
        boxes_range = [(1, None), (1, None), (4,4), (1, None)]
    if num_class == 0:
        score_shape = [batch_size, num_boxes]
        score_range = [(1, None), (1, None)]
    else:
        score_shape = [batch_size, num_class, num_boxes]
        score_range = [(1, None), (1, None), (1, None)]
    if is_need_clip:
        clip_shape = [batch_size, 4]
        clip_range = [(1, None), (4,4)]
    else:
        clip_shape = None
        clip_range = [(1, None), (4,4)]
    if is_need_valid:
        num_shape = [batch_size]
        num_range = [(1, None)]
    else:
        num_shape = None
        num_range = [(1, None)]

    output_boxes_shape = [batch_size, 4, max_total_size]
    output_boxes_range = [(1, None), (4,4), (1, None)]
    output_score_shape = [batch_size, max_total_size]
    output_score_range = [(1, None), (1,None)]
    output_valid_shape = [batch_size]
    output_valid_range = [(1, None)]

    input_list = [get_dict(boxes_shape, boxes_range),
                  get_dict(score_shape, score_range),
                  get_dict(clip_shape, clip_range),
                  get_dict(num_shape, num_range)]
    output_list = [get_dict(output_boxes_shape, output_boxes_range),
                   get_dict(output_score_shape, output_score_range),
                   get_dict(output_score_shape, output_score_range),
                   get_dict(output_valid_shape, output_valid_range)]
    par_list = [score_threshold, iou_threshold, max_size_per_class,
                max_total_size, change_coordinate_frame, False]

    return input_list + output_list + par_list


case1 = {"params": get_impl_list(-1, -1, -1, -1, 0, 0.7, 25, 25, False, False, False),
         "case_name": "dynamic_faster_rcnn_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": get_impl_list(-1, -1, -1, -1, 0, 0.7, 25, 25, False, False, False),
         "case_name": "dynamic_faster_rcnn_case_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": get_impl_list(-1, -1, -1, -1, 0, 0.7, 10, 10, False, False, False),
         "case_name": "dynamic_ssd_mobile_1",
         "expect": "success",
         "support_expect": True}

case4 = {"params": get_impl_list(-1, -1, -1, 0, 0, 0.7, 20, 20, False, False, False),
         "case_name": "dynamic_rainet",
         "expect": "success",
         "support_expect": True}

case5 = {"params": get_impl_list(-1, -1, -1, -1, 0, 0.7, 20, 20, False, False, False),
         "case_name": "dynamic_one_less",
         "expect": "success",
         "support_expect": True}

case6 = {"params": get_impl_list(-1, -1, -1, -1, 0, 0.7, 10, 10, False, False, False),
         "case_name": "dynamic_one_more_out",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)

