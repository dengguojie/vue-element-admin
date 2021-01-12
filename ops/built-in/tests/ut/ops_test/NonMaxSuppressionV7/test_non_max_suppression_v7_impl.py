#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("NonMaxSuppressionV7", "impl.non_max_suppression_v7", "non_max_suppression_v7")


def get_dict(_shape, dtype="float16"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND"}


def get_impl_list(batch_size, class_size, box_size,
                  center_point_box, max_boxes_size):

    boxes_shape = [batch_size, box_size, 4]
    scores_shape = [batch_size, class_size, box_size]

    max_size_per_class_shape = [1]
    iou_theshold_shape = [1]
    score_threshold_shape = [1]

    index_id_shape = [batch_size, class_size, box_size, 3]

    output_shape = [max_boxes_size, 3]

    input_list = [get_dict(boxes_shape), get_dict(scores_shape),
                  get_dict(max_size_per_class_shape, "int32"), get_dict(iou_theshold_shape, "float32"),
                  get_dict(score_threshold_shape, "float32"), get_dict(index_id_shape)]
    output_list = [get_dict(output_shape, "int32")]
    par_list = [center_point_box, max_boxes_size]
    return input_list + output_list + par_list


case1 = {"params": get_impl_list(2, 1, 6, 0, 20),
         "case_name": "nms_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": get_impl_list(2, 1, 6, 1, 10),
         "case_name": "nms_case_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": get_impl_list(2, 2, 720, 0, 100),
         "case_name": "nms_case_3",
         "expect": "success",
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)
