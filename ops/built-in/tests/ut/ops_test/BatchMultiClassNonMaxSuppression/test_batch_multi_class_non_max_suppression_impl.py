#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BatchMultiClassNonMaxSuppression", "impl.batch_multi_class_non_max_suppression", "batch_multi_class_non_max_suppression")


def get_dict(_shape, dtype="float16"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND"}


def get_impl_list(batch_size, num_boxes, num_class, num_class_boxes, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, is_need_clip, is_need_valid):
    if num_class_boxes == 0:
        boxes_shape = [batch_size, 4, num_boxes]
    else:
        boxes_shape = [batch_size, num_class_boxes, 4, num_boxes]
    if num_class == 0:
        score_shape = [batch_size, num_boxes]
    else:
        score_shape = [batch_size, num_class, num_boxes]
    if is_need_clip:
        clip_shape = [batch_size, 4]
    else:
        clip_shape = None
    if is_need_valid:
        num_shape = [batch_size]
    else:
        num_shape = None

    output_boxes_shape = [batch_size, 4, max_total_size]
    output_score_shape = [batch_size, max_total_size]
    output_valid_shape = [batch_size]
    input_list = [get_dict(boxes_shape), get_dict(score_shape), get_dict(clip_shape), get_dict(num_shape)]
    output_list = [get_dict(output_boxes_shape), get_dict(output_score_shape), get_dict(output_score_shape), get_dict(output_valid_shape)]
    par_list = [score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, False]

    return input_list + output_list + par_list


case1 = {"params": get_impl_list(1, 29782, 1, 1, 0.5, 0.5, 100, 100, False, True, False),
         "case_name": "faster_rcnn_case_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": get_impl_list(8, 100, 90, 90, 0.5, 0.5, 100, 100, True, True, True),
         "case_name": "faster_rcnn_case_2",
         "expect": "success",
         "support_expect": True}

case3 = {"params": get_impl_list(4, 51150, 6, 1, 0.5, 0.5, 200, 300, False, True, False),
         "case_name": "ssd_mobile_1",
         "expect": "success",
         "support_expect": True}

case4 = {"params": get_impl_list(2, 190000, 0, 0, 0.5, 0.5, 200, 300, False, False, False),
         "case_name": "rainet",
         "expect": "success",
         "support_expect": True}

case5 = {"params": get_impl_list(2, 1000, 2, 2, 0.5, 0.5, 5, 100, False, False, False),
         "case_name": "one_less",
         "expect": "success",
         "support_expect": True}

case6 = {"params": get_impl_list(2, 1000, 150, 150, 0.5, 0.5, 300, 100, False, False, False),
         "case_name": "one_more_out",
         "expect": "success",
         "support_expect": True}

case7 = {"params": [get_dict([1, 4, 200])] + get_impl_list(1, 200, 150, 150, 0.5, 0.5, 100, 100, False, False, False)[1:],
         "case_name": "check_shape_failed",
         "expect": RuntimeError,
         "support_expect": True}

case8 = {"params": get_impl_list(2, 1000, 150, 150, 0.5, 0.5, 300, 0, False, False, False),
         "case_name": "check_max_total_size_failed",
         "expect": RuntimeError,
         "support_expect": True}

case9 = {"params": get_impl_list(2, 1000, 150, 150, 0.5, 0.5, 100, 100, True, False, False),
         "case_name": "check_change_coordinate_frame_failed",
         "expect": RuntimeError,
         "support_expect": True}

case10 = {"params": get_impl_list(2, 20000, 150, 150, 0.5, 0.5, 100, 100, True, True, True),
         "case_name": "check_valied_num_failed",
         "expect": RuntimeError,
         "support_expect": True}

ut_case.add_case(["Ascend310"], case1)
ut_case.add_case(["Ascend310"], case2)
ut_case.add_case(["Ascend310"], case3)
ut_case.add_case(["Ascend310"], case4)
ut_case.add_case(["Ascend310"], case5)
ut_case.add_case(["Ascend310"], case6)
ut_case.add_case(["Ascend310"], case7)
ut_case.add_case(["Ascend310"], case8)
ut_case.add_case(["Ascend310"], case9)
ut_case.add_case(["Ascend310"], case10)


def test_vector_core(test_arg):
    import te.platform as tepf
    from impl.batch_multi_class_non_max_suppression import batch_multi_class_non_max_suppression
    old_soc_version = tepf.get_soc_spec(tepf.SOC_VERSION)
    old_aicore_type = tepf.get_soc_spec(tepf.AICORE_TYPE)
    tepf.te_set_version("Ascend710", "VectorCore")
    params = get_impl_list(8, 100, 90, 90, 0.5, 0.5, 100, 100, True, True, True)
    batch_multi_class_non_max_suppression(*params)
    params = get_impl_list(8, 40800, 90, 90, 0.5, 0.5, 100, 100, True, True, False)
    batch_multi_class_non_max_suppression(*params, impl_mode="high_precision")
    tepf.te_set_version(old_soc_version, old_aicore_type)


ut_case.add_cust_test_func(test_func=test_vector_core)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710"])
    exit(0)
