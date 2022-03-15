#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import te.platform as tepf
from te import platform as cce_conf
from tbe.common.context import op_context
from tbe.common.context.op_context import OpContext
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.batch_multi_class_non_max_suppression import batch_multi_class_non_max_suppression


def get_dict(_shape, dtype="float16"):
    if _shape is None:
        return None
    return {"shape": _shape, "dtype": dtype, "format": "ND", "ori_shape": _shape, "ori_format": "ND"}


def get_impl_list(batch_size, num_boxes, num_class, num_class_boxes, score_threshold, iou_threshold, max_size_per_class,
                  max_total_size, change_coordinate_frame, is_need_clip, is_need_valid):
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
    output_list = [get_dict(output_boxes_shape), get_dict(output_score_shape), get_dict(output_score_shape),
                   get_dict(output_valid_shape)]
    par_list = [score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, False]

    return input_list + output_list + par_list


def test_vector_core():
    old_soc_version = tepf.get_soc_spec(tepf.SOC_VERSION)
    old_aicore_type = tepf.get_soc_spec(tepf.AICORE_TYPE)
    tepf.te_set_version("Ascend710", "VectorCore")
    params = get_impl_list(8, 100, 90, 90, 0.5, 0.5, 100, 100, True, True, True)
    batch_multi_class_non_max_suppression(*params)
    params = get_impl_list(8, 40800, 90, 90, 0.5, 0.5, 100, 100, True, True, False)
    batch_multi_class_non_max_suppression(*params, impl_mode="high_precision")
    tepf.te_set_version(old_soc_version, old_aicore_type)


soc_version = cce_conf.get_soc_spec("SOC_VERSION")
cce_conf.te_set_version("Hi3796CV300CS")
test_vector_core()
cce_conf.te_set_version(soc_version)


def get_impl_list_920(batch_size, num_boxes, num_class, num_class_boxes,
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

    input_list = [get_dict(boxes_shape, boxes_range),
                  get_dict(score_shape, score_range),
                  get_dict(clip_shape, clip_range),
                  get_dict(num_shape, num_range)]
    par_list = [score_threshold, iou_threshold, max_size_per_class,
                max_total_size, change_coordinate_frame, False]

    return input_list + par_list


def side_effects(*args):
    # return vals[args]
    return True

def test_op_mask_1981_test_2():
    from impl.dynamic.batch_multi_class_non_max_suppression_new import BMCNMS
    with op_context.OpContext("dynamic"):
        with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
            lis = get_impl_list_920(-1, -1, -1, -1, 0, 0.7, 100, 100, False, False, False)
            obj = BMCNMS(*lis, "test_op_mask_1981_test_2")
            obj.bmcnms_compute()


soc_version = cce_conf.get_soc_spec("SOC_VERSION")
cce_conf.te_set_version("Ascend920A")
test_op_mask_1981_test_2()
cce_conf.te_set_version(soc_version)
