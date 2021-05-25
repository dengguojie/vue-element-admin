# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np

ut_case = OpUT("GridAssignPositive", None, None)


def get_params(k_num, n_num, pos_iou_thr, min_pos_iou, gt_max_assign_all, data_type):
    data_format = "ND"
    flag_type = "uint8"
    int_type = "int32"
    shape_0 = (n_num,)
    shape_1 = (k_num,)
    shape_2 = (k_num, n_num)
    assigned_gt_inds = {"shape": shape_0, "format": data_format, "dtype": data_type,
                        "ori_shape": shape_0, "ori_format": data_format}
    overlaps = {"shape": shape_2, "format": data_format, "dtype": data_type,
                "ori_shape": shape_2, "ori_format": data_format}
    box_responsible_flags = {"shape": shape_0, "format": data_format, "dtype": flag_type,
                             "ori_shape": shape_0, "ori_format": data_format}
    max_overlaps = {"shape": shape_0, "format": data_format, "dtype": data_type,
                    "ori_shape": shape_0, "ori_format": data_format}
    argmax_overlaps = {"shape": shape_0, "format": data_format, "dtype": int_type,
                       "ori_shape": shape_0, "ori_format": data_format}
    gt_max_overlaps = {"shape": shape_1, "format": data_format, "dtype": data_type,
                       "ori_shape": shape_1, "ori_format": data_format}
    gt_argmax_overlaps = {"shape": shape_1, "format": data_format, "dtype": int_type,
                          "ori_shape": shape_1, "ori_format": data_format}
    num_gts = {"shape": (1, ), "format": data_format, "dtype": int_type,
               "ori_shape": (1, ), "ori_format": data_format}
    assigned_gt_inds_pos = {"shape": shape_0, "format": data_format, "dtype": data_type,
                            "ori_shape": shape_0, "ori_format": data_format}
    params = [assigned_gt_inds, overlaps, box_responsible_flags,
              max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps,
              num_gts, assigned_gt_inds_pos,
              pos_iou_thr, min_pos_iou, gt_max_assign_all]
    return params


case1 = {
    "params": get_params(128, 6300, 0.5, 0.0, True, "float32"),
    "case_name": "mish_grad_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {
    "params": get_params(128, 6300, 0.5, 0.0, False, "float32"),
    "case_name": "mish_grad_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
