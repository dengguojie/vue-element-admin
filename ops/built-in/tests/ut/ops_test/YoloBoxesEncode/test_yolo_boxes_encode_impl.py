# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("yolo_boxes_encode")


def gen_yolo_boxes_encode_case(bboxes, gt_bboxes, stride, encoded_bboxes, performance_mode, kernel_name_val,
                               expect, calc_expect_func=None):
    if calc_expect_func:
        return {"params": [bboxes, gt_bboxes, stride, encoded_bboxes, performance_mode],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True,
                "calc_expect_func": calc_expect_func}
    else:
        return {"params": [bboxes, gt_bboxes, stride, encoded_bboxes, performance_mode],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True}


ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (6300, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300,), "dtype": "int32", "format": "ND",
                      "ori_shape": (6300,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", "success"))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (6300, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300,), "dtype": "int32", "format": "ND",
                      "ori_shape": (6300,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", "success"))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (6300, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300,), "dtype": "int32", "format": "ND",
                      "ori_shape": (6300,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6300, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (6300, 4), "ori_format": "ND", "param_type": "output"},
                     "high_performance",
                     "yolo_boxes_encode", "success"))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (16, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (16, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (16,), "dtype": "int32", "format": "ND",
                      "ori_shape": (16,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (16, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (16, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", "success"))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (200, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200,), "dtype": "int32", "format": "ND",
                      "ori_shape": (200,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", "success"))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (5120 * 4, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (5120 * 4, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (5120 * 4, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (5120 * 4, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (5120 * 4,), "dtype": "int32", "format": "ND",
                      "ori_shape": (5120 * 4,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (5120 * 4, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (5120 * 4, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", "success"))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (5121 * 4, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (5121 * 4, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (5121 * 4, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (5121 * 4, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (5121 * 4,), "dtype": "int32", "format": "ND",
                      "ori_shape": (5121 * 4,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (5121 * 4, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (5121 * 4, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", RuntimeError))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (200, 4), "dtype": "int32", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200, 4), "dtype": "int32", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200,), "dtype": "int32", "format": "ND",
                      "ori_shape": (200,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200, 4), "dtype": "int32", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", RuntimeError))

ut_case.add_case("Ascend910A",
                 gen_yolo_boxes_encode_case(
                     {"shape": (200, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (100, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200,), "dtype": "int32", "format": "ND",
                      "ori_shape": (200,), "ori_format": "ND", "param_type": "input"},
                     {"shape": (200, 4), "dtype": "float16", "format": "ND",
                      "ori_shape": (200, 4), "ori_format": "ND", "param_type": "output"},
                     "high_precision",
                     "yolo_boxes_encode", RuntimeError))

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
