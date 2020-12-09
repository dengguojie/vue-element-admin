#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import json
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
ut_case.add_case(["Ascend910"],
                 gen_target_crop_and_resize_case(
                     (1,3,224,224), (5,4), [5], (5,1,100,120,4), 100, 120,
                     "YUV420SP_U8", "target_crop_and_resize_1", RuntimeError))

if __name__ == '__main__':
    ut_case.run()

