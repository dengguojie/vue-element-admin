#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("PriorBoxDV2", "impl.prior_box_d_v2", "prior_box_d_v2")

def gen_prior_data_case(feature_shape, img_shape, boxes_shape, y_shape, dtype_val, dformat, min_size,
                        max_size, img_h, img_w, step_h, step_w, flip, clip, offset, variance, case_name_val, expect):

    return {"params": [{"shape": feature_shape, "dtype":dtype_val, "format":dformat,"ori_shape": feature_shape,"ori_format":dformat},
                       {"shape":img_shape, "dtype":dtype_val, "format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": boxes_shape, "dtype":dtype_val,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": y_shape, "dtype":dtype_val,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                        min_size, max_size, img_h, img_w, step_h, step_w, flip, clip, offset, variance],
                       "case_name": case_name_val,
                       "expect": expect,
                       "format_expect": [],
                       "support_expect": True}

ut_case.add_case(["Hi3796CV300ES"],
                 gen_prior_data_case((2, 16, 5, 5), (2,16,300,300), (1, 2, 160,1), (1, 2, 160,1),
                                     "float32", "NCHW", [30.0], [60.0], 300, 300, 8.0, 8.0, True,
                                     False, 0.5, [0.1, 0.1, 0.2, 0.2], "prior_fp16_1", RuntimeError))

ut_case.add_case(["Ascend910", "Hi3796CV300ES"],
                 gen_prior_data_case((2, 16, 5, 5), (2,16,300,300), (1, 2, 160,1), (1, 2, 160, 1),
                                     "float16", "NCHW", [30.0], [60.0], 300, 300, 8.0, 8.0, True,
                                     False, 0.5, [0.1, 0.1, 0.2, 0.2], "prior_fp16_3", "success"))


ut_case.run(["Ascend910", "Ascend310", "Ascend710", "Hi3796CV300ES"])
