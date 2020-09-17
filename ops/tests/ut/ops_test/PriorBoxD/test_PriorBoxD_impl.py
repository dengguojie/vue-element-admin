#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("PriorBoxD", None, None)

def prior_box_cce(feature_shape, img_shape, data_h_shape, data_w_shape, box_shape, res_shape, dtype, dformat, min_size, max_size, img_h = 0, img_w = 0, step_h = 0.0, step_w = 0.0, flip = True, clip = False, offset = 0.5, variance = [0.1], case_name = "prior_box"):

    return {"params": [{"shape": feature_shape, "dtype":dtype, "format":dformat,"ori_shape": feature_shape,"ori_format":dformat},
                       {"shape":img_shape, "dtype":dtype, "format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": data_h_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": data_w_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": box_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": box_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       {"shape": res_shape, "dtype":dtype,"format":dformat,"ori_shape": img_shape,"ori_format":dformat},
                       min_size, max_size, img_h, img_w, step_h, step_w, flip, clip, offset, variance],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, False, 0.5, [0.1, 0.1, 0.2, 0.2],"prior_box_1")

case2 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float32", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], "prior_box_2")

case3 = prior_box_cce((2, 3, 5, 5, 16), (2, 3, 300, 300, 16), (5,1,1,1), (5,1,1,1), (6,1,1,1), (1, 2, 5, 5, 6, 4), "float16", "NC1HWC0", [162.0], [213.0], 300, 300, 64.0, 64.0, True, True, 0.5, [0.1, 0.1, 0.2, 0.2], "prior_box_3")


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)