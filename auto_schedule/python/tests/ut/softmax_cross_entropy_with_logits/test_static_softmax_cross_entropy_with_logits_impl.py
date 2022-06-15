#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sch_test_frame.ut import OpUT

ut_case = OpUT("SoftmaxCrossEntropyWithLogits", "impl.softmax_cross_entropy_with_logits",
               "softmax_cross_entropy_with_logits")

case1 = {"params": [{"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (2, ), "dtype": "float16", "format": "NCHW", "ori_shape": (2, ),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (2, ), "dtype": "float16", "format": "NCHW", "ori_shape": (2, ),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (5, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2, ), "dtype": "float32", "format": "NCHW", "ori_shape": (2, ),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"},
                    {"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"},
                    {"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"},
                    {"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (221, 8), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 8),"ori_format": "NHWC"},
                    {"shape": (8, ), "dtype": "float32", "format": "NHWC", "ori_shape": (8, ),"ori_format": "NHWC"},
                    {"shape": (221, ), "dtype": "float32", "format": "NHWC", "ori_shape": (221, ),"ori_format": "NHWC"},
                    {"shape": (221, 8), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 8),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (4771178, 139), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178, 139),"ori_format": "NHWC"},
                    {"shape": (4771178, 139), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178, 139),"ori_format": "NHWC"},
                    {"shape": (4771178,), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178,),"ori_format": "NHWC"},
                    {"shape": (4771178, 139), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178, 139),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (16022, 16213), "dtype": "float32", "format": "NHWC", "ori_shape": (16022, 16213),"ori_format": "NHWC"},
                    {"shape": (16022, 16213), "dtype": "float32", "format": "NHWC", "ori_shape": (16022, 16213),"ori_format": "NHWC"},
                    {"shape": (16022,), "dtype": "float32", "format": "NHWC", "ori_shape": (16022,),"ori_format": "NHWC"},
                    {"shape": (16022, 16213), "dtype": "float32", "format": "NHWC", "ori_shape": (16022, 16213),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (99, 2, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 2, 2, 32),"ori_format": "NHWC"},
                     {"shape": (99, 2, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 2, 2, 32),"ori_format": "NHWC"},
                     {"shape": (99, 1, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 1, 2, 32),"ori_format": "NHWC"},
                     {"shape": (99, 2, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 2, 2, 32),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_10",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case11 = {"params": [{"shape": (64, 32, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 32, 1, 4),"ori_format": "NHWC"},
                     {"shape": (64, 32, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 32, 1, 4),"ori_format": "NHWC"},
                     {"shape": (64, 1, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 1, 1, 4),"ori_format": "NHWC"},
                     {"shape": (64, 32, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 32, 1, 4),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_11",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case12 = {"params": [{"shape": (72, 9, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 9, 104, 51),"ori_format": "NHWC"},
                     {"shape": (72, 9, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 9, 104, 51),"ori_format": "NHWC"},
                     {"shape": (72, 1, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 1, 104, 51),"ori_format": "NHWC"},
                     {"shape": (72, 9, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 9, 104, 51),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_12",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case13 = {"params": [{"shape": (47521, 19978), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 19978),"ori_format": "NHWC"},
                     {"shape": (47521, 19978), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 19978),"ori_format": "NHWC"},
                     {"shape": (47521, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 1),"ori_format": "NHWC"},
                     {"shape": (47521, 19978), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 19978),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_13",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case14 = {"params": [{"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"},
                     {"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"},
                     {"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"},
                     {"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_14",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case15 = {"params": [{"shape": (24, 30000), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 30000),"ori_format": "NHWC"},
                     {"shape": (24, 30000), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 30000),"ori_format": "NHWC"},
                     {"shape": (24, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 1),"ori_format": "NHWC"},
                     {"shape": (24, 30000), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 30000),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_15",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case16 = {"params": [{"shape": (14, 11760), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"},
                     {"shape": (14, 11760), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"},
                     {"shape": (14, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 1),"ori_format": "NHWC"},
                     {"shape": (14, 11760), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_16",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case17 = {"params": [{"shape": (2125, 10881), "dtype": "float32", "format": "NHWC", "ori_shape": (2125, 10881),"ori_format": "NHWC"},
                     {"shape": (2125, 10881), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"},
                     {"shape": (2125,), "dtype": "float32", "format": "NHWC", "ori_shape": (2125,),"ori_format": "NHWC"},
                     {"shape": (2125, 10881), "dtype": "float32", "format": "NHWC", "ori_shape": (2125, 10881),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_17",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case18 = {"params": [{"shape": (17660, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (17660, 8),"ori_format": "NHWC"},
                     {"shape": (17660, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (17660, 8),"ori_format": "NHWC"},
                     {"shape": (17660,), "dtype": "float16", "format": "NHWC", "ori_shape": (17660,),"ori_format": "NHWC"},
                     {"shape": (17660, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (17660, 8),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_18",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case19 = {"params": [{"shape": (32, 768), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 768),"ori_format": "NHWC"},
                     {"shape": (32, 768), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 768),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float16", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 768), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 768),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_19",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case20 = {"params": [{"shape": (32, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"},
                     {"shape": (32, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float16", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 10), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_20",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case21 = {"params": [{"shape": (32, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"},
                     {"shape": (32, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float32", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_21",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case22 = {"params": [{"shape": (6272, 17910), "dtype": "float32", "format": "NHWC", "ori_shape": (6272, 17910),"ori_format": "NHWC"},
                     {"shape": (17910,), "dtype": "float32", "format": "NHWC", "ori_shape": (17910,),"ori_format": "NHWC"},
                     {"shape": (6272,), "dtype": "float32", "format": "NHWC", "ori_shape": (6272,),"ori_format": "NHWC"},
                     {"shape": (6272, 17910), "dtype": "float32", "format": "NHWC", "ori_shape": (6272, 17910),"ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_22",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case23 = {"params": [{"shape": (6272, 27910), "dtype": "float16", "format": "NHWC", "ori_shape": (6272, 27910),"ori_format": "NHWC"},
                     {"shape": (27910,), "dtype": "float16", "format": "NHWC", "ori_shape": (27910,),"ori_format": "NHWC"},
                     {"shape": (6272,), "dtype": "float16", "format": "NHWC", "ori_shape": (6272,),"ori_format": "NHWC"},
                     {"shape": (6272, 27910), "dtype": "float16", "format": "NHWC", "ori_shape": (6272, 27910),"ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_23",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case24 = {"params": [{"shape": (6272, 27910), "dtype": "float32", "format": "NHWC", "ori_shape": (6272, 27910),"ori_format": "NHWC"},
                     {"shape": (27910,), "dtype": "float32", "format": "NHWC", "ori_shape": (27910,),"ori_format": "NHWC"},
                     {"shape": (6272,), "dtype": "float32", "format": "NHWC", "ori_shape": (6272,),"ori_format": "NHWC"},
                     {"shape": (6272, 27910), "dtype": "float32", "format": "NHWC", "ori_shape": (6272, 27910),"ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_24",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case25 = {"params": [{"shape": (4096, 17191), "dtype": "float16", "format": "NHWC", "ori_shape": (4096, 17191), "ori_format": "NHWC"},
                     {"shape": (4096, 17191), "dtype": "float16", "format": "NHWC", "ori_shape": (4096, 17191), "ori_format": "NHWC"},
                     {"shape": (4096,), "dtype": "float16", "format": "NHWC", "ori_shape": (4096,), "ori_format": "NHWC"},
                     {"shape": (4096, 17191), "dtype": "float16", "format": "NHWC", "ori_shape": (4096, 17191), "ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_25",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

compile_case_list_common = [
    case1,
    case2,
    case3,
    case4,
    case5,
    case6,
    case7,
    case8,
    case9,
    case10,
    case11,
    case12,
    case13,
    case14,
    case15,
    case17,
    case18,
    case19,
    case20,
    case22,
    case23,
    case24,
    case25
]

compile_case_list_910A = [
    case16,
    case21
]

for item in compile_case_list_common:
    ut_case.add_case(["Ascend910B2", "Ascend910A"], case=item)
for item in compile_case_list_910A:
    ut_case.add_case(["Ascend910A"], case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
