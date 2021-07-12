# # -*- coding:utf-8 -*-
"""
test_add_impl
"""
# pylint: disable=unused-import
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.utils.op_param_util import cartesian_set_format_dtype

ut_case = BroadcastOpUT("Add")

# test shape supported, format don't care, actual is ND
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (16, 2, 4, 32),
                                  (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (4096, 1024),
                                  (4096, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (32, 128, 1024),
                                  (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (1, 16, 512, 512),
                                  (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (1024, 1024, 256),
                                  (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (10, 12), (10, 11),
                                  expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Ascend910A", "Ascend310"], ["float16", "float32", "int32"], (10, 13), (10, 11, 12),
                                  expect=RuntimeError)

ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (1,), (1,))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (1, 1), (1, 1))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (16, 32), (16, 32))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (16, 2, 32), (16, 2, 32))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (16, 2, 4, 32), (16, 2, 4, 32))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (512, 1024), (512, 1024))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (2, 1024), (2, 1024))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (4096, 1024), (4096, 1024))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (32, 128, 1024), (32, 128, 1024))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (100, 100), (100, 100))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (1, 512, 1), (1,))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (1, 16, 512, 512), (1, 1, 512, 512))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (9973, 1), (9973, 1))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (1024, 1024, 256), (1024, 1024, 256))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (11, 33), (11, 33))
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (10, 12), (10, 11), expect=RuntimeError)
ut_case.add_broadcast_case_simple(["Hi3796CV300ES"], ["float16", "int32"], (10, 13), (10, 11, 12), expect=RuntimeError)

# test format is different, one input is FRACTAL_NZ, another is NCHW, NHWC, ND
# FRACTAL_NZ, ND
ut_case.add_broadcast_case("all", ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 15), "ND"],
                           ["float16", (512, 1, 1), "ND"], expect=RuntimeError, case_name="nz_nd_1")

ut_case.add_broadcast_case("all", ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"],
                           ["float16", (512, 32, 32), "ND"], expect=RuntimeError, case_name="nz_nd_2")

ut_case.add_broadcast_case("all", ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"],
                           ["float16", (512, 1, 1), "ND"], case_name="nz_nd_3")

ut_case.add_broadcast_case("all", ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"],
                           ["float16", (512, 32, 1), "ND"], case_name="nz_nd_4")

ut_case.add_broadcast_case("all", ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"],
                           ["float16", (512, 1, 32), "ND"])
# ND, FRACTAL_NZ
ut_case.add_broadcast_case("all", ["float16", (512, 1, 32), "ND"],
                           ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 31, 32), "ND"],
                           expect=RuntimeError)

ut_case.add_broadcast_case("all", ["float16", (512, 32, 32), "ND"],
                           ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"],
                           expect=RuntimeError)

ut_case.add_broadcast_case("all", ["float16", (512, 32, 1), "ND"],
                           ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"])

ut_case.add_broadcast_case("all", ["float16", (512, 1, 32), "ND"],
                           ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"])

ut_case.add_broadcast_case("all", ["float16", (512, 1, 1), "ND"],
                           ["float16", (512, 2, 2, 16, 16), "FRACTAL_NZ", (512, 32, 32), "ND"])

# NHWC, FRACTAL_NZ
ut_case.add_broadcast_case("all", ["float16", (32, 64, 1, 64), "NHWC"],
                           ["float16", (32, 64, 4, 4, 16, 16), "FRACTAL_NZ", (32, 64, 64, 64), "ND"])

ut_case.add_broadcast_case("all", ["float16", (32, 64, 4, 4, 16, 16), "FRACTAL_NZ", (32, 64, 64, 64), "ND"],
                           ["float16", (32, 64, 1, 64), "NHWC"])

ut_case.add_broadcast_case("all", ["float16", (32, 64, 1, 64), "NCHW"],
                           ["float16", (32, 64, 4, 4, 16, 16), "FRACTAL_NZ", (32, 64, 64, 64), "ND"])

ut_case.add_broadcast_case("all", ["float16", (32, 64, 4, 4, 16, 16), "FRACTAL_NZ", (32, 64, 64, 64), "ND"],
                           ["float16", (32, 64, 1, 64), "NCHW"])
ut_case.add_broadcast_case("Ascend910A", ["float16", (512,), "HWCN", (32, 130,16,512), "FRACTAL_NZ"],
                           ["float16", (32, 130,16,512), "FRACTAL_NZ"])

# pylint: disable=unused-argument
# ut_case.add_test_cfg_cov_case("all")
def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.add import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     "test_add_op_select_format_1")
    op_select_format({"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     "test_add_op_select_format_2")
    op_select_format({"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     "test_add_op_select_format_3")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     "test_add_op_select_format_4")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     "test_add_op_select_format_5")
    op_select_format({"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     "test_add_op_select_format_6")
    op_select_format({"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     "test_add_op_select_format_7")
    op_select_format({"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     "test_add_op_select_format_8")
    op_select_format({"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_add_op_select_format_9")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_add_op_select_format_10")
    op_select_format({"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_add_op_select_format_11")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_add_op_select_format_12")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NHWC"},
                     {"shape": (1,), "dtype": "int8", "format": "NHWC", "ori_shape": (1,),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 16, 1), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NHWC"},
                     "test_add_op_select_format_13")
    op_select_format({"shape": (4,64,200,320), "dtype": "float16", "format": "NCHW", "ori_shape": (4,64,200,320),
                      "ori_format": "NCHW"},
                     {"shape": (1,64,1,1), "dtype": "float16", "format": "NCHW", "ori_shape": (1,64,1,1),
                      "ori_format": "NCHW"},
                     {"shape": (4,64,200,320), "dtype": "float16", "format": "NCHW", "ori_shape": (4,64,200,320),
                      "ori_format": "NCHW"},
                     "test_add_op_select_format_14")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     "test_add_op_select_format_15")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     "test_add_op_select_format_16")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 8},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 8},
                     "test_add_op_select_format_17")
ut_case.add_cust_test_func(test_func=test_op_select_format)
