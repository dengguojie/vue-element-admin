from __future__ import absolute_import
from op_test_frame.ut import OpUT
from impl.strided_read import strided_read
from te import platform as cceconf

ut_case = OpUT("StridedRead", "impl.strided_read", "strided_read")


def test_strided_read_singleop(test_arg):
    x = {
        "shape": (1, 2, 16, 16, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
    }
    y = {
        "shape": (1, 1, 16, 16, 16),
        "dtype": "float16",
        "format": "NC1HWC0",
    }
    axis = 1
    stride = 2
    cceconf.te_set_version("Ascend310")
    strided_read(x, y, axis, stride, kernel_name="strided_read")


print("adding StridedRead singleop ut testcases")
ut_case.add_cust_test_func(test_func=test_strided_read_singleop)
