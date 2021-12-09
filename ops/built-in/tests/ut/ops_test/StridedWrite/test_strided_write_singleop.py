from __future__ import absolute_import
from op_test_frame.ut import OpUT
from impl.strided_write import strided_write
from te import platform as cceconf

ut_case = OpUT("StridedWrite", "impl.strided_write", "strided_write")


def test_strided_write_singleop(test_arg):
    x = {
        "shape": (1, 1, 16, 16, 16),
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
    strided_write(x, y, axis, stride, kernel_name="strided_write")


print("adding StridedWrite singleop ut testcases")
ut_case.add_cust_test_func(test_func=test_strided_write_singleop)
