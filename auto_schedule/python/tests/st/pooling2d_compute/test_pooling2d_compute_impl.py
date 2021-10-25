# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")

ut_case = OpUT("pooling2d_compute", "pooling2d_compute.test_pooling2d_compute_impl")


def test_get_caffe_out_size_and_pad(_):
    """
    @return: Ture
    """
    ceil_mode, in_size_h, in_size_w, window_h, window_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, \
        pad_bottom, pad_left, pad_right = 0, 10, 10, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
    tbe.get_caffe_out_size_and_pad(ceil_mode, in_size_h, in_size_w, window_h, window_w,
                                   stride_h, stride_w, dilation_h, dilation_w, pad_top,
                                   pad_bottom, pad_left, pad_right)
    return True


ut_case.add_cust_test_func(test_func=test_get_caffe_out_size_and_pad)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
