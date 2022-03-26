# -*- coding:utf-8 -*-
import numpy as np
from op_test_frame.ut import OpUT

ut_case = OpUT("BalanceRois")


# 'pylint: disable=unused-argument
def test_710(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.balance_rois import balance_rois

    te_set_version("Ascend710")

    roi_shape = [1000, 5]
    index_shape = [1000]
    dtype = "float16"

    rois = {
        "dtype": dtype,
        "ori_shape": roi_shape,
        "shape": roi_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "input",
    }
    re_rois = {
        "dtype": dtype,
        "ori_shape": roi_shape,
        "shape": roi_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "output",
    }
    index = {
        "dtype": "int32",
        "ori_shape": index_shape,
        "shape": index_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "input",
    }
    params = [rois, re_rois, index, "balance_roi"]
    balance_rois(*params)
    te_set_version("Ascend310")


ut_case.add_cust_test_func("Ascend310", test_710)
ut_case.run("Ascend310")
