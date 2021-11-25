# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

import tbe
import tbe.dsl as tbe_dsl
from tbe import tvm
from tbe.dsl.base import operation

GENERAL_FORWARD = "general.forward"
GENERAL_BACKWARD = "general.backward"

warnings.filterwarnings("ignore")
ut_case = OpUT("transdata", "transdata.test_dynamic_transdata_impl")


def test_dsl_transdata_backward_interface(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        # NC1HWC0 -> NHWC
        with tbe_dsl.compute():
            current_compute = operation.get_context().get_current_compute()
            current_compute.add("_transdata_category", GENERAL_BACKWARD)
            params = tvm.placeholder([64, 128, 3, 1, 16], name='params', dtype="float16")
            dst_shape = [64, 3, 1, 2042]
            axes_map = {0: 0, 2: 1, 3: 2, (1, 4): 3}
            res = tbe_dsl.transdata(params, dst_shape, axes_map, 0)
        return [int(x) for x in list(res.shape)] == dst_shape


def test_dsl_transdata_forward_interface(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        # NHWC -> NC1HWC0
        with tbe_dsl.compute():
            current_compute = operation.get_context().get_current_compute()
            current_compute.add("_transdata_category", GENERAL_FORWARD)
            params = tvm.placeholder([64, 3, 1, 2042], name='params', dtype="float16")
            dst_shape = [64, 128, 3, 1, 16]
            axes_map = {0: 0, 1: 2, 2: 3, 3: (1, 4)}
            res = tbe_dsl.transdata(params, dst_shape, axes_map, 0)
        return [int(x) for x in list(res.shape)] == dst_shape


ut_case.add_cust_test_func(test_func=test_dsl_transdata_backward_interface)
ut_case.add_cust_test_func(test_func=test_dsl_transdata_forward_interface)
