# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
import tbe
from tbe import tvm

warnings.filterwarnings("ignore")
ut_case = OpUT("gather", "gather.test_dynamic_gather_impl")


def test_dsl_gather_interface(_):
    params = tvm.placeholder([999, 888, 777], name='params', dtype="float16")
    indices = tvm.placeholder([999, 555,], name='indices', dtype="int32")
    gather_res = tbe.dsl.gather(params, indices, 1, 1)
    return gather_res.shape[0].value == 999 and gather_res.shape[1].value == 555 and gather_res.shape[2].value == 777

def test_dsl_gather_nd_interface(_):
    params = tvm.placeholder([999, 888, 777], name='params', dtype="float16")
    indices = tvm.placeholder([999, 555, 1], name='indices', dtype="int32")
    gather_res = tbe.dsl.gather_nd(params, indices, 1)
    return gather_res.shape[0].value == 999 and gather_res.shape[1].value == 555 and gather_res.shape[2].value == 777



ut_case.add_cust_test_func(test_func=test_dsl_gather_interface)
ut_case.add_cust_test_func(test_func=test_dsl_gather_nd_interface)