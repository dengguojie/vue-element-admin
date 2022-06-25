# # -*- coding:utf-8 -*-
import warnings

import numpy as np
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.dsl.padding import util
from tbe.dsl.padding.simulators import spvalue
from tbe.dsl.padding.value import PaddingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_scmp_impl")


# relu
@add_cust_test_func(ut_case)
def test_relu_exact(_):
    pvalue0 = util.new_pvalue_x(-4.5, "float16")
    pvalue = spvalue.relu(pvalue0)

    return pvalue.value == 0


@add_cust_test_func(ut_case)
def test_relu_tensor(_):
    pvalue0 = util.new_pvalue_tensor("float32")
    pvalue = spvalue.relu(pvalue0)

    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_relu_any(_):
    pvalue0 = util.new_pvalue_any("float32")
    pvalue = spvalue.relu(pvalue0)

    return pvalue.type == PaddingValueType.ANY


# lrelu
@add_cust_test_func(ut_case)
def test_lrelu_exact(_):
    pvalue0 = util.new_pvalue_x(-100.5, "float16")
    scalar1 = tvm.const(0.01, "float16")
    pvalue = spvalue.lrelu(pvalue0, scalar1)

    return pvalue.value == np.float16(-1.005)


@add_cust_test_func(ut_case)
def test_lrelu_tensor(_):
    pvalue0 = util.new_pvalue_tensor("float32")
    scalar1 = tvm.const(0.01, "float32")
    pvalue = spvalue.lrelu(pvalue0, scalar1)

    return pvalue.type == PaddingValueType.TENSOR


@add_cust_test_func(ut_case)
def test_lrelu_any(_):
    pvalue0 = util.new_pvalue_any("float16")
    scalar1 = tvm.const(0.01, "float16")
    pvalue = spvalue.lrelu(pvalue0, scalar1)

    return pvalue.type == PaddingValueType.ANY


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_cmp_any_any_alone_node":
        #     continue

        try:
            ret = v.test_func(None)
        except Exception:
            import traceback
            print(f"\033[93mException: {k}\033[0m")
            print(traceback.format_exc())
            continue

        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
