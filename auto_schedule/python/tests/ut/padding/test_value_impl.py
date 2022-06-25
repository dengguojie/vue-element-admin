# # -*- coding:utf-8 -*-
import warnings

import numpy as np
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.dsl.padding.value import PaddingValue
from tbe.dsl.padding.value import PaddingValueType
from tbe.dsl.padding.value import SettingValue
from tbe.dsl.padding.value import SettingValueType

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_value_impl")


@add_cust_test_func(ut_case)
def test_padding_value(_):
    pv = PaddingValue(PaddingValueType.EXACT, "int32", np.int32(10))

    asert_type = pv.type == PaddingValueType.EXACT
    assert_value = pv.value == 10

    return all((asert_type, assert_value))


@add_cust_test_func(ut_case)
def test_setting_value(_):
    sv = SettingValue(SettingValueType.NORMAL, "float32")
    condition = lambda *i: i[0] > 10
    sv.condition = condition
    value = tvm.const(100)
    sv.value = value
    ph_1 = tvm.placeholder((10,), "float32")
    sv.target = ph_1

    asert_type = sv.type == SettingValueType.NORMAL
    asert_condition = sv.condition == condition
    asert_value = sv.value == value
    asert_target = sv.target == ph_1

    return all((asert_type, asert_condition, asert_value, asert_target))


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
