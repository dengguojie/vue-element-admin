# # -*- coding:utf-8 -*-
import warnings

import numpy as np
from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl.base.padding import smath

warnings.filterwarnings("ignore")
ut_case = OpUT("padding", "padding.test_smath_impl")


@add_cust_test_func(ut_case)
def test_abs_with_positive(_):
    a = np.int32(10)
    return smath.abs_(a) == 10


@add_cust_test_func(ut_case)
def test_abs_with_negative(_):
    a = np.float32(-0.5)
    return smath.abs_(a) == np.float32(0.5)


@add_cust_test_func(ut_case)
def test_abs_with_zero(_):
    a = np.float16(0)
    return smath.abs_(a) == 0


@add_cust_test_func(ut_case)
def test_exp(_):
    a = np.int32(0)
    return smath.exp_(a) == 1


@add_cust_test_func(ut_case)
def test_log(_):
    a = np.int32(1)
    return smath.log_(a) == 0


@add_cust_test_func(ut_case)
def test_rec(_):
    a = np.float32(0.5)
    return smath.rec_(a) == 2


@add_cust_test_func(ut_case)
def test_rsqrt(_):
    a = np.float32(100)
    return smath.rsqrt_(a) == np.float32(0.1)


@add_cust_test_func(ut_case)
def test_sqrt(_):
    a = np.float32(100)
    return smath.sqrt_(a) == np.float32(10)


@add_cust_test_func(ut_case)
def test_add(_):
    a = np.int32(100)
    b = np.int32(5)
    return smath.add_(a, b) == np.int32(105)


@add_cust_test_func(ut_case)
def test_sub(_):
    a = np.int32(100)
    b = np.int32(5)
    return smath.sub_(a, b) == np.int32(95)


@add_cust_test_func(ut_case)
def test_mul(_):
    a = np.int32(100)
    b = np.int32(5)
    return smath.mul_(a, b) == np.int32(500)


@add_cust_test_func(ut_case)
def test_div(_):
    a = np.int32(100)
    b = np.int32(5)
    return smath.div_(a, b) == np.int32(20)


@add_cust_test_func(ut_case)
def test_max(_):
    a = np.int32(100)
    b = np.int32(5)
    return smath.max_(a, b) == np.int32(100)


@add_cust_test_func(ut_case)
def test_min(_):
    a = np.int32(100)
    b = np.int32(5)
    return smath.min_(a, b) == np.int32(5)


@add_cust_test_func(ut_case)
def test_bitwise_and(_):
    a = np.int32(0b100)
    b = np.int32(0b101)
    return smath.bitwise_and_(a, b) == np.int32(0b100)


@add_cust_test_func(ut_case)
def test_bitwise_or(_):
    a = np.int32(0b100)
    b = np.int32(0b101)
    return smath.bitwise_or_(a, b) == np.int32(0b101)


@add_cust_test_func(ut_case)
def test_bitwise_not(_):
    a = np.int8(0b100)
    return smath.bitwise_not_(a) == np.int8(0b11111011)


@add_cust_test_func(ut_case)
def test_relu_with_positive(_):
    a = np.float32(100.5)
    return smath.relu_(a) == a


@add_cust_test_func(ut_case)
def test_relu_with_negative(_):
    a = np.float32(-100.5)
    return smath.relu_(a) == 0


@add_cust_test_func(ut_case)
def test_relu_with_zero(_):
    a = np.float32(0)
    b = np.float32(0)
    return smath.relu_(a) == 0


@add_cust_test_func(ut_case)
def test_lrelu_with_positive(_):
    a = np.float32(100.5)
    b = np.float32(0.01)
    return smath.lrelu_(a, b) == a


@add_cust_test_func(ut_case)
def test_lrelu_with_negative(_):
    a = np.float32(-100.5)
    b = np.float32(0.01)
    return smath.lrelu_(a, b) == np.float32(-1.005)


@add_cust_test_func(ut_case)
def test_lrelu_with_zero(_):
    a = np.float32(0)
    b = np.float32(0.01)
    return smath.lrelu_(a, b) == 0


@add_cust_test_func(ut_case)
def test_cast(_):
    a = np.float32(100.365)
    return smath.cast_(a, "int32") == np.int32(100)


@add_cust_test_func(ut_case)
def test_ceil(_):
    a = np.float32(100.365)
    return smath.ceil_(a, "float32") == np.float32(101)


@add_cust_test_func(ut_case)
def test_floor(_):
    a = np.float32(100.765)
    return smath.floor_(a, "int32") == np.int32(100)


@add_cust_test_func(ut_case)
def test_trunc(_):
    a = np.float32(100.765)
    return smath.trunc_(a, "int32") == np.int32(100)


@add_cust_test_func(ut_case)
def test_round(_):
    a = np.float32(100.465)
    return smath.round_(a, "int32") == np.float32(100)


@add_cust_test_func(ut_case)
def test_round_d(_):
    a = np.float32(101.565)
    return smath.round_d_(a, "int32") == np.float32(102)


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
