# # -*- coding:utf-8 -*-
import warnings

from tbe import tvm
from tbe.dsl.base import d_format_util
from tbe.dsl.base import var_api

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("d_format_util", "d_format_util.test_d_format_util_impl")

# get_format
@add_cust_test_func(ut_case)
def test_get_format(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.get_format((n, c1, hw, c0))

    return ret == ["N", "C1", "H", "W", "C0"]


@add_cust_test_func(ut_case)
def test_get_format_ignore_none(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.get_format((1, n, c1, hw, c0), ignore_none=True)

    return ret == ["N", "C1", "H", "W", "C0"]


@add_cust_test_func(ut_case)
def test_get_format_with_none(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.get_format((var_api.const(1), n, c1, hw, c0), ignore_none=False)

    return ret == [None, "N", "C1", "H", "W", "C0"]


@add_cust_test_func(ut_case)
def test_get_format_default(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.get_format((var_api.const(1), n, c1, hw, c0))

    return ret == ["N", "C1", "H", "W", "C0"]


# is_5hd_format
@add_cust_test_func(ut_case)
def test_is_5hd_format_ignore_none_true(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.is_5hd_format((1, n, c1, hw, c0), ignore_none=True)

    return ret is True


@add_cust_test_func(ut_case)
def test_is_5hd_format_ignore_none_false(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.is_5hd_format((1, n, c1, hw, c0), ignore_none=True)

    return ret is False


@add_cust_test_func(ut_case)
def test_is_5hd_format_with_none_true(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.is_5hd_format((n, c1, hw, c0), ignore_none=False)

    return ret is True


@add_cust_test_func(ut_case)
def test_is_5hd_format_with_none_false(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.is_5hd_format((1, n, c1, hw, c0), ignore_none=False)

    return ret is False


@add_cust_test_func(ut_case)
def test_is_5hd_format_default_true(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "W"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.is_5hd_format((1, n, c1, hw, c0))

    return ret is True


@add_cust_test_func(ut_case)
def test_is_5hd_format_default_false(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})

    ret = d_format_util.is_5hd_format((1, n, c1, hw, c0))

    return ret is False


# get_axis_type
@add_cust_test_func(ut_case)
def test_set_axis_type(_):
    n = var_api.var("n")
    d_format_util.set_axis_type(n, "N")
    ret = d_format_util.get_axis_type(n)

    return ret == "N"


@add_cust_test_func(ut_case)
def test_get_axis_type(_):
    n = var_api.var("n")
    d_format_util.set_axis_type(n, "N")
    ret = d_format_util.get_axis_type(n)

    return ret == "N"


# get_original
@add_cust_test_func(ut_case)
def test_set_original(_):
    c = var_api.var("C")
    c1 = var_api.var("C1")
    d_format_util.set_original(c1, c)
    ret = d_format_util.get_original(c1)

    return str(ret) == "C"


@add_cust_test_func(ut_case)
def test_get_original(_):
    c = var_api.var("C")
    c1 = var_api.var("C1")
    d_format_util.set_original(c1, c)
    ret = d_format_util.get_original(c1)

    return str(ret) == "C"


# eq_axis_type
@add_cust_test_func(ut_case)
def test_eq_axis_type_none_true(_):
    axis_type1, axis_type2 = None, None
    return d_format_util.eq_axis_type(axis_type1, axis_type2) is True


@add_cust_test_func(ut_case)
def test_eq_axis_type_none_false(_):
    axis_type1, axis_type2 = None, "C"
    return d_format_util.eq_axis_type(axis_type1, axis_type2) is False


@add_cust_test_func(ut_case)
def test_eq_axis_type_false(_):
    axis_type1, axis_type2 = "C1", "C"
    return d_format_util.eq_axis_type(axis_type1, axis_type2) is False


@add_cust_test_func(ut_case)
def test_eq_axis_type_str_list(_):
    axis_type1, axis_type2 = ["C"], "C"
    return d_format_util.eq_axis_type(axis_type1, axis_type2) is True


@add_cust_test_func(ut_case)
def test_eq_axis_type_tuple_list(_):
    axis_type1, axis_type2 = ("C",), ["C"]
    return d_format_util.eq_axis_type(axis_type1, axis_type2) is True


@add_cust_test_func(ut_case)
def test_eq_axis_type_str_list_false(_):
    axis_type1, axis_type2 = "C1", ["C"]
    return d_format_util.eq_axis_type(axis_type1, axis_type2) is False


# in_axis_type
@add_cust_test_func(ut_case)
def test_in_axis_type_true(_):
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    return d_format_util.in_axis_type(c1, ["C1", "C0"])


@add_cust_test_func(ut_case)
def test_in_axis_type_false(_):
    c1 = var_api.var("c1", annotation={"axis_type": "H"})
    return d_format_util.in_axis_type(c1, ["C1", "C0"]) is False


# get_axis
@add_cust_test_func(ut_case)
def test_get_axis_exist(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})
    shape = (n, c1, hw, c0)

    return d_format_util.get_axis(shape, "C1") == c1


@add_cust_test_func(ut_case)
def test_get_axis_none(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})
    shape = (n, c1, hw, c0)

    return d_format_util.get_axis(shape, "C") is None


# get_c0
@add_cust_test_func(ut_case)
def test_get_c0(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})
    shape = (n, c1, hw, c0)

    return d_format_util.get_c0(shape) == c0


# get_c1
@add_cust_test_func(ut_case)
def test_get_c1(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})
    shape = (n, c1, hw, c0)

    return d_format_util.get_c1(shape) == c1


# get_c
@add_cust_test_func(ut_case)
def test_get_c(_):
    n = var_api.var("n", annotation={"axis_type": "N"})
    c1 = var_api.var("c1", annotation={"axis_type": ["C1"]})
    hw = var_api.var("hw", annotation={"axis_type": ["H", "A"]})
    c0 = var_api.const(16, annotation={"axis_type": "C0"})
    c = var_api.var("c", annotation={"axis_type": ["C"]})
    d_format_util.set_original(c1, c)
    d_format_util.set_original(c0, c)
    shape = (n, c1, hw, c0)

    return d_format_util.get_c(shape) == c


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_tensor_brc_adjust_calc_with_pad_and_exist_pad_brc_and_src_not_exist_pad":
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
