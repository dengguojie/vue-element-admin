# # -*- coding:utf-8 -*-
import warnings

from tbe import tvm
from tbe.dsl.base import var_api

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("var_api", "var_api.test_var_api_impl")


@add_cust_test_func(ut_case)
def test_get_annotation(_):
    annotation = {"k1": "v1", "k2": ["v21", "v22"]}
    var_ = var_api.var("var_0", annotation=annotation)
    ret = var_api.get_annotation(var_)

    return ret == annotation


@add_cust_test_func(ut_case)
def test_set_annotation(_):
    annotation = {"k3": "v3", "k4": ["v41", "v42"]}
    var_ = var_api.var("var_1")
    var_api.set_annotation(var_, annotation=annotation)
    ret = var_api.get_annotation(var_)

    return ret == annotation


@add_cust_test_func(ut_case)
def test_get_attr_keys(_):
    annotation = {"k5": "v5", "k6": ["v61", "v62"]}
    var_ = var_api.var("var_0", annotation=annotation)
    ret = var_api.get_attr_keys(var_)

    return ret == {"k5", "k6"}


@add_cust_test_func(ut_case)
def test_get_attr_keys_with_empty_keys(_):
    var_ = var_api.var("var_1")
    ret = var_api.get_attr_keys(var_)

    return ret == set()


@add_cust_test_func(ut_case)
def test_get_attr(_):
    annotation = {"k1": "v1", "k2": ["v21", "v22"]}
    var_ = var_api.var("var_0", annotation=annotation)
    ret = var_api.get_attr(var_, "k1")

    return ret == "v1"


@add_cust_test_func(ut_case)
def test_set_attr(_):
    var_ = var_api.var("var_0")
    var_api.set_attr(var_, "k1", "v1")
    ret = var_api.get_attr(var_, "k1")

    return ret == "v1"


@add_cust_test_func(ut_case)
def test_const(_):
    annotation = {"k1": "v1", "k2": "v2"}
    var_ = var_api.const(20, annotation=annotation)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.IntImm) and str(var_) == "20"
    annotation_assert = ret_annotation == annotation

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_var(_):
    annotation = {"k1": "v1", "k2": "v2"}
    var_ = var_api.var("var_0", annotation=annotation)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.Var) and str(var_) == "var_0"
    annotation_assert = ret_annotation == annotation

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_div(_):
    annotation = {"k1": "v1", "k2": "v2"}
    var_0 = var_api.var("var_0", annotation=annotation)
    var_ = var_api.div(var_0, 10)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.Div) and str(var_) == "(var_0/10)"
    annotation_assert = ret_annotation == annotation

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_indexdiv(_):
    annotation1 = {"k1": "v1", "k2": "v2"}
    annotation2 = {"k1": "v1", "k2": "v2_1"}
    var_0 = var_api.var("var_0", annotation=annotation1)
    var_1 = var_api.var("var_1", annotation=annotation2)
    var_ = var_api.indexdiv(var_0, var_1)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.FloorDiv) and str(var_) == "floordiv(var_0, var_1)"
    annotation_assert = ret_annotation == {"k1": "v1"}

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_indexmod(_):
    annotation1 = {"k1": "v1", "k2": "v2"}
    annotation2 = {"k1": "v1", "k2": "v2"}
    var_0 = var_api.var("var_0", annotation=annotation1)
    var_1 = var_api.var("var_1", annotation=annotation2)
    var_ = var_api.indexmod(var_0, var_1)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.FloorMod) and str(var_) == "floormod(var_0, var_1)"
    annotation_assert = ret_annotation == {"k1": "v1", "k2": "v2"}

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_truncdiv(_):
    annotation1 = {"k1": "v1", "k2": "v2"}
    annotation2 = {"k1": "v1-1", "k2": "v2-1"}
    var_0 = var_api.var("var_0", annotation=annotation1)
    var_1 = var_api.var("var_1", annotation=annotation2)
    var_ = var_api.truncdiv(var_0, var_1)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.Div) and str(var_) == "(var_0/var_1)"
    annotation_assert = ret_annotation == {}

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_truncmod(_):
    annotation1 = {"k1": "v1", "k2": "v2"}
    var_0 = var_api.var("var_0", annotation=annotation1)
    var_1 = var_api.var("var_1", annotation=annotation1)
    var_ = var_api.truncmod(var_0, var_1)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.Mod) and str(var_) == "(var_0 % var_1)"
    annotation_assert = ret_annotation == annotation1

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_floordiv(_):
    var_0 = var_api.var("var_0")
    var_1 = var_api.var("var_1")
    var_ = var_api.floordiv(var_0, var_1)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.FloorDiv) and str(var_) == "floordiv(var_0, var_1)"
    annotation_assert = ret_annotation == {}

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_floormod(_):
    annotation1 = {"k1": "v1", "k2": "v2"}
    annotation2 = {"k1": "v1", "k2": "v2-1"}
    var_0 = var_api.var("var_0", annotation=annotation1)
    var_1 = var_api.var("var_1", annotation=annotation2)
    var_ = var_api.floormod(var_0, var_1)
    ret_annotation = var_api.get_annotation(var_)

    var_assert = isinstance(var_, tvm.expr.FloorMod) and str(var_) == "floormod(var_0, var_1)"
    annotation_assert = ret_annotation == {"k1": "v1"}

    return all((var_assert, annotation_assert))


@add_cust_test_func(ut_case)
def test_sum(_):
    ori_0 = var_api.var("ori_0", annotation={"axis_type": "C"})
    ori_1 = var_api.var("ori_1", annotation={"axis_type": "C"})
    ori_2 = var_api.var("ori_2", annotation={"axis_type": "C"})

    var_0 = var_api.var("var_0", annotation={"axis_type": "C1", "original": ori_0})
    var_1 = var_api.var("var_1", annotation={"axis_type": "C1", "original": ori_1})
    var_2 = var_api.var("var_2", annotation={"axis_type": "C1", "original": ori_2})

    var_ = var_api.sum(var_0, var_1, var_2)
    ori_var = var_api.get_attr(var_, "original")

    var_annotation = var_api.get_annotation(var_)
    ori_annotation = var_api.get_annotation(ori_var)

    var_assert = isinstance(var_, tvm.expr.Add) and str(var_) == "((var_0 + var_1) + var_2)"
    ori_var_assert = isinstance(ori_var, tvm.expr.Add) and str(ori_var) == "((ori_0 + ori_1) + ori_2)"

    annotation_assert = var_annotation["axis_type"] == "C1" and \
                        str(var_annotation["original"]) == "((ori_0 + ori_1) + ori_2)"
    ori_annotation_assert = ori_annotation == {"axis_type": "C"}

    return all((var_assert, ori_var_assert, annotation_assert, ori_annotation_assert))


@add_cust_test_func(ut_case)
def test_min(_):
    ori_0 = var_api.var("ori_0", annotation={"axis_type": "C"})
    ori_1 = var_api.var("ori_1", annotation={"axis_type": "C"})
    ori_2 = var_api.var("ori_2", annotation={"axis_type": "C"})

    var_0 = var_api.var("var_0", annotation={"axis_type": "C1", "original": ori_0})
    var_1 = var_api.var("var_1", annotation={"axis_type": "C1", "original": ori_1})
    var_2 = var_api.var("var_2", annotation={"axis_type": "C1", "original": ori_2})

    var_ = var_api.min(var_0, var_1, var_2)
    ori_var = var_api.get_attr(var_, "original")

    var_annotation = var_api.get_annotation(var_)
    ori_annotation = var_api.get_annotation(ori_var)

    var_assert = isinstance(var_, tvm.expr.Min) and str(var_) == "min(min(var_0, var_1), var_2)"
    ori_var_assert = isinstance(ori_var, tvm.expr.Min) and str(ori_var) == "min(min(ori_0, ori_1), ori_2)"

    annotation_assert = var_annotation["axis_type"] == "C1" and \
                        str(var_annotation["original"]) == "min(min(ori_0, ori_1), ori_2)"
    ori_annotation_assert = ori_annotation == {"axis_type": "C"}

    return all((var_assert, ori_var_assert, annotation_assert, ori_annotation_assert))


@add_cust_test_func(ut_case)
def test_max(_):
    ori_0 = var_api.var("ori_0", annotation={"axis_type": "C"})
    ori_1 = var_api.var("ori_1", annotation={"axis_type": "C"})
    ori_2 = var_api.var("ori_2", annotation={"axis_type": "C"})

    var_0 = var_api.var("var_0", annotation={"axis_type": "C1", "original": ori_0})
    var_1 = var_api.var("var_1", annotation={"axis_type": "C1", "original": ori_1})
    var_2 = var_api.var("var_2", annotation={"axis_type": "C1", "original": ori_2})

    var_ = var_api.max(var_0, var_1, var_2)
    ori_var = var_api.get_attr(var_, "original")

    var_annotation = var_api.get_annotation(var_)
    ori_annotation = var_api.get_annotation(ori_var)

    var_assert = isinstance(var_, tvm.expr.Max) and str(var_) == "max(max(var_0, var_1), var_2)"
    ori_var_assert = isinstance(ori_var, tvm.expr.Max) and str(ori_var) == "max(max(ori_0, ori_1), ori_2)"

    annotation_assert = var_annotation["axis_type"] == "C1" and \
                        str(var_annotation["original"]) == "max(max(ori_0, ori_1), ori_2)"
    ori_annotation_assert = ori_annotation == {"axis_type": "C"}

    return all((var_assert, ori_var_assert, annotation_assert, ori_annotation_assert))


@add_cust_test_func(ut_case)
def test_prod(_):
    ori_0 = var_api.var("ori_0", annotation={"axis_type": "C"})
    ori_1 = var_api.var("ori_1", annotation={"axis_type": "C"})
    ori_2 = var_api.var("ori_2", annotation={"axis_type": "C"})

    var_0 = var_api.var("var_0", annotation={"axis_type": "C1", "original": ori_0})
    var_1 = var_api.var("var_1", annotation={"axis_type": "C1", "original": ori_1})
    var_2 = var_api.var("var_2", annotation={"axis_type": "C1", "original": ori_2})

    var_ = var_api.prod(var_0, var_1, var_2)
    ori_var = var_api.get_attr(var_, "original")

    var_annotation = var_api.get_annotation(var_)
    ori_annotation = var_api.get_annotation(ori_var)

    var_assert = isinstance(var_, tvm.expr.Mul) and str(var_) == "((var_0*var_1)*var_2)"
    ori_var_assert = isinstance(ori_var, tvm.expr.Mul) and str(ori_var) == "((ori_0*ori_1)*ori_2)"

    annotation_assert = var_annotation["axis_type"] == "C1" and \
                        str(var_annotation["original"]) == "((ori_0*ori_1)*ori_2)"
    ori_annotation_assert = ori_annotation == {"axis_type": "C"}

    return all((var_assert, ori_var_assert, annotation_assert, ori_annotation_assert))


@add_cust_test_func(ut_case)
def test_bit(_):
    ori_0 = var_api.var("ori_0", annotation={"axis_type": "C"})
    ori_1 = var_api.var("ori_1", annotation={"axis_type": "C"})
    ori_2 = var_api.var("ori_2", annotation={"axis_type": "C"})

    var_0 = var_api.var("var_0", annotation={"axis_type": "C1", "original": ori_0})
    var_1 = var_api.var("var_1", annotation={"axis_type": "C1", "original": ori_1})
    var_2 = var_api.var("var_2", annotation={"axis_type": "C1", "original": ori_2})

    var_ = var_api.bit(var_0, var_1, var_2)
    ori_var = var_api.get_attr(var_, "original")

    var_annotation = var_api.get_annotation(var_)
    ori_annotation = var_api.get_annotation(ori_var)

    var_assert = isinstance(var_, tvm.expr.Call) and \
                 str(var_) == "bitwise_and(bitwise_and(var_0, var_1), var_2)"
    ori_var_assert = isinstance(ori_var, tvm.expr.Call) and \
                     str(ori_var) == "bitwise_and(bitwise_and(ori_0, ori_1), ori_2)"

    annotation_assert = var_annotation["axis_type"] == "C1" and \
                        str(var_annotation["original"]) == "bitwise_and(bitwise_and(ori_0, ori_1), ori_2)"
    ori_annotation_assert = ori_annotation == {"axis_type": "C"}

    return all((var_assert, ori_var_assert, annotation_assert, ori_annotation_assert))


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        ret = v.test_func(None)
        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
