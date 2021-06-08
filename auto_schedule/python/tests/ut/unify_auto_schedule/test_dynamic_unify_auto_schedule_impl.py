# # -*- coding:utf-8 -*-
import warnings
import copy

from sch_test_frame.ut import OpUT
import tbe
from tbe import tvm
from tbe.dsl.base.classifier import classify_elewise
from tbe.common import buildcfg
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common.register import register_op_compute


warnings.filterwarnings("ignore")
ut_case = OpUT("unify_auto_schedule", "unify_auto_schedule.test_dynamic_unify_auto_schedule_impl")


@register_op_compute("test_unify_auto_schedule", op_mode="dynamic", support_fusion=True)
def test_reduce_sum_d_compute(x,
                              y,
                              axis=None,
                              keepdims=None):
    res = tbe.dsl.reduce_sum(x, axis, keepdims)
    return res


@register_operator("test_unify_auto_schedule")
def test_dyn_reduce_sum(x, y, axis, keepdims, kernel_name="test_unify_auto_schedule"):
    input_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = 'before'
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})

    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = test_reduce_sum_d_compute(data1, y, axis.get("value"), keepdims)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


def test_unify_auto_schedule(_):
    org_dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
    dynamic_config = copy.deepcopy(org_dynamic_config)
    dynamic_config.update({"enable_op_prebuild": True})
    with buildcfg.build_config(**dynamic_config):
        with tbe.common.context.op_context.OpContext("dynamic"):
            op_info = tbe.common.context.op_info.OpInfo("test_unify_auto_schedule", "test_unify_auto_schedule")
            tbe.common.context.op_context.get_context().add_op_info(op_info)

            test_dyn_reduce_sum(
                {"dtype": "float32", "shape": (-1, -1), "org_shape": (-1, -1), "range": [(1, None), (1, None)], },
                {"dtype": "float32", "shape": (-1, -1), "org_shape": (-1, -1), "range": [(1, None), (1, None)], },
                [1],
                True
            )

            ins = tbe.common.context.op_context.get_context().get_build_res("pattern")
            expect_ins = "CommReduce"

            return ins == expect_ins


ut_case.add_cust_test_func(test_func=test_unify_auto_schedule)
