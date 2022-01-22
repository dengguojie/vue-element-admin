# # -*- coding:utf-8 -*-
import warnings

from sch_test_frame.ut import OpUT
import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.common.register import register_op_compute


warnings.filterwarnings("ignore")
ut_case = OpUT("dsl_build", "dsl_build.test_dynamic_dsl_build_impl")


@register_op_compute("test_dsl_build", op_mode="dynamic", support_fusion=True)
def test_build_reduce_sum_d_compute(x,
                                    y,
                                    axis=None,
                                    keepdims=None):
    res = tbe.dsl.reduce_sum(x, axis, keepdims)
    return res


@register_operator("test_dsl_build")
def test_dyn_build_reduce_sum(x, y, axis, keepdims, kernel_name="test_dsl_build"):
    input_dtype = x.get("dtype")
    x["rel_pos_to_reduce"] = 'before'
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
    ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})

    schedules, tensors = [], []

    for (x, axis) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = test_build_reduce_sum_d_compute(data1, y, axis.get("value"), keepdims)
            tensors.append([data1, res])

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


def test_dsl_build(_):
    with tbe.common.context.op_context.OpContext("static"):
        op_info = tbe.common.context.op_info.OpInfo("test_dsl_build", "test_dsl_build")
        tbe.common.context.op_context.get_context().add_op_info(op_info)

        try:
            test_dyn_build_reduce_sum(
                {"dtype": "float32", "shape": (10, 10), "org_shape": (10, 10), "range": [(10, 10), (10, 10)], },
                {"dtype": "float32", "shape": (10, 1), "org_shape": (10, 1), "range": [(10, 10), (1, 1)], },
                [1],
                True
            )
        except RuntimeError:
            return True
        else:
            return False


ut_case.add_cust_test_func(support_soc=["Ascend310"], test_func=test_dsl_build)
