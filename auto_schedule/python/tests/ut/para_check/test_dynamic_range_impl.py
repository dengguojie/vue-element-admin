import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.common import context as tbe_context
from tbe.dsl import classify
from tbe.common.utils import para_check


@register_operator("para_check")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_OUTPUT, para_check.KERNEL_NAME)
def dsl_dynamic_range(x, y, z, kernel_name="dsl_dynamic_range"):
    """

    Parameters
    ----------
    Algorithm: para_check

    Parameters:

    x: the dict of input data, support float16

    y: the dict of output

    kernel_name: cce kernel name, default value is "para_check".

    Returns
    -------
    None
    """

    input_dtype = x.get("dtype")

    ins = classify([x, y], "elewise")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = tbe.dsl.vadd(data1, data2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)

    tbe_context.get_context().add_compile_info("ops_compile_info", "ops_compile_info")


ut_case = OpUT("para_check", "para_check.test_dynamic_range_impl", "dsl_dynamic_range")

case1 = {
    "params": [
        {
            "shape": (-2,),
            "ori_shape":(-2,),
            "format": "ND",
            "ori_format": "ND",
            "dtype": "float16",
        },
        {
            "shape": (-2,),
            "ori_shape":(-2,),
            "format": "ND",
            "ori_format": "ND",
            "dtype": "float16",
        },
        {
            "shape": (-2,),
            "ori_shape":(-2,),
            "format": "ND",
            "ori_format": "ND",
            "dtype": "float16",
        }],
    "case_name":
        "test_dynamic_range_1",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["all", ], case1)