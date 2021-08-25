import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check


@register_op_compute("conv2d")
def conv2d_compute(x, filter, y, strides, pads, dilations, kernel_name="conv2d"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    res = tbe.XXX(x, filter)
    return res

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def conv2d(x, filter, y, strides, pads, dilations, kernel_name="conv2d"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_filter = tvm.placeholder(filter.get("shape"), dtype=filter.get("dtype"), name="data_filter")

    res = conv2d_compute(data_x, data_filter, y, strides, pads, dilations, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_filter, res]}
    tbe.build(schedule, config)
    