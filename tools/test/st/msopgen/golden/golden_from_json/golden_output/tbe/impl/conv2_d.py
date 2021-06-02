import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute
from topi import generic


@register_op_compute("conv2_d")
def conv2_d_compute(x, filter, y, strides, pads, dilations, kernel_name="conv2_d"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    res = tbe.XXX(x, filter)
    return res

def conv2_d(x, filter, y, strides, pads, dilations, kernel_name="conv2_d"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x = tvm.placeholder(x.get("shape"), dtype=x.get("dtype"), name="data_x")
    data_filter = tvm.placeholder(filter.get("shape"), dtype=filter.get("dtype"), name="data_filter")

    res = conv2_d_compute(data_x, data_filter, y, strides, pads, dilations, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x, data_filter, res]}
    tbe.build(schedule, config)
    