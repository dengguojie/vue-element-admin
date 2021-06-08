import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute


@register_op_compute("conv2d_tik")
def conv2d_tik_compute(x1, x2, y, strides, pads, dilations, kernel_name="conv2d_tik"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    res = tbe.XXX(x1, x2)
    return res

def conv2d_tik(x1, x2, y, strides, pads, dilations, kernel_name="conv2d_tik"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """
    data_x1 = tvm.placeholder(x1.get("shape"), dtype=x1.get("dtype"), name="data_x1")
    data_x2 = tvm.placeholder(x2.get("shape"), dtype=x2.get("dtype"), name="data_x2")

    res = conv2d_tik_compute(data_x1, data_x2, y, strides, pads, dilations, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, res]}
    tbe.build(schedule, config)
    