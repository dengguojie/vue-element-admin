from __future__ import absolute_import
from tbe import tvm
import tbe.dsl as tbe
from tbe.common.register import register_op_compute
from tbe.common.utils import shape_refine
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType


@register_op_compute("conv2d")
def conv2d_compute(x, filter, y):
    """
    The compute function of the Conv2D implementation.
    """
    res = tbe.XXX(x, filter)
    return res


# Define the kernel info of Conv2D.
conv2d_op_info = TBERegOp("Conv2D") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("conv2d.so") \
    .compute_cost(10) \
    .kernel_name("conv2d_impl") \
    .attr("strides", "required", "listInt", "all")\
    .attr("pads", "required", "listInt", "all")\
    .attr("dilations", "optional", "listInt", "all")\
    .input(0, "x", False, "required", "all")\
    .input(1, "filter", False, "required", "all")\
    .output(0, "y", False, "required", "all")\
    .dtype_format(DataType.I8_NHWC, DataType.I8_NHWC, DataType.I8_NHWC)\
    .dtype_format(DataType.I8_NCHW, DataType.I8_NCHW, DataType.I8_NCHW)\
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(conv2d_op_info)
def conv2d_impl(x, filter, y, kernel_name="conv2d_impl"):
    """
    The entry function of the Conv2D implementation.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    shape = shape_refine(shape)
    input0 = tvm.placeholder(shape, name="input0", dtype=dtype.lower())
    input1 = tvm.placeholder(shape, name="input1", dtype=dtype.lower())

    with tvm.target.cce():
        res = conv2d_compute(input0, input1, y)
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input0, input1, res]}

    tbe.build(sch, config)
