from __future__ import absolute_import
from tbe import tvm
import tbe.dsl as tbe
from tbe.common.register import register_op_compute
from tbe.common.utils import shape_refine
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType


@register_op_compute("square")
def square_compute(x, y, z):
    """
    The compute function of the Square implementation.
    """
    res = tbe.XXX(x, y)
    return res


# Define the kernel info of Square.
square_op_info = TBERegOp("Square") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("square.so") \
    .compute_cost(10) \
    .kernel_name("square_impl") \
    .input(0, "x", False, "required", "all")\
    .input(1, "y", False, "required", "all")\
    .output(0, "z", False, "required", "all")\
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default)\
    .dtype_format(DataType.I16_Default, DataType.I16_Default, DataType.I16_Default)\
    .dtype_format(DataType.I32_Default, DataType.I32_Default, DataType.I32_Default)\
    .dtype_format(DataType.I64_Default, DataType.I64_Default, DataType.I64_Default)\
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(square_op_info)
def square_impl(x, y, z, kernel_name="square_impl"):
    """
    The entry function of the Square implementation.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    shape = shape_refine(shape)
    input0 = tvm.placeholder(shape, name="input0", dtype=dtype.lower())
    input1 = tvm.placeholder(shape, name="input1", dtype=dtype.lower())

    with tvm.target.cce():
        res = square_compute(input0, input1, z)
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input0, input1, res]}

    tbe.build(sch, config)
