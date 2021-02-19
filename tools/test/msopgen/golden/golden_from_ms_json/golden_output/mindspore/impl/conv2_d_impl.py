from __future__ import absolute_import
from te import tvm
from topi import generic
import te.lang.cce
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

def conv2_d_compute(x, filter, y):
    """
    The compute function of the Conv2D implementation.
    """
    res = te.lang.cce.XXX(x, filter)
    return res


# Define the kernel info of Conv2D.
conv2_d_op_info = TBERegOp("Conv2D") \
    .fusion_type("OPAQUE") \
    .partial_flag(True) \
    .async_flag(False) \
    .binfile_name("conv2_d.so") \
    .compute_cost(10) \
    .kernel_name("conv2_d_impl") \
    .input(0, "x", False, "required", "all")\
    .input(0, "filter", False, "required", "all")\
    .output(0, "y", False, "required", "all")\
    .dtype_format(DataType.I8_NHWC, DataType.I8_NHWC, DataType.I8_NHWC)\
    .dtype_format(DataType.I8_NCHW, DataType.I8_NCHW, DataType.I8_NCHW)\
    .get_op_info()


# Binding kernel info with the kernel implementation.
@op_info_register(conv2_d_op_info)
def conv2_d_impl(x, filter, y, kernel_name="conv2_d_impl"):
    """
    The entry function of the Conv2D implementation.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    shape = util.shape_refine(shape)
    data1 = tvm.placeholder(shape, name="data1", dtype=dtype.lower())
    data2 = tvm.placeholder(shape, name="data2", dtype=dtype.lower())

    with tvm.target.cce():
        res = conv2_d_compute(data1, data2, y)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}

    te.lang.cce.cce_build_code(sch, config)
