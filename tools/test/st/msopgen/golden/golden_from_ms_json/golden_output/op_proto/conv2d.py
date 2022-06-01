from mindspore.ops import prim_attr_register, PrimitiveWithInfer
import mindspore.ops as ops
# description


class Conv2D(PrimitiveWithInfer):
    """
    The definition of the Conv2D primitive.
    """
    @prim_attr_register
    def __init__(self, strides, pads, dilations):
        self.init_prim_io_names(inputs=['x', 'filter'], outputs=['y'])
        # Import the entry function of the kernel implementation from relative
        #  path or PYTHONPATH.
        from conv2d_impl import conv2d_impl

    def infer_shape(self, x_shape, filter_shape):
        return x_shape

    def infer_dtype(self, x_dtype, filter_dtype):
        return x_dtype