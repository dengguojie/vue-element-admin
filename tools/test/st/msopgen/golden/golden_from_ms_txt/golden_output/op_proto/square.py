from mindspore.ops import prim_attr_register, PrimitiveWithInfer
import mindspore.ops as ops
# description


class Square(PrimitiveWithInfer):
    """
    The definition of the Square primitive.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['z'])
        # Import the entry function of the kernel implementation from relative
        #  path or PYTHONPATH.
        from square_impl import square_impl

    def infer_shape(self, x_shape, y_shape):
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype