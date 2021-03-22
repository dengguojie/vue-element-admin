from mindspore.ops import prim_attr_register
from mindspore.ops import PrimitiveWithInfer


class MatrixCombine(PrimitiveWithInfer):
    """
    move the batch matrix to result matrix diag part.
    The rank of input tensors must be `3`.

    Inputs:
        - **input_x** (Tensor) - The shape of the tensor is :math:`(N, D, D)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N * D, N * D)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 128, 128]), mindspore.float32)
        >>> cusmatrixcombine = MatrixCombine()
        >>> output = cusmatrixcombine(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize MatrixCombine"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from matrix_combine_impl import matrix_combine

    def infer_shape(self, data_shape):
        dim_a, dim_b, dim_c = data_shape
        shape = [dim_a * dim_b, dim_a * dim_c]
        return shape

    def infer_dtype(self, data_dtype):
        return data_dtype
