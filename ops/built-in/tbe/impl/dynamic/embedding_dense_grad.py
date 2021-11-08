"""
Copyright (C) Huawei Technologies Co., Ltd 2021-2021. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

embedding_dense_grad
"""
# 'pylint: disable=too-many-arguments,invalid-name,too-many-statements
# 'pylint: disable=unused-argument,too-many-instance-attributes
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator

RESERVE_SIZE = 16 * 1024
MAX_INT32 = 2 ** 31 - 1
SCALAR_TENSOR_SIZE = 32
TILING_ARG_NUM = 64
TILING_MODE_1 = 1
GRAD_TENSOR_PART = 512


class EmbeddingDenseGrad():
    """
    Function: store EmbeddingDenseGrad parameters  and compute EmbeddingDenseGrad
    """
    def __init__(
            self,
            grad,
            indices,
            y,
            num_weights,
            padding_idx,
            scale_grad_by_freq,
            kernel_name="embedding_dense_grad"):
        """
        init the ShuffleChannel parameters

        Parameters
        ----------
            input_dict: input_dict is a dict, the keys as follow:
                grad: dict,shape and datatype,datatype supports float32
                indices: dict,shape and datatype,datatype supports int32
                y:dict,shape and datatype,datatype supports float32
                num_weights:the number of words in dict
                padding_idx:judge grad_weight of which word is zero
                scale_grad_by_freq: judge whether or not  scale_grad
                kernel_name: cce kernel name, default value is "embedding_dense_grad"
        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.num_weights = num_weights
        self.dtype_grad = grad.get("dtype")
        self.dtype_indices = indices.get("dtype")
        self.kernel_name = kernel_name
        self.tiling_dtype = 'int32'
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

        '''Data reading and writing on UB must be 32B aligned. This parameter is used
        to calculate tensor division and data handling instruction parameters'''
        self.aicore_num = 32
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)

        # Calculate how many indicators elements can be stored in a block
        # according to the input data type
        self.dtype_bytes_size_indices = cce.get_bit_len(
            self.dtype_indices) // 8
        self.indices_each_block = block_bite_size // self.dtype_bytes_size_indices
        '''Calculate how many grad elements can be stored in a block according to
        the input data type'''
        self.dtype_bytes_size_grad = cce.get_bit_len(
            self.dtype_grad) // 8
        self.grad_each_block = block_bite_size // self.dtype_bytes_size_grad

        '''Calculate how many counts elements can be stored in a block according
        to the input data type'''
        self.dtype_bytes_size_tiling = cce.get_bit_len(
            self.tiling_dtype) // 8
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling
        self.dtype_bytes_size_counts = cce.get_bit_len(
            self.dtype_indices) // 8
        self.counts_each_block = block_bite_size // self.dtype_bytes_size_counts

        '''Fix the space of self.num_weights*self.dtype_bytes_size_counts Bytes to counts on ub,
        so you only need to calculate how many elements are placed on ub for indicators and grad,
        and perform 32B alignment'''
        if self.num_weights // self.aicore_num > 0:
            self.ub_indices_size = (self.ub_size_bytes - (self.num_weights // self.aicore_num + self.num_weights %
                                                          self.aicore_num) * self.dtype_bytes_size_counts
                                    - RESERVE_SIZE) \
            // (GRAD_TENSOR_PART * self.dtype_bytes_size_grad + self.dtype_bytes_size_indices) // \
            self.indices_each_block * self.indices_each_block
            self.counts_size = self.num_weights // self.aicore_num + self.num_weights % self.aicore_num
        else:
            self.ub_indices_size = (self.ub_size_bytes - self.num_weights * self.dtype_bytes_size_counts
                                    - RESERVE_SIZE) \
                // (GRAD_TENSOR_PART * self.dtype_bytes_size_grad + self.dtype_bytes_size_indices) // \
                self.indices_each_block * self.indices_each_block
            self.counts_size = self.num_weights
            self.aicore_num = 1
        self.ub_grad_size = self.ub_indices_size * GRAD_TENSOR_PART
        '''The vector instruction calculates a maximum of 8 blocks per repeat.
        This parameter is the maximum value of the mask when grad performs vector calculation'''
        self.vector_mask_max_grad = 8 * self.grad_each_block
        '''The vector instruction calculates a maximum of 8 blocks per repeat. This parameter
        is the maximum value of the mask when counts is used for vector calculation.'''
        self.vector_max_repeat = 255
        self.vector_mask_max_counts = 8 * self.counts_each_block
        self.vec_max_grad_element = self.vector_max_repeat * self.vector_mask_max_grad
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,), name='tiling_gm',
                                                  scope=tik.scope_gm)
        self.indices = None
        self.scale_float = None
        self.grad_weight = None
        self.counts_ub = None
        self.numel_grad = None
        self.numel_indices = None
        self.index = None
        self.add_tensor = None
        self.embedding_dim = None
        self.end = None
        self.scale_int = None
        self.grad_ub = None
        self.new_numel_indices = None
        self.indices_ub = None
        self.grad = None
        self.k = None
        self.begin = None

    def get_tiling_args(self, tiling_ub):
        """
        get tiling args from tling_ub

        Parameters
        ----------
        tiling_ub: s tensor with tiling_args in ub

        Returns
        -------
        None
        """
        self.embedding_dim = self.tik_instance.Scalar(
            self.dtype_indices, name='embedding_dim')
        self.numel_indices = self.tik_instance.Scalar(
            self.dtype_indices, name='numel_indices')

        self.embedding_dim.set_as(tiling_ub[1])
        self.numel_indices.set_as(tiling_ub[0])

    def embedding_dense_grad_compute_tiling(self):
        """
        Compute the embedding_dense_grad op

        Parameters
       ----------
        None

        Returns
        -------
        None
        """
        self.gm_to_data()
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as block_id:
            self.begin = block_id * self.num_weights // self.aicore_num
            with self.tik_instance.if_scope(block_id == self.aicore_num - 1):
                self.end = self.num_weights
            with self.tik_instance.else_scope():
                self.end = (block_id + 1) * self.num_weights // self.aicore_num
            tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,), name='tiling_ub',
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, SCALAR_TENSOR_SIZE //
                                        self.tiling_each_block, 0, 0)
            self.get_tiling_args(tiling_ub)
            self.fill_grad_weight()
            self.ub_to_data_fill_counts()
            mode_of_cal = self.tik_instance.Scalar(self.dtype_indices, name='mode_of_cal')
            mode_of_cal.set_as(tiling_ub[2])
            with self.tik_instance.if_scope(mode_of_cal == TILING_MODE_1):
                self.first_cal_mode()

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name, inputs=[
                self.grad, self.indices], outputs=[
                self.grad_weight], flowtable=[self.tiling_gm])
        tbe_context.get_context().add_compile_info('vars', {'core_num': self.aicore_num,
                                        'num_weights': self.num_weights,
                                        'padding_idx': self.padding_idx,
                                        'scale_grad_by_freq': self.scale_grad_by_freq})
        return self.tik_instance

    def gm_to_data(self):
        """
        Allocate space for grad, indices and grad_weight on gm
        Parameters
       ----------
        None

        Returns
        -------
        None
        """
        # Allocate space for grad, indices and grad_weight on gm
        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (MAX_INT32, ),
                                                    name="grad_weight", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.dtype_indices, (MAX_INT32, ), name="indices",
                                                scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor(self.dtype_grad, (MAX_INT32, ), name="grad", scope=tik.scope_gm)

    def fill_grad_weight(self):
        """
        Fill the grad_weight tesnor with 0

        Parameters
       ----------
        None

        Returns
        -------
        None
        """
        # Create a new space to initialize grad_weightm
        with self.tik_instance.new_stmt_scope():
            # Initialize fill_tensor with 0, which is used to initialize grad_weight later
            fill_tensor = self.tik_instance.Tensor(self.dtype_grad, (GRAD_TENSOR_PART * 2, ), name="fill_tensor",
                                                   scope=tik.scope_ubuf)
            # Define scalar_float0 to fill grad_weight
            scalar_float0 = self.tik_instance.Scalar(init_value=0, dtype=self.dtype_grad)
            with self.tik_instance.if_scope(self.embedding_dim // self.vector_mask_max_grad > 0):
                self.tik_instance.vec_dup(self.vector_mask_max_grad, fill_tensor, scalar_float0,
                                          self.embedding_dim // self.vector_mask_max_grad, 8)
            with self.tik_instance.if_scope(self.embedding_dim % self.vector_mask_max_grad != 0):
                self.tik_instance.vec_dup(self.embedding_dim % self.vector_mask_max_grad, fill_tensor[
                    self.embedding_dim // self.vector_mask_max_grad * self.vector_mask_max_grad],
                                          scalar_float0, 1, 8)
            # Use fill_tensor on ub to fill grad_weight on gm and do 0 initialization
            with self.tik_instance.for_range(self.begin, self.end) as i:
                self.tik_instance.data_move(self.grad_weight[i * self.embedding_dim],
                                            fill_tensor, 0, 1, self.embedding_dim // self.grad_each_block, 0, 0)

    def ub_to_data_fill_counts(self):
        """
        Allocate space for grad, indices and counts on ub
        use 0 to fill counts

        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        # Allocate space for grad, indices and counts on ub
        self.counts_ub = self.tik_instance.Tensor(self.dtype_indices, (self.counts_size,), name="counts_ub",
                                                  scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, (self.ub_indices_size, ), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_grad_size, ),
                                                name="grad_ub", scope=tik.scope_ubuf)

        scalar_int0 = self.tik_instance.Scalar(init_value=0, dtype=self.dtype_indices)
        vec_max_count_element = self.vector_max_repeat * self.vector_mask_max_counts
        if self.counts_size // vec_max_count_element > 0:
            with self.tik_instance.for_range(0, self.counts_size // vec_max_count_element) as i:
                self.tik_instance.vec_dup(self.vector_mask_max_counts, self.counts_ub[i * vec_max_count_element],
                                          scalar_int0, self.vector_max_repeat, 8)

        repeat_count_num = self.counts_size % vec_max_count_element
        if repeat_count_num > 0 and repeat_count_num // self.vector_mask_max_counts > 0:
            self.tik_instance.vec_dup(self.vector_mask_max_counts,
                                      self.counts_ub[self.counts_size // vec_max_count_element * vec_max_count_element],
                                      scalar_int0, repeat_count_num // self.vector_mask_max_counts, 8)

        last_num_counts = self.counts_size - self.counts_size // vec_max_count_element * vec_max_count_element - \
            self.counts_size % vec_max_count_element // self.vector_mask_max_counts * self.vector_mask_max_counts
        if last_num_counts > 0:
            offset_counts = self.counts_size // vec_max_count_element * vec_max_count_element + \
                self.counts_size % vec_max_count_element // self.vector_mask_max_counts \
                * self.vector_mask_max_counts
            self.tik_instance.vec_dup(last_num_counts, self.counts_ub[offset_counts], scalar_int0, 1, 8)

    def first_cal_mode(self):
        """
        the first cal mode of embedding_dense_grad op

        Parameters
       ----------
        None

        Returns
        -------
        None
        """
        self.base_count_words_compute()

    def base_count_words_compute(self):
        """
        when sf is True,use base function to count words

        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        if self.scale_grad_by_freq:
            # Define k, the scalar used to index the elements of indicators
            self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
            # Move indexes blocks from gm to ub
            with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
                self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.ub_indices_size)
        self.remaining_count_words_compute()

    def remaining_count_words_compute(self):
        """
        when sf is True,use remaining function to count words

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if self.scale_grad_by_freq:
            with self.tik_instance.if_scope(self.numel_indices % self.ub_indices_size != 0):
                offset_indices_move = self.tik_instance.Scalar(self.dtype_indices, name='offset_indices_move')
                offset_indices_move.set_as(self.numel_indices // self.ub_indices_size * self.ub_indices_size)
                burst_len_indices = self.tik_instance.Scalar(self.dtype_indices, name='burst_len_indices')
                burst_len_indices.set_as((self.numel_indices % self.ub_indices_size - 1) // self.indices_each_block + 1)
                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices,
                                            0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.numel_indices % self.ub_indices_size)
        self.base_compute_grad_weight()

    def base_compute_grad_weight(self):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        self.add_tensor = self.tik_instance.Tensor(
            self.dtype_grad, (1, 2 * GRAD_TENSOR_PART), name="add_tensor", scope=tik.scope_ubuf)
        self.scale_int = self.tik_instance.Scalar(dtype=self.dtype_indices)
        self.scale_float = self.tik_instance.Scalar(
            init_value=1.0, dtype=self.dtype_grad)
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices, name='k')
        # Move indexes and grad blocks from gm to ub
        line_num = self.tik_instance.Scalar(dtype='int32', name='line_num')
        line_num.set_as(self.ub_grad_size // GRAD_TENSOR_PART // 2)
        with self.tik_instance.for_range(0, self.numel_indices // line_num) as i1:
            self.tik_instance.data_move(self.indices_ub, self.indices[i1 * line_num], 0, 1,
                                        line_num // self.indices_each_block, 0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad[i1 * line_num * self.embedding_dim],
                                        0, 1, line_num * self.embedding_dim // self.grad_each_block, 0, 0)
            self.add_same_word_grad(line_num)
        self.remaining_compute_grad_weight(line_num)

    def remaining_compute_grad_weight(self, line_num):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.numel_indices % line_num != 0):
            offset_indices_move = self.tik_instance.Scalar(self.dtype_indices, name='offset_indices_move')
            offset_indices_move.set_as(self.numel_indices // line_num * line_num)
            burst_len_indices = self.tik_instance.Scalar(self.dtype_indices, name='burst_len_indices')
            burst_len_indices.set_as((self.numel_indices % line_num - 1) // self.indices_each_block + 1)
            offset_grad_move = self.numel_indices // line_num * \
                line_num * self.embedding_dim
            burst_len_grad = self.numel_indices % line_num * \
                self.embedding_dim // self.grad_each_block

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad(self.numel_indices % line_num)

    def count_words_compute(self, total):
        """
        when sf is True,use this function to count word frequency

        Parameters
        ----------
        total:int32,the total size need to count word frequency

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, total) as index:
            self.k.set_as(self.indices_ub[index])
            with self.tik_instance.if_scope(tik.all(self.k < self.end, self.k >= self.begin)):
                with self.tik_instance.if_scope(self.k != self.padding_idx):
                    tmp = self.tik_instance.Scalar(dtype=self.dtype_indices)
                    tmp.set_as(self.counts_ub[self.k - self.begin])
                    self.counts_ub[self.k - self.begin].set_as(tmp + 1)

    def add_same_word_grad(self, total):
        """
        when sf is False,use this function to compute grad_weight
        by add and scale=1
        when sf is True,use this function to compute grad_weight
        by add and scale=1/counts[k]

        Parameters
        ----------
        total:int32,the total size need to compute grad_weight

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, total) as self.index:
            self.k.set_as(self.indices_ub[self.index])
            with self.tik_instance.if_scope(self.k != self.padding_idx):
                with self.tik_instance.if_scope(tik.all(self.k < self.end, self.k >= self.begin)):
                    self.tik_instance.data_move(self.add_tensor, self.grad_weight[self.k * self.embedding_dim], 0, 1,
                                                self.embedding_dim // self.grad_each_block, 0, 0)
                    self.cac_result()

    def cac_result(self):
        """
        caculate the rersult of grad_weight

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.scale_grad_by_freq:
            self.scale_int.set_as(self.counts_ub[self.k - self.begin])
            self.tik_instance.scalar_conv('', self.scale_float, self.scale_int)
            self.scale_float.set_as(1.0 / self.scale_float)
        with self.tik_instance.if_scope(self.embedding_dim // self.vector_mask_max_grad > 0):
            self.tik_instance.vec_axpy(self.vector_mask_max_grad, self.add_tensor,
                                       self.grad_ub[self.index * self.embedding_dim], self.scale_float,
                                       self.embedding_dim // self.vector_mask_max_grad, 8, 8)
        with self.tik_instance.if_scope(self.embedding_dim % self.vector_mask_max_grad > 0):
            self.tik_instance.vec_axpy(self.embedding_dim % self.vector_mask_max_grad,
                                       self.add_tensor[self.embedding_dim // self.vector_mask_max_grad *
                                                       self.vector_mask_max_grad],
                                       self.grad_ub[self.index * self.embedding_dim + self.embedding_dim //
                                                    self.vector_mask_max_grad * self.vector_mask_max_grad],
                                       self.scale_float, 1, 8, 8)
        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim], self.add_tensor, 0, 1,
                                    self.embedding_dim // self.grad_each_block, 0, 0)


@register_operator('EmbeddingDenseGrad')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_BOOL,
    para_check.KERNEL_NAME)
def embedding_dense_grad(
        grad,
        indices,
        y,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        kernel_name="embedding_dense_grad"):
    """
    the main function of embedding_dense_grad

    Parameters
    ----------
    grad: dict,shape and datatype,
    datatype supports float32
    indices: dict,shape and datatype,
    datatype supports int32
    y:dict,shape and datatype,
    datatype supports float32
    num_weights:the number of words in dict
    padding_idx:judge grad_weight of which word is zero
    scale_grad_by_freq: judge whether or not  scale_grad
    kernel_name: cce kernel name, default value is "embedding_dense_grad"
    Returns
    -------
    tik_instance: tik_instance
    """
    check_indices_param(indices)
    check_grad_param(grad)
    embedding_dense_grad_instance = EmbeddingDenseGrad(
        grad, indices, y, num_weights, padding_idx, scale_grad_by_freq, kernel_name)
    tik_instance = embedding_dense_grad_instance.embedding_dense_grad_compute_tiling()
    return tik_instance


def check_grad_param(grad_dic):
    """
    check the parameters grad is valid

    Parameters
    ----------
    grad_dic: dict,shape and datatype,datatype supports float32
    Returns
    -------
    None
    """
    grad_dtype = grad_dic.get("dtype").lower()
    para_check.check_dtype(grad_dtype, ["float32"], param_name="grad")


def check_indices_param(indices_dic):
    """
    check the parameters indices is valid

    Parameters
    ----------
    indices_dic: dict,shape and datatype,datatype supports int32
    Returns
    -------
    None
    """
    indices_dtype = indices_dic.get("dtype").lower()
    para_check.check_dtype(indices_dtype, ["int32"], param_name="indices")
