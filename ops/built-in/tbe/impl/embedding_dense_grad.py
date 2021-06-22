"""
Copyright (C) Huawei Technologies Co., Ltd 2020-2020. All rights reserved.

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

from te import tik
import math
from te import platform as cce
from te.utils import op_utils

RESERVE_SIZE = 16 * 1024
BLOCK = 8
FILL_DIM = 1024


class EmbeddingDenseGrad(object):
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
        self.grad_shape = grad["shape"]
        self.dtype_grad = grad.get("dtype")
        self.indices_shape = indices["shape"]
        self.dtype_indices = indices.get("dtype")
        self.embedding_dim = grad["shape"][-1]
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.padding_idx = padding_idx
        self.num_weights = num_weights
        self.kernel_name = kernel_name
        self.scale_grad_by_freq = scale_grad_by_freq

        '''Data reading and writing on UB must be 32B aligned. This parameter is used
        to calculate tensor division and data handling instruction parameters
        '''
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.CceProductParams().getParams("Unified_Buffer")
        self.aicore_num = 32

        '''Calculate how many grad elements can be stored in a block according to
        the input data type
        '''
        self.dtype_bytes_size_grad = cce.cce_intrin.get_bit_len(
            self.dtype_grad) // 8
        self.grad_each_block = block_bite_size // self.dtype_bytes_size_grad
        '''Calculate how many counts elements can be stored in a block according 
        to the input data type
        '''
        self.dtype_bytes_size_counts = cce.cce_intrin.get_bit_len(
            self.dtype_indices) // 8
        self.counts_each_block = block_bite_size // self.dtype_bytes_size_counts

        '''Calculate how many indicators elements can be stored in a block
        according to the input data type
        '''
        self.dtype_bytes_size_indices = cce.cce_intrin.get_bit_len(
            self.dtype_indices) // 8
        self.indices_each_block = block_bite_size // self.dtype_bytes_size_indices

        '''Fix the space of self.num_weights*self.dtype_bytes_size_counts Bytes 
        to counts on ub, so you only need to calculate how many elements are 
        placed on ub for indicators and grad, and perform 32B alignment
        '''
        if self.num_weights // self.aicore_num > 0 and self.embedding_dim >= 8:
            if self.scale_grad_by_freq:
                self.ub_indices_size = (self.ub_size_bytes - (self.num_weights // self.aicore_num +
                                                              self.num_weights % self.aicore_num) *
                                        self.dtype_bytes_size_counts - RESERVE_SIZE) \
                                       // (self.embedding_dim * self.dtype_bytes_size_grad +
                                           self.dtype_bytes_size_indices)\
                                       // self.indices_each_block * self.indices_each_block
                self.counts_size = self.num_weights // self.aicore_num + \
                                   self.num_weights % self.aicore_num
            else:
                self.ub_indices_size = (self.ub_size_bytes - RESERVE_SIZE) \
                                       // (self.embedding_dim * self.dtype_bytes_size_grad +
                                           self.dtype_bytes_size_indices)\
                                       // self.indices_each_block * self.indices_each_block
        else:
            if self.scale_grad_by_freq:
                self.ub_indices_size = (self.ub_size_bytes - self.num_weights * self.dtype_bytes_size_counts -
                                        RESERVE_SIZE)// (self.embedding_dim * self.dtype_bytes_size_grad +
                                                         self.dtype_bytes_size_indices) // \
                                       self.indices_each_block * self.indices_each_block
                self.counts_size = self.num_weights
            else:
                self.ub_indices_size = (self.ub_size_bytes - RESERVE_SIZE)// (self.embedding_dim *
                                                                              self.dtype_bytes_size_grad +
                                                                              self.dtype_bytes_size_indices) // \
                                       self.indices_each_block * self.indices_each_block
        self.ub_grad_size = self.ub_indices_size * self.embedding_dim
        '''The vector instruction calculates a maximum of 8 blocks per repeat.
        This parameter is the maximum value of the mask when grad performs vector calculation
        '''
        self.vector_mask_max_counts = 8 * self.counts_each_block
        self.scale_int = None
        self.end = None
        self.new_numel_indices = None
        self.grad_ub = None
        self.vec_max_grad_element = None
        self.indices_ub = None
        self.grad = None
        self.vector_max_repeat = None
        self.begin = None
        self.k = None
        self.scale_float = None
        self.indices = None
        self.counts_ub = None
        self.grad_weight = None
        self.numel_indices = None
        self.numel_grad = None
        self.add_tensor = None
        self.index = None
        self.vector_mask_max_grad = 8 * self.grad_each_block

    def embedding_dense_grad_compute(self):
        """
        compute embedding_dense_grad

        Parameters
       ----------
        None
        Returns
        -------
        tik_instance: tik_instance
        """
        self.element_of_grad_and_indices()
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name, inputs=[
                self.grad, self.indices], outputs=[
                self.grad_weight])
        return self.tik_instance

    def element_of_grad_and_indices(self):
        """
        Count the number of elements of indicators and grad

        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        self.numel_grad = 1
        self.numel_indices = 1
        for y in self.grad_shape:
            self.numel_grad *= y
        for x in self.indices_shape:
            self.numel_indices *= x
        self.new_numel_indices = math.ceil(
            self.numel_indices / self.dtype_bytes_size_indices) * self.dtype_bytes_size_indices

        if self.embedding_dim >= 8:
            self.gm_for_data_and_fill_grad_weight()
        else:
            self.gm_for_data_and_fill_grad_weight_low_dim()

    def gm_for_data_and_fill_grad_weight_low_dim(self):
        """
        Allocate space for grad, indices and grad_weight on gm
        use 0 to fill grad_weight in low dim
        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        self.indices = self.tik_instance.Tensor(self.dtype_indices, (self.numel_indices,), name="indices",
                                                scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor(self.dtype_grad, self.grad_shape, name="grad", scope=tik.scope_gm)
        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (self.num_weights, self.embedding_dim),
                                                    name="grad_weight", scope=tik.scope_gm)
        self.aicore_num = 1
        self.vector_max_repeat = 255
        # Create a new space to initialize grad_weight
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
            self.begin = index * self.num_weights // self.aicore_num
            with self.tik_instance.if_scope(index == self.aicore_num - 1):
                self.end = self.num_weights
            with self.tik_instance.else_scope():
                self.end = (index + 1) * self.num_weights // self.aicore_num
            with self.tik_instance.new_stmt_scope():
                # Initialize fill_tensor with 0, which is used to initialize grad_weight later
                fill_tensor = self.tik_instance.Tensor(self.dtype_grad, (1, FILL_DIM), name="tmp_tensor_grad",
                                                       scope=tik.scope_ubuf)
                # Define scalar_float0 to fill grad_weight
                fill_value = self.tik_instance.Scalar(dtype='int32')
                fill_value.set_as(((self.num_weights * self.embedding_dim - 1) // self.grad_each_block + 1)
                                  * self.grad_each_block)
                scalar_float0 = self.tik_instance.Scalar(init_value=0, dtype=self.dtype_grad)
                self.tik_instance.vec_dup(self.vector_mask_max_grad, fill_tensor, scalar_float0,
                                          FILL_DIM // self.vector_mask_max_grad, 8)
                # Use fill_tensor on ub to fill grad_weight on gm and do 0 initialization
                with self.tik_instance.if_scope(fill_value // FILL_DIM > 0):
                    with self.tik_instance.for_range(0, (fill_value - 1) // FILL_DIM + 1) as i:
                        self.tik_instance.data_move(self.grad_weight[i * FILL_DIM],
                                                    fill_tensor, 0, 1, FILL_DIM // self.grad_each_block, 0, 0)

            self.ub_for_data_fill_counts_low_dim()

    def gm_for_data_and_fill_grad_weight(self):
        """
        Allocate space for grad, indices and grad_weight on gm
        use 0 to fill grad_weight
        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        # Allocate space for grad, indices and grad_weight on gm
        self.indices = self.tik_instance.Tensor(self.dtype_indices, (self.new_numel_indices, ), name="indices",
                                                scope=tik.scope_gm)
        self.vector_max_repeat = 255
        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (self.num_weights, self.embedding_dim),
                                                    name="grad_weight", scope=tik.scope_gm)
        self.grad = self.tik_instance.Tensor(self.dtype_grad, self.grad_shape, name="grad", scope=tik.scope_gm)
        self.vec_max_grad_element = self.vector_max_repeat * self.vector_mask_max_grad
        # Create a new space to initialize grad_weight
        if self.num_weights // self.aicore_num == 0:
            self.aicore_num = 1
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index:
            self.begin = index * self.num_weights // self.aicore_num
            with self.tik_instance.if_scope(index == self.aicore_num - 1):
                self.end = self.num_weights
            with self.tik_instance.else_scope():
                self.end = (index + 1) * self.num_weights // self.aicore_num
            with self.tik_instance.new_stmt_scope():
                # Initialize fill_tensor with 0, which is used to initialize grad_weight later
                fill_tensor = self.tik_instance.Tensor(self.dtype_grad, (1, self.embedding_dim),
                                                       name="tmp_tensor_grad",
                                                       scope=tik.scope_ubuf)
                # Define scalar_float0 to fill grad_weight
                scalar_float0 = self.tik_instance.Scalar(init_value=0, dtype=self.dtype_grad)
                if self.embedding_dim // self.vector_mask_max_grad > 0:
                    self.tik_instance.vec_dup(self.vector_mask_max_grad, fill_tensor, scalar_float0,
                                              self.embedding_dim // self.vector_mask_max_grad, 8)
                if self.embedding_dim % self.vector_mask_max_grad != 0:
                    self.tik_instance.vec_dup(self.embedding_dim % self.vector_mask_max_grad, fill_tensor[
                        self.embedding_dim // self.vector_mask_max_grad * self.vector_mask_max_grad],
                                              scalar_float0, 1, 8)
                # Use fill_tensor on ub to fill grad_weight on gm and do 0 initialization
                with self.tik_instance.for_range(self.begin, self.end) as i:
                    self.tik_instance.data_move(self.grad_weight[i * self.embedding_dim],
                                                fill_tensor, 0, 1, self.embedding_dim // self.grad_each_block, 0, 0)

            self.ub_for_data_fill_counts()

    def ub_for_data_fill_counts_low_dim(self):
        """
        Allocate space for grad, indices and counts on ub
        use 0 to fill counts in low embedding_dim

        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, (self.ub_indices_size,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        if self.scale_grad_by_freq:
            self.counts_ub = self.tik_instance.Tensor(self.dtype_indices, (self.counts_size,), name="counts_ub",
                                                      scope=tik.scope_ubuf)
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_indices_size, self.embedding_dim),
                                                name="grad_ub", scope=tik.scope_ubuf)

        if self.scale_grad_by_freq:
            scalar_int0 = self.tik_instance.Scalar(init_value=0, dtype=self.dtype_indices)
            vec_max_count_element = self.vector_max_repeat * self.vector_mask_max_counts
            if self.counts_size // vec_max_count_element > 0:
                with self.tik_instance.for_range(0, self.counts_size // vec_max_count_element) as i:
                    self.tik_instance.vec_dup(self.vector_mask_max_counts, self.counts_ub[i * vec_max_count_element],
                                              scalar_int0, self.vector_max_repeat, 8)

            repeat_count_num = self.counts_size % vec_max_count_element
            if repeat_count_num // self.vector_mask_max_counts > 0 and repeat_count_num > 0:
                self.tik_instance.vec_dup(self.vector_mask_max_counts,
                                          self.counts_ub[self.counts_size // vec_max_count_element *
                                                         vec_max_count_element],
                                          scalar_int0, repeat_count_num // self.vector_mask_max_counts, 8)

            last_num_counts = self.counts_size - self.counts_size // vec_max_count_element * vec_max_count_element - \
                              self.counts_size % vec_max_count_element // self.vector_mask_max_counts * \
                              self.vector_mask_max_counts
            if last_num_counts > 0:
                offset_counts = self.counts_size // vec_max_count_element * vec_max_count_element + \
                                self.counts_size % vec_max_count_element // self.vector_mask_max_counts \
                                * self.vector_mask_max_counts
                self.tik_instance.vec_dup(last_num_counts, self.counts_ub[offset_counts], scalar_int0, 1, 8)
        self.base_count_words_compute_low_dim()

    def ub_for_data_fill_counts(self):
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
        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, (self.ub_indices_size, ), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        if self.scale_grad_by_freq:
            self.counts_ub = self.tik_instance.Tensor(self.dtype_indices, (self.counts_size,), name="counts_ub",
                                                      scope=tik.scope_ubuf)
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_indices_size, self.embedding_dim),
                                                name="grad_ub", scope=tik.scope_ubuf)

        if self.scale_grad_by_freq:
            scalar_int0 = self.tik_instance.Scalar(init_value=0, dtype=self.dtype_indices)
            vec_max_count_element = self.vector_max_repeat * self.vector_mask_max_counts
            if self.counts_size // vec_max_count_element > 0:
                with self.tik_instance.for_range(0, self.counts_size // vec_max_count_element) as i:
                    self.tik_instance.vec_dup(self.vector_mask_max_counts, self.counts_ub[i * vec_max_count_element],
                                              scalar_int0, self.vector_max_repeat, 8)

            repeat_count_num = self.counts_size % vec_max_count_element
            if repeat_count_num // self.vector_mask_max_counts > 0 and repeat_count_num > 0:
                self.tik_instance.vec_dup(self.vector_mask_max_counts,
                                          self.counts_ub[self.counts_size // vec_max_count_element *
                                                         vec_max_count_element],
                                          scalar_int0, repeat_count_num // self.vector_mask_max_counts, 8)

            last_num_counts = self.counts_size - self.counts_size // vec_max_count_element * vec_max_count_element - \
                self.counts_size % vec_max_count_element // self.vector_mask_max_counts * self.vector_mask_max_counts
            if last_num_counts > 0:
                offset_counts = self.counts_size // vec_max_count_element * vec_max_count_element + \
                    self.counts_size % vec_max_count_element // self.vector_mask_max_counts \
                    * self.vector_mask_max_counts
                self.tik_instance.vec_dup(last_num_counts, self.counts_ub[offset_counts], scalar_int0, 1, 8)
        self.base_count_words_compute()

    def base_count_words_compute_low_dim(self):
        """
        when sf is True,use base function to count words
        in low embedding_dim

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
            with self.tik_instance.if_scope(self.numel_indices // self.ub_indices_size > 0):
                with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
                    self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                                self.ub_indices_size // self.indices_each_block, 0, 0)
                    # Use self.counts_ub to count word frequency
                    self.count_words_compute(self.ub_indices_size)
        self.remaining_count_words_compute_low_dim()

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

    def remaining_count_words_compute_low_dim(self):
        """
        in low embedding_dim
        when sf is True,use remaining function to count words

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if self.scale_grad_by_freq:
            if self.numel_indices % self.ub_indices_size != 0:
                offset_indices_move = self.numel_indices // self.ub_indices_size * self.ub_indices_size
                burst_len_indices = math.ceil(self.numel_indices % self.ub_indices_size / self.indices_each_block)
                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices,
                                            0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.numel_indices % self.ub_indices_size)
        self.base_compute_grad_weight_low_dim()

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
            if self.numel_indices % self.ub_indices_size != 0:
                offset_indices_move = self.numel_indices // self.ub_indices_size * self.ub_indices_size
                burst_len_indices = math.ceil(self.numel_indices % self.ub_indices_size / self.indices_each_block)
                self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices,
                                            0, 0)
                # Use self.counts_ub to count word frequency
                self.count_words_compute(self.numel_indices % self.ub_indices_size)
        self.base_compute_grad_weight()

    def base_compute_grad_weight_low_dim(self):
        """
        in low embedding_dim
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
       ----------
        None
        Returns
        -------
        None
        """
        self.scale_float = self.tik_instance.Scalar(
            init_value=1.0, dtype=self.dtype_grad)
        scalar_float0 = self.tik_instance.Scalar(
            init_value=0.0, dtype=self.dtype_grad)
        self.add_tensor = self.tik_instance.Tensor(
            self.dtype_grad, (1, BLOCK), name="add_tensor", scope=tik.scope_ubuf)
        self.scale_int = self.tik_instance.Scalar(dtype=self.dtype_indices)
        self.tik_instance.vec_dup(BLOCK, self.add_tensor, scalar_float0, 1, 8)
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
        # Move indexes and grad blocks from gm to ub
        with self.tik_instance.if_scope(self.numel_indices // self.ub_indices_size > 0):
            with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
                self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                self.tik_instance.data_move(self.grad_ub, self.grad[i1 * self.ub_indices_size * self.embedding_dim],
                                            0, 1, self.ub_indices_size * self.embedding_dim //
                                            self.grad_each_block, 0, 0)

                self.add_same_word_grad_low_dim(self.ub_indices_size)
        self.remaining_compute_grad_weight_low_dim()

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
        self.scale_int = self.tik_instance.Scalar(dtype=self.dtype_indices)
        self.add_tensor = self.tik_instance.Tensor(
            self.dtype_grad, (1, self.embedding_dim), name="add_tensor", scope=tik.scope_ubuf)
        self.scale_float = self.tik_instance.Scalar(
            init_value=1.0, dtype=self.dtype_grad)
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
        # Move indexes and grad blocks from gm to ub
        with self.tik_instance.for_range(0, self.numel_indices // self.ub_indices_size) as i1:
            self.tik_instance.data_move(self.indices_ub, self.indices[i1 * self.ub_indices_size], 0, 1,
                                        self.ub_indices_size // self.indices_each_block, 0, 0)
            self.tik_instance.data_move(self.grad_ub, self.grad[i1 * self.ub_indices_size * self.embedding_dim],
                                        0, 1, self.ub_indices_size * self.embedding_dim // self.grad_each_block, 0, 0)

            self.add_same_word_grad(self.ub_indices_size)
        self.remaining_compute_grad_weight()

    def remaining_compute_grad_weight_low_dim(self):
        """
        in low embedding_dim
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        if self.numel_indices % self.ub_indices_size != 0:
            offset_indices_move = self.numel_indices // self.ub_indices_size * self.ub_indices_size
            burst_len_indices = math.ceil(
                self.numel_indices %
                self.ub_indices_size /
                self.indices_each_block)
            offset_grad_move = self.numel_indices // self.ub_indices_size *\
                               self.ub_indices_size * self.embedding_dim
            burst_len_grad = (self.numel_indices % self.ub_indices_size *
                              self.embedding_dim - 1) // self.grad_each_block + 1

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad_low_dim(self.numel_indices % self.ub_indices_size)

    def remaining_compute_grad_weight(self):
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
        if self.numel_indices % self.ub_indices_size != 0:
            offset_indices_move = self.numel_indices // self.ub_indices_size * self.ub_indices_size
            burst_len_indices = math.ceil(
                self.numel_indices %
                self.ub_indices_size /
                self.indices_each_block)
            offset_grad_move = self.numel_indices // self.ub_indices_size * \
                self.ub_indices_size * self.embedding_dim
            burst_len_grad = self.numel_indices % self.ub_indices_size * \
                self.embedding_dim // self.grad_each_block

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad(self.numel_indices % self.ub_indices_size)

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

    def add_same_word_grad_low_dim(self, total):
        """
        when sf is False,use this function to compute grad_weight
        by add and scale=1 in low dim
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
                                                BLOCK // self.grad_each_block, 0, 0)
                    self.cac_result_low_dim()

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

    def cac_result_low_dim(self):
        """
        caculate the rersult of grad_weight in low dim

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        scalar_float0 = self.tik_instance.Scalar(dtype=self.dtype_grad, init_value=0)
        if self.scale_grad_by_freq:
            self.scale_int.set_as(self.counts_ub[self.k - self.begin])
            self.tik_instance.scalar_conv('', self.scale_float, self.scale_int)
            self.scale_float.set_as(1.0 / self.scale_float)
        block_tensor = self.tik_instance.Tensor(self.dtype_grad, (1, BLOCK), name='block_tensor',
                                                scope=tik.scope_ubuf)
        with self.tik_instance.for_range(self.embedding_dim * self.index,
                                         self.embedding_dim * self.index + self.embedding_dim) as i:
            block_tensor[i - self.embedding_dim * self.index].set_as(self.grad_ub[i])
        with self.tik_instance.for_range(self.embedding_dim, BLOCK) as j:
            block_tensor[j].set_as(scalar_float0)
        self.tik_instance.vec_axpy(BLOCK,
                                   self.add_tensor,
                                   block_tensor,
                                   self.scale_float, 1, 8, 8)
        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim], self.add_tensor, 0, 1,
                                    BLOCK // self.grad_each_block, 0, 0)

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
        if self.embedding_dim // self.vector_mask_max_grad > 0:
            self.tik_instance.vec_axpy(self.vector_mask_max_grad, self.add_tensor,
                                       self.grad_ub[self.index * self.embedding_dim], self.scale_float,
                                       self.embedding_dim // self.vector_mask_max_grad, 8, 8)
        if self.embedding_dim % self.vector_mask_max_grad > 0:
            self.tik_instance.vec_axpy(self.embedding_dim % self.vector_mask_max_grad,
                                       self.add_tensor[self.embedding_dim // self.vector_mask_max_grad *
                                                       self.vector_mask_max_grad],
                                       self.grad_ub[self.index * self.embedding_dim + self.embedding_dim //
                                                    self.vector_mask_max_grad * self.vector_mask_max_grad],
                                       self.scale_float, 1, 8, 8)
        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim], self.add_tensor, 0, 1,
                                    self.embedding_dim // self.grad_each_block, 0, 0)


@op_utils.check_op_params(
    op_utils.REQUIRED_INPUT,
    op_utils.REQUIRED_INPUT,
    op_utils.REQUIRED_OUTPUT,
    op_utils.OPTION_ATTR_INT,
    op_utils.OPTION_ATTR_INT,
    op_utils.OPTION_ATTR_BOOL,
    op_utils.KERNEL_NAME)
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
    check_attr_param(num_weights)
    check_same_dim(grad, indices)
    check_grad_param(grad)
    check_indices_param(indices)
    check_max_align(num_weights, padding_idx)
    embedding_dense_grad_instance = EmbeddingDenseGrad(
        grad, indices, y, num_weights, padding_idx, scale_grad_by_freq, kernel_name)
    tik_instance = embedding_dense_grad_instance.embedding_dense_grad_compute()
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
    grad_shape = grad_dic.get("shape")
    op_utils.check_shape(grad_shape)
    op_utils.check_dtype(grad_dtype, ["float32"])


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
    indices_shape = indices_dic.get("shape")
    op_utils.check_shape(indices_shape)
    op_utils.check_dtype(indices_dtype, ["int32"])


def check_attr_param(n_w):
    """
    check the parameters num_weights is valid

    Parameters
    ----------
    n_w: the number of words in dict
    Returns
    -------
    None
    """
    if n_w <= 0:
        raise RuntimeError('num_weights must be greater than 0')


def check_same_dim(grad_dic, indices_dic):
    """
    check if grad_shape[:-1] same as indices_shape

    Parameters
    ----------
    grad_dic: dict,shape and datatype,
    datatype supports float32
    indices_dic: dict,shape and datatype,
    datatype supports int32
    Returns
    -------
    None
    """
    grad_shape = grad_dic.get("shape")
    indices_shape = indices_dic.get("shape")
    if grad_shape[:-1] != indices_shape:
        raise RuntimeError('shape_indices and shape_grad[:-1] must be same')


def check_max_align(n_w, p_i):
    """
    check the max_shape of grad and indices,
    judge embedding_dim whether an integer multiple of 32 or not
    judge the padding_index whether less than num_weights or not

    Parameters
    ----------
    n_w:the number of words in dict
    p_i:judge grad_weight of which word is zero
    Returns
    -------
    None
    """
    if n_w - 1 < p_i:
        raise RuntimeError('padding_index must be less than num_weights!')

