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

import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce

RESERVE_SIZE = 16 * 1024
BLOCK = 8


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
        self.block_ub = None

        '''Data reading and writing on UB must be 32B aligned. This parameter is used
        to calculate tensor division and data handling instruction parameters
        '''
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.aicore_num = 32

        '''Calculate how many grad elements can be stored in a block according to
        the input data type
        '''
        self.dtype_bytes_size_grad = cce.get_bit_len(
            self.dtype_grad) // 8
        self.grad_each_block = block_bite_size // self.dtype_bytes_size_grad
        '''Calculate how many counts elements can be stored in a block according 
        to the input data type
        '''
        self.dtype_bytes_size_counts = cce.get_bit_len(
            self.dtype_indices) // 8
        self.counts_each_block = block_bite_size // self.dtype_bytes_size_counts

        '''Calculate how many indicators elements can be stored in a block
        according to the input data type
        '''
        self.dtype_bytes_size_indices = cce.get_bit_len(
            self.dtype_indices) // 8
        self.indices_each_block = block_bite_size // self.dtype_bytes_size_indices

        if self.scale_grad_by_freq:
            self.ub_indices_size = (self.ub_size_bytes - self.num_weights *
                                    self.dtype_bytes_size_counts - RESERVE_SIZE) \
                                   // (self.embedding_dim * self.dtype_bytes_size_grad +
                                       self.dtype_bytes_size_indices)\
                                   // self.indices_each_block * self.indices_each_block
            self.counts_size = self.num_weights
        else:
            self.ub_indices_size = (self.ub_size_bytes - RESERVE_SIZE) \
                                   // (self.embedding_dim * self.dtype_bytes_size_grad +
                                       self.dtype_bytes_size_indices)\
                                   // self.indices_each_block * self.indices_each_block
        self.ub_grad_size = self.ub_indices_size * self.embedding_dim

        '''The vector instruction calculates a maximum of 8 blocks per repeat.
        This parameter is the maximum value of the mask when grad performs vector calculation
        '''
        self.vector_mask_max_counts = 8 * self.counts_each_block
        self.new_numel_indices = None
        self.grad_ub = None
        self.indices_ub = None
        self.grad = None
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
        self.ranges = None
        self.ele_not_last_core = None
        self.ele_last_core = None
        self.used_core = None

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
        if self.numel_indices // self.aicore_num * self.embedding_dim < BLOCK:
            self.aicore_num = 1
        self.gm_for_data_and_fill_grad_weight()

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
        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (self.num_weights, self.embedding_dim),
                                                    name="grad_weight", scope=tik.scope_gm, is_atomic_add=True)
        self.grad = self.tik_instance.Tensor(self.dtype_grad, self.grad_shape, name="grad", scope=tik.scope_gm)
        # Create a new space to initialize grad_weight
        self.ele_not_last_core = (self.numel_indices - 1) // self.aicore_num + 1
        self.used_core = (self.numel_indices - 1) //self.ele_not_last_core + 1
        self.ele_last_core = self.numel_indices - (self.used_core - 1) * self.ele_not_last_core
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_index:
            with self.tik_instance.if_scope(core_index < self.used_core):
                self.ub_for_data(core_index)

    def ub_for_data(self, core_index):
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
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_indices_size, self.embedding_dim),
                                                name="grad_ub", scope=tik.scope_ubuf)
        self.base_compute_grad_weight(core_index)

    def base_compute_grad_weight(self, core_index):
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
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
        scalar_float0 = self.tik_instance.Scalar(dtype=self.dtype_grad, init_value=0)
        if core_index == self.used_core - 1:
            self.ranges = self.ele_last_core
        else:
            self.ranges = self.ele_not_last_core
        # Move indexes and grad blocks from gm to ub
        if self.embedding_dim < 8:
            self.block_ub = self.tik_instance.Tensor(self.dtype_grad, (BLOCK, ), name="block_ub",
                                                     scope=tik.scope_ubuf)
            self.tik_instance.vec_dup(BLOCK, self.block_ub, scalar_float0, 1, 8)
        if self.ranges // self.ub_indices_size > 0:
            with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i1:
                self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                          i1 * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                self.tik_instance.data_move(self.grad_ub, self.grad[(core_index * self.ele_not_last_core +
                                                                    i1 * self.ub_indices_size) * self.embedding_dim],
                                            0, 1, self.ub_indices_size * self.embedding_dim // self.grad_each_block,
                                            0, 0)

                self.add_same_word_grad(self.ub_indices_size)
        self.remaining_compute_grad_weight(core_index)

    def remaining_compute_grad_weight(self, core_index):
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
        if self.ranges % self.ub_indices_size != 0:
            offset_indices_move = core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                        * self.ub_indices_size
            burst_len_indices = math.ceil(
                self.ranges %
                self.ub_indices_size /
                self.indices_each_block)
            offset_grad_move = (core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                self.ub_indices_size) * self.embedding_dim
            burst_len_grad = ((self.ranges % self.ub_indices_size) * self.embedding_dim - 1) // self.grad_each_block + 1

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad(self.ranges % self.ub_indices_size)

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
                with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                    self.tik_instance.set_atomic_add(1)
                    if self.embedding_dim < 8:
                        with self.tik_instance.for_range(0, self.embedding_dim) as i:
                            self.block_ub[i].set_as(self.grad_ub[self.index * self.embedding_dim + i])
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                self.block_ub, 0,
                                                1, BLOCK // self.grad_each_block, 0, 0)
                    else:
                        self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                    self.grad_ub[self.index * self.embedding_dim], 0,
                                                    1, self.embedding_dim // self.grad_each_block, 0, 0)
                    self.tik_instance.set_atomic_add(0)


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
    embedding_dense_grad_instance = EmbeddingDenseGrad(
        grad, indices, y, num_weights, padding_idx, scale_grad_by_freq, kernel_name)
    tik_instance = embedding_dense_grad_instance.embedding_dense_grad_compute()
    return tik_instance

