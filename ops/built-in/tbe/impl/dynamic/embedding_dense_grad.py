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

import math
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform as cce
from impl.util.platform_adapter import tbe_context


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    RESERVE_SIZE = 16 * 1024
    BLOCK = 8
    MAX_INT32 = 2 ** 31 - 1
    SCALAR_TENSOR_SIZE = 32
    TILING_ARG_NUM = 64
    TILING_MODE_1 = 1
    GRAD_TENSOR_PART = 512
    TOTAL_PART = 513


class EmbeddingDenseGrad:
    """
    Function: store EmbeddingDenseGrad parameters  and compute EmbeddingDenseGrad
    """

    # 'pylint: disable=unused-argument, too-many-statements, disable=too-many-arguments
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
        self.dtype_grad = grad.get("dtype")
        self.dtype_indices = indices.get("dtype")
        self.embedding_dim = None
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.padding_idx = padding_idx
        self.num_weights = num_weights
        self.kernel_name = kernel_name
        self.scale_grad_by_freq = scale_grad_by_freq
        self.block_ub = None
        self.align_ub = None

        '''Data reading and writing on UB must be 32B aligned. This parameter is used
        to calculate tensor division and data handling instruction parameters
        '''
        block_bite_size = 32
        # Get the size of UB space in Bytes
        self.ub_size_bytes = cce.get_soc_spec(cce.UB_SIZE)
        self.aicore_num = cce.get_soc_spec(cce.CORE_NUM)

        '''Calculate how many grad elements can be stored in a block according to
        the input data type
        '''
        self.dtype_bytes_size_grad = cce.get_bit_len(self.dtype_grad) // Constant.BLOCK
        self.grad_each_block = block_bite_size // self.dtype_bytes_size_grad
        '''Calculate how many counts elements can be stored in a block according
        to the input data type
        '''
        self.dtype_bytes_size_counts = cce.get_bit_len(self.dtype_indices) // Constant.BLOCK
        self.counts_each_block = block_bite_size // self.dtype_bytes_size_counts

        '''Calculate how many indicators elements can be stored in a block
        according to the input data type
        '''
        self.dtype_bytes_size_indices = cce.get_bit_len(self.dtype_indices) // Constant.BLOCK
        self.indices_each_block = block_bite_size // self.dtype_bytes_size_indices

        '''The vector instruction calculates a maximum of 8 blocks per repeat.
        This parameter is the maximum value of the mask when grad performs vector calculation
        '''
        self.vector_mask_max_counts = Constant.BLOCK * self.counts_each_block
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
        self.vector_mask_max_grad = Constant.BLOCK * self.grad_each_block
        self.ele_not_last_core = self.tik_instance.Scalar(self.dtype_indices, name='ele_not_last_core')
        self.ele_last_core = self.tik_instance.Scalar(self.dtype_indices, name='ele_last_core')
        self.ranges = self.tik_instance.Scalar("int32", name='ranges')
        self.index_not_last_core = None
        self.ub_grad_size = self.tik_instance.Scalar(self.dtype_indices, name='ub_grad_size')
        self.end = None
        self.scale_int = None
        self.ub_indices_size = self.tik_instance.Scalar(self.dtype_indices, name='ub_indices_size')
        self.counts_size = None
        self.tiling_dtype = 'int32'
        self.dtype_bytes_size_tiling = cce.get_bit_len(self.tiling_dtype) // 8
        self.tiling_each_block = block_bite_size // self.dtype_bytes_size_tiling
        self.core_used = self.tik_instance.Scalar(self.dtype_indices, name='core_used')
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                                  name='tiling_gm', scope=tik.scope_gm)

    def get_tiling_args(self):
        """
        get tiling args from tling_ub

        Parameters
        ----------
        tiling_ub: s tensor with tiling_args in ub

        Returns
        -------
        None
        """
        tiling_ub = self.tik_instance.Tensor(self.tiling_dtype, (Constant.TILING_ARG_NUM,),
                                             name='tiling_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.SCALAR_TENSOR_SIZE //
                                    self.tiling_each_block, 0, 0)
        self.numel_indices = self.tik_instance.Scalar(self.dtype_indices, name='numel_indices')
        self.embedding_dim = self.tik_instance.Scalar(self.dtype_indices, name='embedding_dim')

        self.numel_indices.set_as(tiling_ub[0])
        self.embedding_dim.set_as(tiling_ub[1])

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
        self.get_tiling_args()
        self.gm_for_data_and_fill_grad_weight()
        with self.tik_instance.if_scope(self.numel_indices // self.core_used * self.embedding_dim < 8):
            self.core_used.set_as(1)
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_used):
                if self.scale_grad_by_freq:
                    self.begin = core_index * self.index_not_last_core
                    with self.tik_instance.if_scope(core_index == self.aicore_num - 1):
                        self.end = self.num_weights
                    with self.tik_instance.else_scope():
                        self.end = (core_index + 1) * self.index_not_last_core
                self.ub_for_data(core_index)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.grad, self.indices],
                                   outputs=[self.grad_weight], flowtable=[self.tiling_gm])
        tbe_context.get_context().add_compile_info('vars', {'core_num': self.aicore_num,
                                                    'num_weights': self.num_weights,
                                                    'padding_idx': self.padding_idx,
                                                    'scale_grad_by_freq': self.scale_grad_by_freq})
        return self.tik_instance

    def cal_ub_size(self):
        """
        cal_ub_size
        """
        if self.scale_grad_by_freq:
            self.index_not_last_core = (self.num_weights - 1) // self.aicore_num + 1
            core_used = (self.num_weights - 1) // self.index_not_last_core + 1
            self.core_used.set_as(core_used)
            self.counts_size = self.num_weights // core_used + self.num_weights % core_used
            self.ub_indices_size.set_as((self.ub_size_bytes - self.counts_size *
                                    self.dtype_bytes_size_counts - Constant.RESERVE_SIZE) \
                                   // (self.embedding_dim * self.dtype_bytes_size_grad +
                                       self.dtype_bytes_size_indices) \
                                   // self.indices_each_block * self.indices_each_block)
        else:
            self.ele_not_last_core.set_as((self.numel_indices - 1) // self.aicore_num + 1)
            self.core_used.set_as((self.numel_indices - 1) // self.ele_not_last_core + 1)
            self.ele_last_core.set_as(self.numel_indices - (self.core_used - 1) * self.ele_not_last_core)
            self.ub_indices_size.set_as((self.ub_size_bytes - Constant.RESERVE_SIZE) \
                                       // (self.embedding_dim * self.dtype_bytes_size_grad +
                                           self.dtype_bytes_size_indices) \
                                       // self.indices_each_block * self.indices_each_block)
            self.ub_grad_size.set_as(self.ub_indices_size * self.embedding_dim)

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
        self.indices = self.tik_instance.Tensor(self.dtype_indices, (Constant.MAX_INT32,), name="indices",
                                                scope=tik.scope_gm)
        self.grad_weight = self.tik_instance.Tensor(self.dtype_grad, (Constant.MAX_INT32,),
                                                    name="grad_weight", scope=tik.scope_gm, is_atomic_add=True)
        self.grad = self.tik_instance.Tensor(self.dtype_grad, (Constant.MAX_INT32,), name="grad", scope=tik.scope_gm)
        self.cal_ub_size()

    @staticmethod
    def get_dtype_size(dtype):
        """
        get byte size of dtype
        """
        dtype_dict = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2}
        return dtype_dict.get(dtype)

    def dup_value(self, dst, num, dup_value=0, offset=0):
        """
        dup value to ub
        """
        dtype_byte_size = EmbeddingDenseGrad.get_dtype_size(dst.dtype)
        mask = 256 // dtype_byte_size
        stride = 8
        loop = num // (mask * 255)
        with self.tik_instance.if_scope(loop > 0):
            with self.tik_instance.for_range(0, loop) as index:
                tmp_offset = offset + index * mask * 255
                self.tik_instance.vec_dup(mask, dst[tmp_offset], dup_value, 255, stride)
            offset += loop * mask * 255

        repeat_time = (num % (mask * 255)) // mask
        with self.tik_instance.if_scope(repeat_time > 0):
            self.tik_instance.vec_dup(mask, dst[offset], dup_value, repeat_time, stride)
            offset += repeat_time * mask
        last_num = num % mask
        with self.tik_instance.if_scope(last_num > 0):
            self.tik_instance.vec_dup(last_num, dst[offset], dup_value, 1, stride)

    def ub_for_data(self, core_index):
        """
        Allocate space for grad, indices and counts on ub
        use 0 to fill counts

        Parameters
       ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        # Allocate space for grad, indices and counts on ub
        self.indices_ub = self.tik_instance.Tensor(self.dtype_indices, (self.ub_indices_size,), name="indices_ub",
                                                   scope=tik.scope_ubuf)
        if self.scale_grad_by_freq:
            self.counts_ub = self.tik_instance.Tensor(self.dtype_indices, (self.counts_size,), name="counts_ub",
                                                      scope=tik.scope_ubuf)
            self.dup_value(self.counts_ub, self.counts_size)
        self.grad_ub = self.tik_instance.Tensor(self.dtype_grad, (self.ub_grad_size,),
                                                name="grad_ub", scope=tik.scope_ubuf)
        self.base_count_words_compute(core_index)

    def base_count_words_compute(self, core_index):
        """
        when sf is True,use base function to count words

        Parameters
       ----------
        core_index: the index of aicore
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
        self.remaining_count_words_compute(core_index)

    def remaining_count_words_compute(self, core_index):
        """
        when sf is True,use remaining function to count words

        Parameters
        ----------
        core_index: the index of aicore
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
            self.base_compute_grad_weight_need_scale()
        else:
            self.base_compute_grad_weight_not_need_scale(core_index)

    def base_compute_grad_weight_need_scale(self):
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
            self.dtype_grad, (1, self.embedding_dim), name="add_tensor", scope=tik.scope_ubuf)
        self.scale_int = self.tik_instance.Scalar(dtype=self.dtype_indices)
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

            self.add_same_word_grad_need_scale(self.ub_indices_size)
        self.remaining_compute_grad_weight_need_scale()

    def add_same_word_grad_need_scale(self, total):
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

    def remaining_compute_grad_weight_need_scale(self):
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
        with self.tik_instance.if_scope(self.numel_indices % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar("int32", name='offset_indices_move')
            burst_len_indices = self.tik_instance.Scalar("int32", name='burst_len_indices')
            offset_grad_move = self.tik_instance.Scalar("int32", name='offset_grad_move')
            burst_len_grad = self.tik_instance.Scalar("int32", name='burst_len_grad')
            offset_indices_move.set_as(self.numel_indices // self.ub_indices_size * self.ub_indices_size)
            burst_len_indices.set_as((self.numel_indices % self.ub_indices_size - 1) // self.indices_each_block + 1)
            offset_grad_move.set_as(self.numel_indices // self.ub_indices_size * self.ub_indices_size
                                    * self.embedding_dim)
            burst_len_grad.set_as(self.numel_indices % self.ub_indices_size *
                                  self.embedding_dim // self.grad_each_block)

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad_need_scale(self.numel_indices % self.ub_indices_size)

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
        self.scale_int.set_as(self.counts_ub[self.k - self.begin])
        self.tik_instance.scalar_conv('', self.scale_float, self.scale_int)
        numerator = 1.0
        self.scale_float.set_as(numerator / self.scale_float)
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

    def add_grad_no_scale_not_align(self, total):
        """
        add_grad_no_scale_not_align
        """
        align_size = ((self.embedding_dim - 1) // Constant.BLOCK + 1) * Constant.BLOCK
        align_ub = self.tik_instance.Tensor(self.dtype_grad, (align_size,), name='align_ub', scope=tik.scope_ubuf)
        self.dup_value(align_ub, align_size)
        with self.tik_instance.for_range(0, total) as index:
            self.k.set_as(self.indices_ub[index])
            with self.tik_instance.if_scope(self.k != self.padding_idx):
                with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                    self.tik_instance.data_move(align_ub, self.grad_ub[index * align_size], 0, 1,
                                                self.embedding_dim // self.grad_each_block, 0, 0)
                    with self.tik_instance.for_range(0, self.embedding_dim % self.grad_each_block) as i:
                        align_ub[self.embedding_dim // self.grad_each_block *
                                 self.grad_each_block + i].set_as(self.grad_ub[index * align_size +
                                                                               self.embedding_dim //
                                                                               self.grad_each_block *
                                                                               self.grad_each_block + i])
                    self.tik_instance.set_atomic_add(1)
                    self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                align_ub, 0,
                                                1, align_size // self.grad_each_block, 0, 0)
                    self.tik_instance.set_atomic_add(0)

    def base_compute_grad_weight_not_need_scale(self, core_index):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
       ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        # Define k, the scalar used to index the elements of indicators
        self.k = self.tik_instance.Scalar(dtype=self.dtype_indices)
        scalar_float0 = self.tik_instance.Scalar(dtype=self.dtype_grad, init_value=0)
        self.block_ub = self.tik_instance.Tensor(self.dtype_grad, (Constant.BLOCK,), name="block_ub",
                                                 scope=tik.scope_ubuf)
        self.tik_instance.vec_dup(Constant.BLOCK, self.block_ub, scalar_float0, 1, 8)
        with self.tik_instance.if_scope(core_index == self.core_used - 1):
            self.ranges.set_as(self.ele_last_core)
        with self.tik_instance.else_scope():
            self.ranges.set_as(self.ele_not_last_core)
        # Move indexes and grad blocks from gm to ub
        with self.tik_instance.if_scope(tik.all(self.embedding_dim > Constant.BLOCK,
                                         self.embedding_dim % Constant.BLOCK != 0)):
            self.base_compute_no_scale_not_align(core_index)
        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
                with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i1:
                    self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                              i1 * self.ub_indices_size], 0, 1,
                                                self.ub_indices_size // self.indices_each_block, 0, 0)
                    self.tik_instance.data_move(self.grad_ub,
                                                self.grad[(core_index * self.ele_not_last_core + i1 *
                                                           self.ub_indices_size) * self.embedding_dim],
                                                0, 1, self.ub_indices_size * self.embedding_dim // self.grad_each_block,
                                                0, 0)
                    self.add_same_word_grad_not_need_scale(self.ub_indices_size)
            self.remaining_compute_grad_weight_not_need_scale(core_index)

    def base_compute_no_scale_not_align(self, core_index):
        """
        base_compute_no_scale_not_align
        """
        align_length = (self.embedding_dim - 1) // Constant.BLOCK + 1
        with self.tik_instance.if_scope(self.ranges // self.ub_indices_size > 0):
            with self.tik_instance.for_range(0, self.ranges // self.ub_indices_size) as i:
                self.tik_instance.data_move(self.indices_ub, self.indices[core_index * self.ele_not_last_core +
                                                                          i * self.ub_indices_size], 0, 1,
                                            self.ub_indices_size // self.indices_each_block, 0, 0)
                with self.tik_instance.for_range(0, self.ub_indices_size) as j:
                    self.tik_instance.data_move(self.grad_ub[j * align_length * Constant.BLOCK],
                                                self.grad[(core_index * self.ele_not_last_core + i *
                                                           self.ub_indices_size + j) * self.embedding_dim],
                                                0, 1, align_length, 0, 0)
                self.add_same_word_grad_not_need_scale(self.ub_indices_size)
        self.remain_compute_no_scale_not_align(core_index, align_length)

    def remain_compute_no_scale_not_align(self, core_index, align_size):
        """
        remain_compute_no_scale_not_align
        """
        with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
            offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                        * self.ub_indices_size)
            burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
            offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
            burst_len_indices.set_as((
                self.ranges %
                self.ub_indices_size - 1) //
                self.indices_each_block + 1)
            offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                self.ub_indices_size) * self.embedding_dim)

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            with self.tik_instance.for_range(0, self.ranges % self.ub_indices_size) as i:
                self.tik_instance.data_move(self.grad_ub[i * align_size * Constant.BLOCK],
                                            self.grad[(core_index * self.ele_not_last_core + self.ranges //
                                                       self.ub_indices_size * self.ub_indices_size + i)
                                                      * self.embedding_dim],
                                            0, 1, align_size, 0, 0)
            self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)

    def remaining_compute_grad_weight_not_need_scale(self, core_index):
        """
        when sf is False,just use base function to compute the grad_weight
        when sf is True,use base function to compute the grad_weight by scale

        Parameters
        ----------
        core_index: the index of aicore
        Returns
        -------
        None
        """
        with self.tik_instance.if_scope(self.ranges % self.ub_indices_size != 0):
            offset_indices_move = self.tik_instance.Scalar('int32', name='offset_indices_move')
            offset_indices_move.set_as(core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size \
                                                                        * self.ub_indices_size)
            burst_len_indices = self.tik_instance.Scalar('int32', name='burst_len_indices')
            burst_len_grad = self.tik_instance.Scalar('int32', name='burst_len_grad')
            offset_grad_move = self.tik_instance.Scalar('int32', name='offset_grad_move')
            burst_len_indices.set_as((
                self.ranges %
                self.ub_indices_size - 1) //
                self.indices_each_block + 1)
            offset_grad_move.set_as((core_index * self.ele_not_last_core + self.ranges // self.ub_indices_size *
                                self.ub_indices_size) * self.embedding_dim)
            burst_len_grad.set_as(((self.ranges % self.ub_indices_size) * self.embedding_dim - 1)
                                  // self.grad_each_block + 1)

            self.tik_instance.data_move(self.indices_ub, self.indices[offset_indices_move], 0, 1, burst_len_indices, 0,
                                        0)
            self.tik_instance.data_move(self.grad_ub, self.grad[offset_grad_move], 0, 1, burst_len_grad, 0, 0)
            self.add_same_word_grad_not_need_scale(self.ranges % self.ub_indices_size)

    def add_same_word_grad_not_need_scale(self, total):
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
        with self.tik_instance.if_scope(tik.all(self.embedding_dim > Constant.BLOCK,
                                         self.embedding_dim % Constant.BLOCK != 0)):
            self.add_grad_no_scale_not_align(total)
        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, total) as self.index:
                self.k.set_as(self.indices_ub[self.index])
                with self.tik_instance.if_scope(self.k != self.padding_idx):
                    with self.tik_instance.if_scope(tik.all(self.k >= 0, self.k < self.num_weights)):
                        with self.tik_instance.if_scope(self.embedding_dim < Constant.BLOCK):
                            with self.tik_instance.for_range(0, self.embedding_dim) as i:
                                self.block_ub[i].set_as(self.grad_ub[self.index * self.embedding_dim + i])
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                        self.block_ub, 0,
                                                        1, Constant.BLOCK // self.grad_each_block, 0, 0)
                            self.tik_instance.set_atomic_add(0)
                        with self.tik_instance.else_scope():
                            self.tik_instance.set_atomic_add(1)
                            self.tik_instance.data_move(self.grad_weight[self.k * self.embedding_dim],
                                                        self.grad_ub[self.index * self.embedding_dim], 0,
                                                        1, self.embedding_dim // self.grad_each_block, 0, 0)
                            self.tik_instance.set_atomic_add(0)


# 'pylint: disable=too-many-arguments
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
    tik_instance = embedding_dense_grad_instance.embedding_dense_grad_compute_tiling()
    return tik_instance

