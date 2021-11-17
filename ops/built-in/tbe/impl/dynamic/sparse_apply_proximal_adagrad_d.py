# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
sparse_apply_proximal_adagrad_d
"""
# 'pylint: disable=unused-argument
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    DTYPE_INT32 = "int32"
    TILING_PARAM_DTYPE = DTYPE_INT32

    SELECT_KEY_LAST_AXIS_80_FLOAT = 1

    # fp32 byte
    BYTE_FP32 = 4
    # int32 byte
    BYTE_INT32 = 4
    # byte of one block
    BYTE_BLOCK = 32
    # ele num fp32 in one block
    FP32_ELE_NUM_BLOCK = 8
    # tiling params num
    TILING_PARAMS_NUM = 32
    # full mask for fp32
    MASK_FP32 = 64

    MAX_INT32 = 2 ** 31 - 1
    MAX_GM_TENSOR_SHAPE = (MAX_INT32,)
    MIN_TENSOR_SHAPE = (32,)
    ONE_FP32 = 1.0
    ZERO_FP32 = 0.0
    NEG_ONE_FP32 = -1.0


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments,invalid-name,too-few-public-methods
# 'pylint: disable=too-many-statements,too-many-instance-attributes,
def ceil(num, factor):
    """
    compute ceil

    Parameters
    ----------
    num: num
    factor: factor

    Returns
    -------
    ceil num by factor
    """
    res = 0
    if num % factor != 0:
        res = (num / factor + 1) * factor
    else:
        res = num
    return res


class SparseApplyProximalAdagradD:
    """
        Function: use to store concat base parameters
    """

    def __init__(self, var_dtype, idx_dtype, kernel_name):
        """
        constructor of class SparseApplyProximalAdagradD

        Parameters
        ----------
        var_dtype: dtype of var
        idx_dtype: dtype of idx
        kernel_name: kernel_name, default value is "SparseApplyProximalAdagradD"

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.var_dtype = var_dtype
        self.idx_dtype = idx_dtype
        self.kernel_name = kernel_name
        self.is_double_buffer = True
        self.ub_size = _tik_get_ub_size(self.is_double_buffer)
        self.core_num = _tik_get_core_num()
        self.ub_tensor_num = 8

        class GmTensor():
            """
            Function: use to store concat base parameters
            """

            def __init__(self, tik_instance, var_dtype, idx_dtype):
                """
                constructor of class GmTensor

                Parameters
                ----------
                tik_instance: tik_instance
                var_dtype: var dtype
                idx_dtype: idx dtype

                Returns
                -------
                None
                """
                self.var_gm = tik_instance.Tensor(var_dtype,
                                                  Constant.MAX_GM_TENSOR_SHAPE,
                                                  name="var_gm",
                                                  scope=tik.scope_gm)
                self.accum_gm = tik_instance.Tensor(var_dtype,
                                                    Constant.MAX_GM_TENSOR_SHAPE,
                                                    name="accum_gm",
                                                    scope=tik.scope_gm)
                self.lr_gm = tik_instance.Tensor(var_dtype,
                                                 Constant.MIN_TENSOR_SHAPE,
                                                 name="lr_gm",
                                                 scope=tik.scope_gm)
                self.l1_gm = tik_instance.Tensor(var_dtype,
                                                 Constant.MIN_TENSOR_SHAPE,
                                                 name="l1_gm",
                                                 scope=tik.scope_gm)
                self.l2_gm = tik_instance.Tensor(var_dtype,
                                                 Constant.MIN_TENSOR_SHAPE,
                                                 name="l2_gm",
                                                 scope=tik.scope_gm)
                self.grad_gm = tik_instance.Tensor(var_dtype,
                                                   Constant.MAX_GM_TENSOR_SHAPE,
                                                   name="grad_gm",
                                                   scope=tik.scope_gm)
                self.idx_gm = tik_instance.Tensor(idx_dtype,
                                                  Constant.MAX_GM_TENSOR_SHAPE,
                                                  name="index_gm",
                                                  scope=tik.scope_gm)
                self.var_out_gm = tik_instance.Tensor(var_dtype,
                                                      Constant.MAX_GM_TENSOR_SHAPE,
                                                      name="var_out_gm",
                                                      scope=tik.scope_gm)
                self.accum_out_gm = tik_instance.Tensor(var_dtype,
                                                        Constant.MAX_GM_TENSOR_SHAPE,
                                                        name="accum_out_gm",
                                                        scope=tik.scope_gm)
                self.tiling_gm = tik_instance.Tensor(Constant.TILING_PARAM_DTYPE,
                                                     Constant.MIN_TENSOR_SHAPE,
                                                     name="tiling_gm",
                                                     scope=tik.scope_gm)

        class Fp32InputScalar():
            """
                Function: use to store concat base parameters
            """

            def __init__(self, tik_instance):
                """
                constructor of class Fp32InputScalar

                Parameters
                ----------
                tik_instance: tik_instance

                Returns
                -------
                None
                """
                self.select_key = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="select_key")
                self.need_core_num = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="need_core_num")
                self.idx_mov_times = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="idx_mov_times")
                self.idx_front_num = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="idx_front_num")
                self.idx_last_num = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="idx_last_num")
                self.idx_front_burstlen = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="idx_front_burstlen")
                self.idx_last_burstlen = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="idx_last_burstlen")
                self.one_row_burstlen = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="one_row_burstlen")
                self.one_row_num = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="one_row_num")
                self.vec_repeat_time = \
                    tik_instance.Scalar(dtype=Constant.TILING_PARAM_DTYPE,
                                        name="vec_repeat_time")

        last_axis = 80
        last_axis_ceil = ceil(last_axis, Constant.MASK_FP32)
        ub_shape = (last_axis_ceil,)

        def compute_row(ub_size):
            """
            compute row in grad and idx

            Parameters
            ----------
            ub_size: ub_size

            Returns
            -------
            row of grad and idx
            """
            row = (ub_size - 1024 * 8) / \
                  (last_axis_ceil + 1) / \
                  Constant.BYTE_FP32
            row = row // Constant.FP32_ELE_NUM_BLOCK * Constant.FP32_ELE_NUM_BLOCK
            return row

        class UbTensor():
            """
                Function: use to store concat base parameters
            """

            def __init__(self, tik_instance, var_dtype, idx_dtype, ub_size):
                """
                constructor of class UbTensor

                Parameters
                ----------
                tik_instance: tik_instance
                var_dtype: var_dtype
                idx_dtype: idx_dtype
                ub_size: ub_size

                Returns
                -------
                None
                """
                self.accum_ub = tik_instance.Tensor(var_dtype,
                                                    ub_shape,
                                                    name="accum_ub",
                                                    scope=tik.scope_ubuf)
                self.var_ub = tik_instance.Tensor(var_dtype,
                                                  ub_shape,
                                                  name="var_ub",
                                                  scope=tik.scope_ubuf)
                self.prox_ub = tik_instance.Tensor(var_dtype,
                                                   ub_shape,
                                                   name="prox_ub",
                                                   scope=tik.scope_ubuf)
                self.tmp_ub = tik_instance.Tensor(var_dtype,
                                                  ub_shape,
                                                  name="tmp_ub",
                                                  scope=tik.scope_ubuf)
                self.grad_ub = tik_instance.Tensor(var_dtype,
                                                   (compute_row(ub_size), last_axis_ceil),
                                                   name="grad_ub",
                                                   scope=tik.scope_ubuf)
                self.idx_ub = tik_instance.Tensor(idx_dtype,
                                                  (compute_row(ub_size),),
                                                  name="idx_ub",
                                                  scope=tik.scope_ubuf)
                self.lr_ub = tik_instance.Tensor(var_dtype,
                                                 ub_shape,
                                                 name="lr_ub",
                                                 scope=tik.scope_ubuf)
                self.l1_ub = tik_instance.Tensor(var_dtype,
                                                 ub_shape,
                                                 name="l1_ub",
                                                 scope=tik.scope_ubuf)
                self.l2_ub = tik_instance.Tensor(var_dtype,
                                                 ub_shape,
                                                 name="l2_ub",
                                                 scope=tik.scope_ubuf)
                self.zero_ub = tik_instance.Tensor(var_dtype,
                                                   ub_shape,
                                                   name="zero_ub",
                                                   scope=tik.scope_ubuf)
                self.one_ub = tik_instance.Tensor(var_dtype,
                                                  ub_shape,
                                                  name="one_ub",
                                                  scope=tik.scope_ubuf)
                self.neg_one_ub = tik_instance.Tensor(var_dtype,
                                                      ub_shape,
                                                      name="neg_one_ub",
                                                      scope=tik.scope_ubuf)
                self.lr1_ub = tik_instance.Tensor(var_dtype,
                                                  ub_shape,
                                                  name="lr1_ub",
                                                  scope=tik.scope_ubuf)
                self.lr2_ub = tik_instance.Tensor(var_dtype,
                                                  ub_shape,
                                                  name="lr2_ub",
                                                  scope=tik.scope_ubuf)
                self.var_t1_ub = tik_instance.Tensor(var_dtype,
                                                     ub_shape,
                                                     name="var_t1_ub",
                                                     scope=tik.scope_ubuf)

        self.obj_fp32_input_scalar = Fp32InputScalar(self.tik_instance)
        self.gm_tensor = \
            GmTensor(self.tik_instance, self.var_dtype, self.idx_dtype)
        self.ub_tensor = \
            UbTensor(self.tik_instance, self.var_dtype, self.idx_dtype,
                     self.ub_size)
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor(Constant.TILING_PARAM_DTYPE,
                                                 Constant.MIN_TENSOR_SHAPE,
                                                 name="tiling_ub",
                                                 scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.gm_tensor.tiling_gm,
                                        0, 1,
                                        Constant.TILING_PARAMS_NUM * Constant.BYTE_INT32 // Constant.BYTE_BLOCK,
                                        0, 0)
            index = 0
            self.obj_fp32_input_scalar.select_key.set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.need_core_num.set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.idx_mov_times.set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.idx_front_num.set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.idx_last_num.set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.idx_front_burstlen. \
                set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.idx_last_burstlen. \
                set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.one_row_burstlen. \
                set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.one_row_num.set_as(tiling_ub[index])
            index = index + 1
            self.obj_fp32_input_scalar.vec_repeat_time. \
                set_as(tiling_ub[index])
        # init ub tensor
        with self.tik_instance.new_stmt_scope():
            lr_ub_t = self.tik_instance.Tensor(self.var_dtype,
                                               Constant.MIN_TENSOR_SHAPE,
                                               name="lr_ub_t",
                                               scope=tik.scope_ubuf)
            l1_ub_t = self.tik_instance.Tensor(self.var_dtype,
                                               Constant.MIN_TENSOR_SHAPE,
                                               name="l1_ub_t",
                                               scope=tik.scope_ubuf)
            l2_ub_t = self.tik_instance.Tensor(self.var_dtype,
                                               Constant.MIN_TENSOR_SHAPE,
                                               name="l2_ub_t",
                                               scope=tik.scope_ubuf)
            self.tik_instance.data_move(lr_ub_t[0], self.gm_tensor.lr_gm,
                                        0, 1, 1,
                                        0, 0)
            self.tik_instance.data_move(l1_ub_t[0], self.gm_tensor.l1_gm,
                                        0, 1, 1,
                                        0, 0)
            self.tik_instance.data_move(l2_ub_t[0], self.gm_tensor.l2_gm,
                                        0, 1, 1,
                                        0, 0)
            lr_scalar = self.tik_instance.Scalar(self.var_dtype,
                                                 name="lr_scalar")
            l1_scalar = self.tik_instance.Scalar(self.var_dtype,
                                                 name="l1_scalar")
            l2_scalar = self.tik_instance.Scalar(self.var_dtype,
                                                 name="l2_scalar")
            lr_scalar.set_as(lr_ub_t[0])
            l1_scalar.set_as(l1_ub_t[0])
            l2_scalar.set_as(l2_ub_t[0])
            self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor.lr_ub[0],
                                         lr_scalar, self.obj_fp32_input_scalar.vec_repeat_time, 1, 8)
            self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor.l1_ub[0],
                                         l1_scalar, self.obj_fp32_input_scalar.vec_repeat_time, 1, 8)
            self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor.l2_ub[0],
                                         l2_scalar, self.obj_fp32_input_scalar.vec_repeat_time, 1, 8)
            self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor.one_ub[0],
                                         Constant.ONE_FP32, self.obj_fp32_input_scalar.vec_repeat_time, 1, 8)
            self.tik_instance.vector_dup(Constant.MASK_FP32, self.ub_tensor.zero_ub[0],
                                         Constant.ZERO_FP32, self.obj_fp32_input_scalar.vec_repeat_time, 1, 8)
            self.tik_instance.vector_dup(Constant.MASK_FP32,
                                         self.ub_tensor.neg_one_ub[0], Constant.NEG_ONE_FP32,
                                         self.obj_fp32_input_scalar.vec_repeat_time, 1, 8)

    def sparse_apply_proximal_adagrad_d(self):
        """
        main process of sparse_apply_proximal_adagrad_d

        Parameters
        ----------
        None

        Returns:
        -------
        None
        """
        with self.tik_instance.for_range(0, self.core_num,
                                         block_num=self.core_num) as \
                block_index:
            with self.tik_instance.if_scope(block_index <
                                            self.obj_fp32_input_scalar.need_core_num):
                with self.tik_instance.if_scope(
                        self.obj_fp32_input_scalar.select_key ==
                        Constant.SELECT_KEY_LAST_AXIS_80_FLOAT):
                    compute_sparse_apply_proximal_adagrad_d(block_index,
                                                            self.tik_instance,
                                                            self.gm_tensor,
                                                            self.ub_tensor,
                                                            self.obj_fp32_input_scalar)
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(inputs=[self.gm_tensor.var_gm,
                                           self.gm_tensor.accum_gm,
                                           self.gm_tensor.lr_gm,
                                           self.gm_tensor.l1_gm,
                                           self.gm_tensor.l2_gm,
                                           self.gm_tensor.grad_gm,
                                           self.gm_tensor.idx_gm],
                                   outputs=[self.gm_tensor.var_out_gm,
                                            self.gm_tensor.accum_out_gm],
                                   flowtable=[self.gm_tensor.tiling_gm],
                                   config=opt_config,
                                   kernel_name=self.kernel_name)


def _tik_get_ub_size(is_double_buffer=True):
    """
    get ub size

    Parameters
    ----------
    is_double_buffer: is_double_buffer flag

    Returns
    -------
    ub_size
    """
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    if is_double_buffer:
        return ub_size // 2
    return ub_size


def _tik_get_core_num():
    """
    get core num

    Parameters
    ----------
    None

    Returns
    -------
    core num
    """
    return tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)


def compute_sparse_apply_proximal_adagrad_d(block_index,
                                            tik_instance,
                                            gm_tensor,
                                            ub_tensor,
                                            fp32_input_scalar):
    """
    compute_sparse_apply_proximal_adagrad_d

    Parameters
    ----------
    block_index: block_index
    tik_instance: tik_instance
    gm_tensor: gm_tensor
    ub_tensor: ub_tensor
    fp32_input_scalar: fp32 tiling param

    Returns
    -------
    None
    """
    idx_val = tik_instance.Scalar(dtype=Constant.DTYPE_INT32,
                                  name="idx_val")
    with tik_instance.for_range(0, fp32_input_scalar.idx_mov_times) as \
            idx_mov_index:
        # tiling idx by ub
        with tik_instance.if_scope(idx_mov_index <
                                   fp32_input_scalar.idx_mov_times - 1):
            # front part
            tik_instance.data_move(ub_tensor.idx_ub[0],
                                   gm_tensor.idx_gm[idx_mov_index *
                                                    fp32_input_scalar.idx_front_num],
                                   0, 1, fp32_input_scalar.idx_front_burstlen,
                                   0, 0)
            with tik_instance.for_range(0, fp32_input_scalar.idx_front_num) as \
                    idx_index:
                # traversal_idx
                traversal_idx(tik_instance,
                              gm_tensor,
                              ub_tensor,
                              fp32_input_scalar,
                              idx_val,
                              idx_mov_index,
                              idx_index)
        with tik_instance.if_scope(idx_mov_index ==
                                   fp32_input_scalar.idx_mov_times - 1):
            # last part
            tik_instance.data_move(ub_tensor.idx_ub[0],
                                   gm_tensor.idx_gm[idx_mov_index *
                                                    fp32_input_scalar.idx_front_num],
                                   0, 1, fp32_input_scalar.idx_last_burstlen,
                                   0, 0)
            with tik_instance.for_range(0, fp32_input_scalar.idx_last_num) as \
                    idx_index:
                # traversal_idx
                traversal_idx(tik_instance,
                              gm_tensor,
                              ub_tensor,
                              fp32_input_scalar,
                              idx_val,
                              idx_mov_index,
                              idx_index)


def traversal_idx(tik_instance,
                  gm_tensor,
                  ub_tensor,
                  fp32_input_scalar,
                  idx_val,
                  idx_mov_index,
                  idx_index):
    """
    update accum and var by traversal idx

    Parameters
    ----------
    tik_instance: tik_instance
    gm_tensor: gm_tensor
    ub_tensor: ub_tensor
    fp32_input_scalar: fp32 tiling param
    idx_val: idx val scalar
    idx_mov_index: idx_mov_index tiling by ub
    idx_index: idx_index in ub

    Returns
    -------
    None
    """
    l1_scalar = tik_instance.Scalar(ub_tensor.l1_ub.dtype, "l1_scalar")
    l1_scalar.set_as(ub_tensor.l1_ub[0])
    # traversal idx
    idx_val.set_as(ub_tensor.idx_ub[idx_index])
    # mov grad
    tik_instance.data_move(ub_tensor.grad_ub[0],
                           gm_tensor.grad_gm[idx_mov_index *
                                             fp32_input_scalar.idx_front_num *
                                             fp32_input_scalar.one_row_num +
                                             idx_index *
                                             fp32_input_scalar.one_row_num],
                           0, 1,
                           fp32_input_scalar.one_row_burstlen,
                           0, 0)
    # mov accum gm2ub
    tik_instance.data_move(ub_tensor.accum_ub[0],
                           gm_tensor.accum_gm[idx_val *
                                              fp32_input_scalar.one_row_num],
                           0, 1,
                           fp32_input_scalar.one_row_burstlen,
                           0, 0)
    # `vmla, accum += grad*grad`
    tik_instance.vmla(Constant.MASK_FP32, ub_tensor.accum_ub[0],
                      ub_tensor.grad_ub[0],
                      ub_tensor.grad_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1, 8, 8, 8)
    # mov accum ub2gm
    tik_instance.data_move(gm_tensor.accum_gm[idx_val *
                                              fp32_input_scalar.one_row_num],
                           ub_tensor.accum_ub[0],
                           0, 1,
                           fp32_input_scalar.one_row_burstlen,
                           0, 0)
    tik_instance.vsqrt(Constant.MASK_FP32, ub_tensor.tmp_ub[0],
                       ub_tensor.accum_ub[0],
                       fp32_input_scalar.vec_repeat_time,
                       1, 1, 8, 8)
    tik_instance.vdiv(Constant.MASK_FP32, ub_tensor.lr1_ub[0],
                      ub_tensor.lr_ub[0],
                      ub_tensor.tmp_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1,
                      8, 0, 8)
    tik_instance.vmul(Constant.MASK_FP32, ub_tensor.tmp_ub[0],
                      ub_tensor.grad_ub[0],
                      ub_tensor.lr1_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1,
                      8, 8, 8)
    # mov var gm2ub
    tik_instance.data_move(ub_tensor.var_ub[0],
                           gm_tensor.var_gm[idx_val * \
                                            fp32_input_scalar.one_row_num],
                           0, 1,
                           fp32_input_scalar.one_row_burstlen,
                           0, 0)
    tik_instance.vsub(Constant.MASK_FP32, ub_tensor.prox_ub[0],
                      ub_tensor.var_ub[0],
                      ub_tensor.tmp_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1,
                      8, 8, 8)
    tik_instance.vmul(Constant.MASK_FP32, ub_tensor.tmp_ub[0],
                      ub_tensor.l2_ub[0],
                      ub_tensor.lr1_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1,
                      8, 0, 8)
    tik_instance.vadd(Constant.MASK_FP32, ub_tensor.accum_ub[0],
                      ub_tensor.one_ub[0],
                      ub_tensor.tmp_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1,
                      8, 0, 8)
    tik_instance.vdiv(Constant.MASK_FP32, ub_tensor.lr2_ub[0],
                      ub_tensor.one_ub[0],
                      ub_tensor.accum_ub[0],
                      fp32_input_scalar.vec_repeat_time,
                      1, 1, 1,
                      8, 0, 8)
    with tik_instance.if_scope(l1_scalar > Constant.ZERO_FP32):
        tik_instance.vmul(Constant.MASK_FP32, ub_tensor.tmp_ub[0],
                          ub_tensor.lr1_ub[0],
                          ub_tensor.l1_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1, 1,
                          8, 8, 0)
        tik_instance.vabs(Constant.MASK_FP32, ub_tensor.accum_ub[0],
                          ub_tensor.prox_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1,
                          8, 8)
        tik_instance.vsub(Constant.MASK_FP32, ub_tensor.var_t1_ub[0],
                          ub_tensor.accum_ub[0],
                          ub_tensor.tmp_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1, 1,
                          8, 8, 8)
        tik_instance.vmax(Constant.MASK_FP32, ub_tensor.tmp_ub[0],
                          ub_tensor.var_t1_ub[0],
                          ub_tensor.zero_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1, 1,
                          8, 8, 0)
        tik_instance.vmul(Constant.MASK_FP32, ub_tensor.accum_ub[0],
                          ub_tensor.tmp_ub[0],
                          ub_tensor.lr2_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1, 1,
                          8, 8, 8)
        with tik_instance.for_range(0,
                                    fp32_input_scalar.vec_repeat_time) as \
                repeat_index:
            ub_offset = repeat_index * Constant.MASK_FP32
            cmp_mask = tik_instance.vcmp_gt(Constant.MASK_FP32,
                                            ub_tensor.prox_ub[ub_offset],
                                            ub_tensor.zero_ub[0],
                                            1, 1)
            tik_instance.vsel(Constant.MASK_FP32, 0,
                              ub_tensor.tmp_ub[0],
                              cmp_mask,
                              ub_tensor.one_ub[0],
                              ub_tensor.zero_ub[0],
                              1, 1, 1, 1, 0, 0, 0)
            cmp_mask = tik_instance.vcmp_lt(Constant.MASK_FP32,
                                            ub_tensor.prox_ub[ub_offset],
                                            ub_tensor.zero_ub[0],
                                            1, 1)
            tik_instance.vsel(Constant.MASK_FP32, 0,
                              ub_tensor.prox_ub[ub_offset],
                              cmp_mask,
                              ub_tensor.neg_one_ub[0],
                              ub_tensor.tmp_ub[0],
                              1, 1, 1, 1, 0, 0, 0)
        tik_instance.vmul(Constant.MASK_FP32, ub_tensor.var_ub[0],
                          ub_tensor.prox_ub[0],
                          ub_tensor.accum_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1, 1,
                          8, 8, 8)
    with tik_instance.else_scope():
        tik_instance.vmul(Constant.MASK_FP32, ub_tensor.var_ub[0],
                          ub_tensor.prox_ub[0],
                          ub_tensor.lr2_ub[0],
                          fp32_input_scalar.vec_repeat_time,
                          1, 1, 1,
                          8, 8, 8)
    # mov var ub2gm
    tik_instance.data_move(gm_tensor.var_gm[idx_val * \
                                            fp32_input_scalar.one_row_num],
                           ub_tensor.var_ub[0],
                           0, 1,
                           fp32_input_scalar.one_row_burstlen,
                           0, 0)


@register_operator("SparseApplyProximalAdagradD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sparse_apply_proximal_adagrad_d(var_dict, accum_dict, lr_dict, l1_dict,
                                    l2_dict, grad_dict, indices_dict,
                                    var_out_dict, accum_out_dict,
                                    use_locking=False,
                                    kernel_name="SparseApplyProximalAdagradD"):
    """
    sparse_apply_proximal_adagrad_d op entry interface

    Parameters
    ----------
    var_dict: var params shape, dtype and range
    accum_dict: accum shape, dtype and range
    lr_dict: lr shape, dtype and range
    l1_dict: l1 shape, dtype and range
    l2_dict: l2 shape, dtype and range
    grad_dict: grad shape, dtype and range
    indices_dict: indices shape, dtype and range
    var_out_dict: var output shape, dtype and range
    accum_out_dict: accum output shape, dtype and range
    use_locking: default value is "False"
    kernel_name: kernel name of SparseApplyProximalAdagradD op

    Returns
    -------
    compile info
    """
    var_dtype_check_list = ("float32")
    indices_dtype_check_list = ("int32")

    var_dtype = var_dict.get("dtype").lower()
    para_check.check_dtype(var_dtype, var_dtype_check_list, param_name="var_dict")

    accum_dtype = accum_dict.get("dtype").lower()
    para_check.check_dtype(accum_dtype, var_dtype_check_list, param_name="accum_dict")

    lr_dtype = lr_dict.get("dtype").lower()
    para_check.check_dtype(lr_dtype, var_dtype_check_list, param_name="lr_dict")

    l1_dtype = l1_dict.get("dtype").lower()
    para_check.check_dtype(l1_dtype, var_dtype_check_list, param_name="l1_dict")

    l2_dtype = l2_dict.get("dtype").lower()
    para_check.check_dtype(l2_dtype, var_dtype_check_list, param_name="l2_dict")

    grad_dtype = grad_dict.get("dtype").lower()
    para_check.check_dtype(grad_dtype, var_dtype_check_list, param_name="grad_dict")

    indices_dtype = indices_dict.get("dtype").lower()
    para_check.check_dtype(indices_dtype, indices_dtype_check_list,
                           param_name="indices_dict")

    var_out_dtype = var_out_dict.get("dtype").lower()
    para_check.check_dtype(var_out_dtype, var_dtype_check_list, param_name="var_out_dict")

    accum_out_dtype = accum_out_dict.get("dtype").lower()
    para_check.check_dtype(accum_out_dtype, var_dtype_check_list,
                           param_name="accum_out_dict")

    if var_dtype != var_out_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name,
                                                              "var",
                                                              "var_out",
                                                              var_dtype,
                                                              var_out_dtype)
    if accum_dtype != accum_out_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name,
                                                              "accum",
                                                              "accum_out",
                                                              accum_dtype,
                                                              accum_out_dtype)

    obj = SparseApplyProximalAdagradD(var_dtype, indices_dtype, kernel_name)
    obj.sparse_apply_proximal_adagrad_d()
    # add compile info
    tbe_context.get_context().add_compile_info("vars",
                                               {"ub_size": obj.ub_size, "core_num": obj.core_num,
                                                "ub_tensor_num": obj.ub_tensor_num})
