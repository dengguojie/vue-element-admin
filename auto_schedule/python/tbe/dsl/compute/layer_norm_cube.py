# Copyright 2021 Huawei Technologies Co., Ltd
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
layer_norm_cube_compute
"""
import tbe
from tbe.common.utils import shape_util
from tbe import tvm

BLOCK_K0 = 16
BLOCK_M0 = 16


class LayerNormCube(object):
    """
    LayerNormCube: compute definition of layernorm

    Functions
    ----------
    __init__ : initialization

    layer_norm_cube_compute : compute definition of layernorm

    """
    def __init__(self, para):
        """
        initialization

        Parameters:
        ----------
        para : the input para

        Returns
        -------
        None
        """
        self._m_dim = None
        self._k_dim = None
        self._shape_nz = None
        self._reduce_shape_nz = None
        self._reduce_cof = None
        self._para = para
        self._batch = None

    def _init_input_tensor(self, x, gamma, beta):

        shape_gamma = shape_util.shape_to_list(gamma.shape)
        shape_x_l0a = [self._m_dim, self._k_dim, BLOCK_M0, BLOCK_K0]
        if self._batch:
            shape_x_l0a = self._batch + shape_x_l0a

        x_l1 = tvm.compute(self._shape_nz, lambda *indices: x(*indices),
                           name="x_l1")

        x_l0a = tvm.compute(shape_x_l0a,
                            lambda *indices: x_l1(*indices[:-4], indices[-3], indices[-4], indices[-2], indices[-1]),
                            name="x_l0a")

        x_ub = tvm.compute(self._shape_nz, lambda *indices: x_l1(*indices),
                           name="x_ub")

        x_ub_fp32 = tvm.compute(self._shape_nz,
                                lambda *indices: (x_ub(*indices)).astype("float32"),
                                name="cast_x_ub")
        gamma_ub = tvm.compute(shape_gamma,
                               lambda *indices: gamma(*indices),
                               name="input_gamma_ub")
        beta_ub = tvm.compute(shape_gamma,
                              lambda *indices: beta(*indices),
                              name="input_beta_ub")
        gamma_ub_fp32 = tvm.compute(shape_gamma,
                                    lambda *indices: (gamma_ub(*indices)).astype("float32"),
                                    name="cast_input_gamma")
        beta_ub_fp32 = tvm.compute(shape_gamma,
                                   lambda *indices: (beta_ub(*indices)).astype("float32"),
                                   name="cast_input_beta")

        return [x_l1, x_l0a, x_ub_fp32], gamma_ub_fp32, beta_ub_fp32

    def _cal_mean_of_ln(self, x_l0a):
        # cal the mean of layernorm
        shape_l0b = [self._k_dim, 1, BLOCK_M0, BLOCK_K0]
        # the reduce axis
        reduce_k = tvm.reduce_axis((0, self._k_dim), name="sum_k1")
        reduce_k0 = tvm.reduce_axis((0, BLOCK_K0), name="sum_k0")

        one_l0b = tvm.compute(shape_l0b, lambda *indices: tvm.const(1, "float16"),
                              name="one_l0b")
        x_sum = tvm.compute(self._reduce_shape_nz,
                            lambda *indices: tvm.sum(
                                (x_l0a(*indices[:-4], indices[-3], reduce_k, indices[-2], reduce_k0)
                                 * one_l0b(reduce_k, indices[-4], indices[-1], reduce_k0)
                                 ).astype("float32"),
                                axis=[reduce_k, reduce_k0]),
                            name="x_sum_l0c")
        x_sum_cub = tvm.compute(self._reduce_shape_nz, lambda *indices: x_sum(*indices),
                                name="x_sum_cub",
                                tag="x_sum_cub")
        mean = tvm.compute(self._reduce_shape_nz,
                           lambda *indices: x_sum_cub(*indices) * self._reduce_cof,
                           name="nz_mean",
                           tag="nz_mean")
        return mean

    def _cal_var_of_ln(self, x_l0a, x_l1, mean):
        # cal the var of layernorm
        shape_xx_sum = [self._m_dim, self._m_dim, BLOCK_M0, BLOCK_M0]
        shape_xx_sum_scalar = [self._m_dim, BLOCK_M0]
        if self._batch:
            shape_xx_sum = self._batch + shape_xx_sum
            shape_xx_sum_scalar = self._batch + shape_xx_sum_scalar
        # the reduce axis
        reduce_k = tvm.reduce_axis((0, self._k_dim), name="square_k1")
        reduce_k0 = tvm.reduce_axis((0, BLOCK_K0), name="square_k0")

        x_l0b = tvm.compute(self._shape_nz,
                            lambda *indices: x_l1(*indices),
                            name="x_l0b")
        xx_sum = tvm.compute(shape_xx_sum,
                             lambda *indices: tvm.sum(
                                 (x_l0a(*indices[:-4], indices[-3], reduce_k, indices[-2], reduce_k0)
                                  * x_l0b(*indices[:-4], reduce_k, indices[-4], indices[-1], reduce_k0)
                                  ).astype("float32"),
                                 axis=[reduce_k, reduce_k0]),
                             name="xx_sum_l0c")

        xx_sum_cub = tvm.compute(shape_xx_sum, lambda *indices: xx_sum(*indices),
                                 name="xx_sum_cub",
                                 tag="xx_sum_cub")

        xx_sum_cub_scalar = tvm.compute(shape_xx_sum_scalar,
                                        lambda *indices: xx_sum_cub(*indices[:-2], indices[-2], indices[-2], indices[-1], indices[-1]),
                                        name="xx_sum_cub_scalar")
        xx_sum_cub_broadcast = tvm.compute(self._reduce_shape_nz,
                                           lambda *indices: xx_sum_cub_scalar(
                                               *indices[:-4], indices[-3], indices[-2]),
                                           name="xx_sum_cub_broadcast")
        xx_sum_cub = tvm.compute(self._reduce_shape_nz,
                                 lambda *indices: xx_sum_cub_broadcast(*indices) * self._reduce_cof,
                                 name="xx_sum")
        mean_multiple = tvm.compute(self._reduce_shape_nz,
                                    lambda *indices: mean(*indices) * mean(*indices),
                                    name="mean_mul",
                                    tag="mean_mul")
        var = tvm.compute(self._reduce_shape_nz,
                          lambda *indices: xx_sum_cub(*indices) - mean_multiple(*indices),
                          name="nz_var",
                          tag="nz_var")
        return var

    def _mean_var_nz2nd(self, nz_input, para_name):
        # chanspose NZ to ND of mean and var
        nd_output_shape = [self._m_dim * BLOCK_M0, 1]
        if self._batch:
            nd_output_shape = self._batch + nd_output_shape

        nd_fp16 = tvm.compute(self._reduce_shape_nz,
                              lambda *indices: (nz_input(*indices)).astype("float16"),
                              name="cast_" + para_name,
                              tag="cast_" + para_name)

        nd_trans = tvm.compute(self._reduce_shape_nz,
                               lambda *indices: nd_fp16(*indices[:-4], indices[-4], indices[-3], indices[-1], indices[-2]),
                               name="trans_" + para_name,
                               tag="trans_" + para_name)
        nd_out = tvm.compute(nd_output_shape,
                             lambda *indices: nd_trans(*indices[:-2], 0, indices[-2] // 16, 0, indices[-2] % 16),
                             name=para_name + "_out",
                             tag=para_name + "_out")
        return nd_out

    def _cal_y_of_ln(self, x, mean, var, epsilon):
        # cal the result of layer_norm

        x_sub_mean = tvm.compute(self._shape_nz,
                                 lambda *indices: x(*indices)
                                                  - mean(*indices[:-4], 0, *indices[-3:]),
                                 name="x_sub_mean")
        var_add_eps = tvm.compute(self._reduce_shape_nz,
                                  lambda *indices: var(*indices) + tvm.const(epsilon, dtype="float32"),
                                  name="var_add_eps",
                                  tag="var_add_eps")
        var_log = tbe.dsl.vlog(var_add_eps)
        var_log_mul = tvm.compute(self._reduce_shape_nz,
                                  lambda *indices: var_log(*indices) * tvm.const(-0.5, dtype="float32"),
                                  name="var_log_mul",
                                  tag="var_log_mul")
        var_exp = tbe.dsl.vexp(var_log_mul)

        y = tvm.compute(self._shape_nz,
                          lambda *indices: x_sub_mean(*indices)
                                           * var_exp(*indices[:-4], 0, *indices[-3:]),
                          name="mean_mul_var")

        return y

    def layer_norm_cube_compute(self, input_x, input_gamma, input_beta):
        """
        input_x : Tensor
            the tvm tensor of input x, only support float16
        input_gamma: dict
            the tvm tensor of input x, only support float16
        input_beta: dict
            the tvm tensor of input x, only support float16

        Returns
        -------
        Result of mean, var, res
        """
        shape_x = shape_util.shape_to_list(input_x.shape)
        ori_shape = self._para.get("ori_shape")
        epsilon = self._para.get("epsilon")
        reduce_num = ori_shape[-1]

        self._shape_nz = shape_x
        self._reduce_cof = (reduce_num * 1.0) ** (-1)
        self._k_dim, self._m_dim = shape_x[-4:-2]
        self._reduce_shape_nz = [1, self._m_dim, BLOCK_M0, 16]
        if len(shape_x) > 4:
            self._batch = shape_x[:-4]
            self._reduce_shape_nz = self._batch + self._reduce_shape_nz

        # get input tensor in l1, ub, l0a
        x_tensorlist, gamma_ub_fp32, beta_ub_fp32 = self._init_input_tensor(input_x, input_gamma, input_beta)
        x_l1, x_l0a, x_ub_fp32 = x_tensorlist

        # get mean in nz
        mean_nz = self._cal_mean_of_ln(x_l0a)

        # get var in nz
        var_nz = self._cal_var_of_ln(x_l0a, x_l1, mean_nz)

        # nz to nd of mean and var
        mean_nd = self._mean_var_nz2nd(mean_nz, "mean")
        var_nd = self._mean_var_nz2nd(var_nz, "var")

        # cal (x-mean)/(var+eps)^0.5
        y = self._cal_y_of_ln(x_ub_fp32, mean_nz, var_nz, epsilon)

        # cal gamma*y + beta
        scale_mul = tvm.compute(shape_x,
                                lambda *indices: y(*indices) * gamma_ub_fp32(indices[-1] + indices[-4]*BLOCK_K0),
                                name="scale_mul")
        res_fp32 = tvm.compute(shape_x,
                               lambda *indices: scale_mul(*indices) + beta_ub_fp32(indices[-1] + indices[-4]*BLOCK_K0),
                               name="scale_add")
        #  turn res from fp32 to fp16
        res = tvm.compute(shape_x,
                          lambda *indices: (res_fp32(*indices)).astype("float16"),
                          name="cast_res")
        res = tvm.compute(shape_x, lambda *indices: res(*indices),
                          name="res_out",
                          tag="res_out")

        return mean_nd, var_nd, res