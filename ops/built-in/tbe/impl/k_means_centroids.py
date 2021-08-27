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
k_means_centroids
"""
from functools import reduce as functools_reduce

from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from tbe.common.platform import get_soc_spec

FP16_TYPE = 2
FP32_TYPE = 4
UB_BLOCK_SIZE = 32
L0C_BLOCK_SIZE = 1024
SCALAR_MAX_FP32 = (2 ** 30 + 2 ** 29)
VECTOR_REPEAT_MAX = 255
FP32_MASK = 64
INPUT_LENGTH = 2
VECTOR_LENGTH = 128
INDEX_DTYPE = "int32"
MASK_DTYPE = "uint64"

dtype_dict = {
    "float16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "int64": 8
}


class KMeansCentroids(object):
    """
    class of k-means centroids operator
    """
    def __init__(self, x, y, sum_square_x, sum_square_y,
                 segment_sum, segment_count, total_distance,
                 use_actual_distance, kernel_name="k_means_centroids"):
        """
        init input, output, platform,
            tik instance, tiling and tensor
        """
        self.input_x1 = x
        self.input_x2 = y
        self.input_x3 = sum_square_x
        self.input_x4 = sum_square_y
        self.output_y1 = segment_sum
        self.output_y2 = segment_count
        self.output_y3 = total_distance
        self.use_actual_distance = use_actual_distance
        self.kernel_name = kernel_name
        self._get_platform_info()
        self.tik_instance = tik.Tik()
        self._default_tiling()
        self._tiling_process()
        self._init_tensor()

    @staticmethod
    def _ceil(x1, x2):
        if x2 == 0:
            reason = "Division by zero."
            error_manager_cube.raise_err_message_cube("k_means_centroids", reason)
        return (x1 + x2 - 1) // x2

    @staticmethod
    def _elecnt_of_shape(shape):
        """ calculate reduce shape
        """
        return functools_reduce(lambda x, y: x * y, shape)

    def _get_platform_info(self):
        self.soc_version = get_soc_spec("SOC_VERSION")
        self.aic_cnt = get_soc_spec("CORE_NUM")

    def _default_tiling(self):
        """ default tiling
        """
        self.tiling = {
            "block_dim": [1, 1, 1, 1],  # [batch,n,m,group]
            "AUB_shape": [128, 256, 1, 1],  # [kAUB,mAUB,batch,group]
            "BUB_shape": [128, 128, 1, 1],  # [kBUB,nBUB,batch,group]
            "AL1_shape": [128, 1, 1, 1],  # [k,multi_m,batch,group]
            "BL1_shape": [128, 1, 1, 1],  # [k,multi_n,batch,group]
            "AL0_matrix": [16, 8, 16, 16, 1, 1],  # [ma,ka,m0,k0,batch,group]
            "BL0_matrix": [8, 8, 16, 16, 1, 1],  # [kb,nb,n0,k0,batch,group]
            "CL0_matrix": [8, 16, 16, 16, 1, 1],  # [nc,mc,m0,n0,batch,group]
            "CUB_matrix": [4, 16, 16, 16, 1, 1],  # [nc_factor,mc_factor,m0,n0,batch,group]
            "manual_pingpong_buffer": {
                "AUB_pbuffer": 8,
                "BUB_pbuffer": 8,
                "AL1_pbuffer": 1,
                "BL1_pbuffer": 1,
                "AL0_pbuffer": 1,
                "BL0_pbuufer": 1,
                "CL0_pbuffer": 1,
                "CUB_pbuffer": 1,
                "UBG_pbuffer": 1
            }
        }

    def _tiling_process(self):
        """
        process tiling in multi-core level,
            one-core level and buffer level
        """
        if self.soc_version.find("Ascend910") != -1:
            self.tiling["BUB_shape"][1] = 224
            self.tiling["BL0_matrix"][1] = 14
            self.tiling["CL0_matrix"][0] = 14
            self.tiling["CUB_matrix"][0] = 7
            self.tiling["manual_pingpong_buffer"]["AUB_pbuffer"] = 1
            self.tiling["manual_pingpong_buffer"]["BUB_pbuffer"] = 1

        self._tiling_multi_core()
        self._tiling_one_core_matmul()
        self._tiling_one_core_argmin()

    def _tiling_multi_core(self):
        """ tiling in multi-core level
        """
        shape_x1 = self.input_x1.get("shape")
        shape_x2 = self.input_x2.get("shape")
        if shape_x1[0] < self.aic_cnt * self.tiling["AUB_shape"][1]:
            self.aic_cnt = self._ceil(shape_x1[0], self.tiling["AUB_shape"][1])
        # m axis bounds multi-core
        self.tiling["block_dim"][2] = self.aic_cnt

        if shape_x1[0] < self.tiling["AUB_shape"][1]:
            self.tiling["AUB_shape"][1] = shape_x1[0]
            self.tiling["AL0_matrix"][0] = self._ceil(shape_x1[0], 16)
            self.tiling["CL0_matrix"][1] = self._ceil(shape_x1[0], 16)
            self.tiling["CUB_matrix"][1] = self._ceil(shape_x1[0], 16)
        if shape_x2[0] < self.tiling["BUB_shape"][1]:
            self.tiling["BUB_shape"][1] = shape_x2[0]
            self.tiling["BL0_matrix"][1] = self._ceil(shape_x2[0], 16)
            self.tiling["CL0_matrix"][0] = self._ceil(shape_x2[0], 16)
            if shape_x2[0] < self.tiling["CUB_matrix"][0] * self.tiling["CUB_matrix"][3]:
                self.tiling["CUB_matrix"][0] = self._ceil(shape_x2[0], 16)

        m_dim = self.tiling["block_dim"][2]
        n_dim = self.tiling["block_dim"][1]
        self.m_each_core = max(self.tiling["AUB_shape"][1], self._ceil(shape_x1[0], m_dim))
        self.m_last_core = shape_x1[0] % self.m_each_core
        self.n_each_core = self._ceil(shape_x2[0], n_dim)
        self.n_last_core = shape_x2[0] % self.n_each_core

    def _tiling_one_core_matmul(self):
        """ matmul tiling in each core
        """
        k_aub, m_aub = self.tiling["AUB_shape"][:2]
        k_bub, n_bub = self.tiling["BUB_shape"][:2]
        k_al1, m_al1 = self.tiling["AL1_shape"][0], \
            self.tiling["AL1_shape"][1] * self.tiling["AL0_matrix"][0] * self.tiling["AL0_matrix"][2]
        k_bl1, n_bl1 = self.tiling["BL1_shape"][0], \
            self.tiling["BL1_shape"][1] * self.tiling["BL0_matrix"][1] * self.tiling["BL0_matrix"][2]
        ma, ka, m0, k0 = self.tiling["AL0_matrix"][:4]
        kb, nb, n0, k0 = self.tiling["BL0_matrix"][:4]
        nc, mc = self.tiling["CL0_matrix"][:2]
        nc_factor, mc_factor = self.tiling["CUB_matrix"][:2]

        self.m_tiling_loop = self.m_each_core // m_aub
        self.m_tiling_left = self.m_each_core % m_aub
        self.m_last_tiling_loop = 0
        self.m_last_tiling_left = 0
        if self.m_last_core > 0:
            self.m_last_tiling_loop = self.m_last_core // m_aub
            self.m_last_tiling_left = self.m_last_core % m_aub
        self.n_tiling_loop = self.n_each_core // n_bub
        self.n_tiling_left = self.n_each_core % n_bub

        self.n_tiling_ub_loop = nc // nc_factor
        self.n_tiling_cub_loop = 0
        self.n_tiling_cub_left = 0
        if self.n_tiling_left > 0:
            self.n_tiling_cub_loop = self.n_tiling_left // (nc_factor * n0)
            self.n_tiling_cub_left = self.n_tiling_left % (nc_factor * n0)

        self.shape_x_ub = (m_aub, k_aub)
        self.shape_x_ub_trans = (self._ceil(m_aub, 16), k_aub, 16)
        self.shape_y_ub = (n_bub, k_bub)
        self.shape_y_ub_trans = (self._ceil(n_bub, 16), k_bub, 16)

        self.shape_x_l1 = (self._ceil(m_al1, 16), k_al1, 16)
        self.shape_y_l1 = (self._ceil(n_bl1, 16), k_bl1, 16)
        self.shape_x_l0a = (ma, ka, m0, k0)
        self.shape_y_l0b = (kb, nb, n0, k0)
        self.shape_z_l0c = (nc, mc * m0, n0)

        self.shape_z_ub = (nc_factor, mc_factor * m0, n0)
        self.shape_z_ub_nd = (mc_factor * m0, nc_factor * n0)

    def _tiling_one_core_argmin(self):
        """ argmin and UnsortedSegmentSum tiling in each core
        """
        nc_factor, mc_factor, m0, n0 = self.tiling["CUB_matrix"][:4]
        self.m_tiling = mc_factor * m0
        self.n_tiling = nc_factor * n0
        self.shape_input_3_ub = (self.m_tiling, 1)
        self.shape_input_4_ub = (1, self.n_tiling)
        self.shape_broadcast_ub = (self.m_tiling, self.n_tiling)
        self.shape_global_min_distance_ub = (self.m_tiling,)
        self.shape_total_distance = (1,)

    def _init_tensor(self):
        """ init fixed-shape tensor
        """
        shape_x1 = self.input_x1.get("shape")
        shape_x2 = self.input_x2.get("shape")
        shape_x4 = self.input_x4.get("shape")
        self.input_dtype = self.input_x1.get("dtype")
        output_dtype = self.output_y1.get("dtype")
        n_gm, d_gm = shape_x2
        self.ub_min_num = UB_BLOCK_SIZE // dtype_dict[self.input_dtype]

        self.data_input_gm_1 = self.tik_instance.Tensor(self.input_dtype, shape_x1,
                                                        name="data_input_gm_1", scope=tik.scope_gm)
        self.data_input_gm_2 = self.tik_instance.Tensor(self.input_dtype, shape_x2,
                                                        name="data_input_gm_2", scope=tik.scope_gm)
        if self.use_actual_distance:
            shape_x3 = self.input_x3.get("shape")
            self.data_input_gm_3 = self.tik_instance.Tensor(self.input_dtype, shape_x3,
                                                            name="data_input_gm_3", scope=tik.scope_gm)
        self.data_input_gm_4 = self.tik_instance.Tensor(self.input_dtype, shape_x4,
                                                        name="data_input_gm_4", scope=tik.scope_gm)
        self.data_output_gm_1 = self.tik_instance.Tensor(output_dtype, (n_gm, d_gm),
                                                         name="data_output_gm_1", scope=tik.scope_gm,
                                                         is_atomic_add=True)
        self.data_output_gm_2 = self.tik_instance.Tensor(output_dtype, (n_gm, 1),
                                                         name="data_output_gm_2", scope=tik.scope_gm,
                                                         is_atomic_add=True)
        self.data_output_gm_3 = self.tik_instance.Tensor(output_dtype, self.shape_total_distance,
                                                         name="data_output_gm_3", scope=tik.scope_gm,
                                                         is_atomic_add=True)

        self.data_input_l1_1 = self.tik_instance.Tensor("float16", self.shape_x_l1,
                                                        name="data_input_l1_1", scope=tik.scope_cbuf)
        self.data_input_l1_2 = self.tik_instance.Tensor("float16", self.shape_y_l1,
                                                        name="data_input_l1_2", scope=tik.scope_cbuf)
        self.data_input_l0a = self.tik_instance.Tensor("float16", self.shape_x_l0a,
                                                       name="data_input_l0a", scope=tik.scope_ca)
        self.data_input_l0b = self.tik_instance.Tensor("float16", self.shape_y_l0b,
                                                       name="data_input_l0b", scope=tik.scope_cb)
        self.data_output_l0c = self.tik_instance.Tensor(output_dtype, self.shape_z_l0c,
                                                        name="data_output_l0c", scope=tik.scope_cc)

    def _init_matmul_tensor_a_ub(self):
        """ init tensor_a of matmul in ub buffer
        """
        self.data_input_ub_1 = self.tik_instance.Tensor("float32", self.shape_x_ub,
                                                        name="data_input_ub_1", scope=tik.scope_ubuf)
        self.data_input_ub_1_fp16 = self.tik_instance.Tensor("float16", self.shape_x_ub,
                                                             name="data_input_ub_1_fp16", scope=tik.scope_ubuf)
        self.data_input_ub_1_trans = self.tik_instance.Tensor("float16", self.shape_x_ub_trans,
                                                              name="data_input_ub_1_trans", scope=tik.scope_ubuf)

    def _init_matmul_tensor_a_ub_db(self, shape_x_ub, shape_x_ub_trans, double_buffer):
        """
        double buffer strategy

        Parameters:
        -------------
        shape_x_ub: (m, k)
        shape_x_ub_trans: (m1, k, m0)
        double_buffer: support 1,2,4,8

        Returns:
        -------------
        None
        """
        self.data_input_ub_1 = self.tik_instance.Tensor("float32", self.shape_x_ub,
                                                        name="data_input_ub_1", scope=tik.scope_ubuf)
        self.data_input_ub_1_fp16_list = []
        self.data_input_ub_1_trans_list = []
        for db_idx in range(double_buffer):
            tensor_name1 = "data_input_ub_1_fp16_%d" % (db_idx + 1)
            tensor_name2 = "data_input_ub_1_trans_%d" % (db_idx + 1)
            tensor_ins1 = self.tik_instance.Tensor("float16", shape_x_ub,
                                                   name=tensor_name1, scope=tik.scope_ubuf)
            tensor_ins2 = self.tik_instance.Tensor("float16", shape_x_ub_trans,
                                                   name=tensor_name2, scope=tik.scope_ubuf)
            self.data_input_ub_1_fp16_list.append(tensor_ins1)
            self.data_input_ub_1_trans_list.append(tensor_ins2)

    def _init_matmul_tensor_b_ub(self):
        """ init tensor_b of matmul in ub buffer
        """
        self.data_input_ub_2 = self.tik_instance.Tensor("float32", self.shape_y_ub,
                                                        name="data_input_ub_2", scope=tik.scope_ubuf)
        self.data_input_ub_2_fp16 = self.tik_instance.Tensor("float16", self.shape_y_ub,
                                                             name="data_input_ub_2_fp16", scope=tik.scope_ubuf)
        self.data_input_ub_2_trans = self.tik_instance.Tensor("float16", self.shape_y_ub_trans,
                                                              name="data_input_ub_2_trans", scope=tik.scope_ubuf)

    def _init_matmul_tensor_b_ub_db(self, shape_y_ub, shape_y_ub_trans, double_buffer):
        """
        double buffer strategy

        Parameters:
        -------------
        shape_y_ub: (n, k)
        shape_y_ub_trans: (n1, k, n0)
        double_buffer: support 1,2,4,8

        Returns:
        -------------
        None
        """
        self.data_input_ub_2 = self.tik_instance.Tensor("float32", self.shape_y_ub,
                                                        name="data_input_ub_2", scope=tik.scope_ubuf)
        self.data_input_ub_2_fp16_list = []
        self.data_input_ub_2_trans_list = []
        for db_idx in range(double_buffer):
            tensor_name1 = "data_input_ub_2_fp16_%d" % (db_idx + 1)
            tensor_name2 = "data_input_ub_2_trans_%d" % (db_idx + 1)
            tensor_ins1 = self.tik_instance.Tensor("float16", shape_y_ub,
                                                   name=tensor_name1, scope=tik.scope_ubuf)
            tensor_ins2 = self.tik_instance.Tensor("float16", shape_y_ub_trans,
                                                   name=tensor_name2, scope=tik.scope_ubuf)
            self.data_input_ub_2_fp16_list.append(tensor_ins1)
            self.data_input_ub_2_trans_list.append(tensor_ins2)

    def _init_tensor_ub(self):
        """ init tensor_c of matmul, normal tensor of argmin and scalar
        """
        self.matmul_output_ub = self.tik_instance.Tensor(self.input_dtype, self.shape_z_ub,
                                                         name="matmul_output_ub", scope=tik.scope_ubuf)
        self.matmul_output_ub_nd = self.tik_instance.Tensor(self.input_dtype, self.shape_z_ub_nd,
                                                            name="matmul_output_ub_nd", scope=tik.scope_ubuf)

        if self.soc_version.find("Ascend710") != -1:
            self.min_distance_ub = self.tik_instance.Tensor(self.input_dtype, (self.m_tiling, 2),
                                                            name="min_distance_ub_fp32", scope=tik.scope_ubuf)
            self.local_min_distance_ub = self.tik_instance.Tensor(self.input_dtype, (self.m_tiling,),
                                                                  name="local_min_distance_ub", scope=tik.scope_ubuf)
            self.local_min_index_ub = self.tik_instance.Tensor(self.input_dtype, (self.m_tiling,),
                                                               name="local_min_index_ub", scope=tik.scope_ubuf)
        else:
            self.ub_min_8 = self.tik_instance.Tensor(self.input_dtype, (8, 8), name="ub_min_8", scope=tik.scope_ubuf)
            self.cmp_mask_ub = self.tik_instance.Tensor(MASK_DTYPE, (64,), name="cmp_mask_ub", scope=tik.scope_ubuf)
            self.ub_index_int32 = self.tik_instance.Tensor(INDEX_DTYPE, (8, 8),
                                                           name="ub_index_int32", scope=tik.scope_ubuf)

        self.scalar_two = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=2)
        self.scalar_vd = self.tik_instance.Scalar(dtype=self.input_dtype)
        self.scalar_index_offset = self.tik_instance.Scalar(dtype=INDEX_DTYPE)
        self.global_scalar_min = self.tik_instance.Scalar(dtype=self.input_dtype)

    def _init_tensor_ub_global(self):
        """ init tensor of global domain in ub buffer
        """
        if self.soc_version.find("Ascend710") != -1:
            self.global_index_dtype = self.input_dtype
        else:
            self.global_index_dtype = INDEX_DTYPE
        self.global_min_index_ub = self.tik_instance.Tensor(self.global_index_dtype, self.shape_global_min_distance_ub,
                                                            name="global_min_index_ub", scope=tik.scope_ubuf)
        self.output_count_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_min_num,),
                                                        name="output_count_ub", scope=tik.scope_ubuf)
        self.global_min_distance_ub = self.tik_instance.Tensor(self.input_dtype, self.shape_global_min_distance_ub,
                                                               name="global_min_distance_ub", scope=tik.scope_ubuf)
        self.output_total_distance_ub = self.tik_instance.Tensor(self.input_dtype, (self.ub_min_num,),
                                                                 name="output_total_distance_ub", scope=tik.scope_ubuf)

        self.scalar_max_fp32 = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=SCALAR_MAX_FP32)
        self.scalar_zero = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=0)
        self.scalar_one = self.tik_instance.Scalar(dtype=self.input_dtype, init_value=1)

    def k_means_centroids_compute(self):
        """
        MAIN function of k-means centroids operator

        Parameters:
        -------------
        None

        Returns:
        -------------
        tik_instance: tik instance
        """
        with self.tik_instance.for_range(0, self.aic_cnt, block_num=self.aic_cnt) as blk_idx:
            self._compute_one_core(blk_idx)

        if self.use_actual_distance:
            inputs = [self.data_input_gm_1, self.data_input_gm_2,
                      self.data_input_gm_4, self.data_input_gm_3]
        else:
            inputs = [self.data_input_gm_1, self.data_input_gm_2, self.data_input_gm_4]

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=inputs,
            outputs=[self.data_output_gm_1, self.data_output_gm_2, self.data_output_gm_3],
        )

        return self.tik_instance

    def _compute_one_core(self, blk_idx):
        """
        compute in m tiling loop

        Parameters:
        -------------
        blk_idx: index of ai-core, expr

        Returns:
        -------------
        None
        """
        m_aub = self.tiling["AUB_shape"][1]
        with self.tik_instance.if_scope(tik.all(blk_idx == self.aic_cnt - 1, self.m_last_core > 0)):
            if self.m_last_tiling_loop > 0:
                with self.tik_instance.for_range(0, self.m_last_tiling_loop) as mlt_idx:
                    start_gm = self.m_each_core * blk_idx + mlt_idx * m_aub
                    self._matmul_gm_to_l0a(self.data_input_gm_1[start_gm:(start_gm + m_aub), :])
                    self._compute_one_core_n(start_gm)
            if self.m_last_tiling_left > 0:
                start_gm = self.m_each_core * blk_idx + self.m_last_tiling_loop * m_aub
                self._matmul_gm_to_l0a_tail(self.data_input_gm_1[start_gm:, :], is_last_core=True)
                self._compute_one_core_n(start_gm, is_last_core=True, is_m_tail=True)
        with self.tik_instance.else_scope():
            if self.m_tiling_loop > 0:
                with self.tik_instance.for_range(0, self.m_tiling_loop) as mt_idx:
                    start_gm = self.m_each_core * blk_idx + mt_idx * m_aub
                    self._matmul_gm_to_l0a_db(self.data_input_gm_1[start_gm:(start_gm + m_aub), :])
                    self._compute_one_core_n(start_gm)
            if self.m_tiling_left > 0:
                start_gm = self.m_each_core * blk_idx + self.m_tiling_loop * m_aub
                self._matmul_gm_to_l0a_tail(self.data_input_gm_1[start_gm:, :])
                self._compute_one_core_n(start_gm, is_m_tail=True)

    def _compute_one_core_n(self, m_gm_idx, is_last_core=False, is_m_tail=False):
        """
        compute matmul, argmin and unsorted_segment_sum in n tiling loop,
        then move result to gm

        Parameters:
        -------------
        m_gm_idx: global index of m axis in gm, expr
        is_last_core: whether is last core, bool
        is_m_tail: whether axis m has tail in each core, whose tail length
            maybe is different between the last core and the other cores, bool

        Returns:
        -------------
        None
        """
        self._init_tensor_ub_global()
        vdup_rpt = self.m_tiling // FP32_MASK
        self.tik_instance.vector_dup(FP32_MASK, self.global_min_distance_ub, self.scalar_max_fp32,
                                     vdup_rpt, 1, 8)

        n_bub = self.tiling["BUB_shape"][1]
        if self.n_tiling_loop > 0:
            with self.tik_instance.for_range(0, self.n_tiling_loop) as nt_idx:
                start_gm = nt_idx * n_bub
                self._matmul_gm_to_l0b_db(self.data_input_gm_2[start_gm:(start_gm + n_bub), :])
                self._mmad(start_gm)
        if self.n_tiling_left > 0:
            start_gm = self.n_tiling_loop * n_bub
            self._matmul_gm_to_l0b_tail(self.data_input_gm_2[start_gm:, :])
            self._mmad(start_gm, is_n_tail=True)

        self._unsorted_segment_sum(m_gm_idx, is_last_core=is_last_core, is_m_tail=is_m_tail)

    def _matmul_gm_to_l0a(self, tensor_a_gm):
        """
        move tensor_a: gm -> ub -> l1 -> l0a

        Parameters:
        -------------
        tensor_a_gm: tensor_a in gm

        Returns:
        -------------
        None
        """
        # release ub buffer of tensor_a when tensor_a moves to l1
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_a_ub()
            self.tik_instance.data_move(self.data_input_ub_1, tensor_a_gm,
                                        0, 1, self._elecnt_of_shape(self.shape_x_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0)

            self.vconv(self.data_input_ub_1_fp16, self.data_input_ub_1, self.shape_x_ub)

            self.nd_to_zn_3d(self.data_input_ub_1_trans, self.data_input_ub_1_fp16, self.shape_x_ub_trans)

            self.tik_instance.data_move(self.data_input_l1_1, self.data_input_ub_1_trans,
                                        0, 1, self._elecnt_of_shape(self.shape_x_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE,
                                        0, 0)

        self.zn_to_zz(self.data_input_l0a, self.data_input_l1_1, self.shape_x_l0a)

    def _matmul_gm_to_l0a_tail(self, tensor_a_gm, is_last_core=False):
        """
        move tensor_a_tail: gm -> ub -> l1 -> l0a

        Parameters:
        -------------
        tensor_a_gm: tensor_a_tail in gm
        is_last_core: whether is last core, bool

        Returns:
        -------------
        None
        """
        m_aub = self.m_tiling_left
        if is_last_core:
            m_aub = self.m_last_tiling_left
        k_aub = tensor_a_gm.shape[1]
        shape_x_ub = (m_aub, k_aub)
        m1_aub = self._ceil(m_aub, 16)
        shape_x_ub_trans = (m1_aub, k_aub, 16)
        k1_aub = self._ceil(k_aub, 16)
        shape_x_l0a = (m1_aub, k1_aub, 16, 16)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_a_ub()
            self.tik_instance.data_move(self.data_input_ub_1[:m_aub, :], tensor_a_gm,
                                        0, 1, self._elecnt_of_shape(shape_x_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0)

            self.vconv(self.data_input_ub_1_fp16[:m_aub, :], self.data_input_ub_1[:m_aub, :], shape_x_ub)

            self.nd_to_zn_3d(self.data_input_ub_1_trans[:m1_aub, :, :], self.data_input_ub_1_fp16[:m_aub, :],
                             shape_x_ub_trans)

            self.tik_instance.data_move(self.data_input_l1_1[:m1_aub, :, :], self.data_input_ub_1_trans[:m1_aub, :, :],
                                        0, 1, self._elecnt_of_shape(shape_x_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE,
                                        0, 0)

        self.zn_to_zz(self.data_input_l0a[:m1_aub, :, :, :], self.data_input_l1_1[:m1_aub, :, :], shape_x_l0a)

    def _matmul_gm_to_l0a_db(self, tensor_a_gm):
        """
        move tensor_a: (gm -> ub -> l1) * double_buffer -> l0a

        Parameters:
        -------------
        tensor_a_gm: tensor_a in gm

        Returns:
        -------------
        None
        """
        double_buffer = self.tiling["manual_pingpong_buffer"]["AUB_pbuffer"]
        m_aub, k_aub = tensor_a_gm.shape
        m_aub //= double_buffer
        shape_x_ub = (m_aub, k_aub)
        m1_aub = self._ceil(m_aub, 16)
        shape_x_ub_trans = (m1_aub, k_aub, 16)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_a_ub_db(shape_x_ub, shape_x_ub_trans, double_buffer)
            for idx in range(double_buffer):
                self.tik_instance.data_move(
                    self.data_input_ub_1[idx * m_aub:(idx + 1) * m_aub, :],
                    tensor_a_gm[idx * m_aub:(idx + 1) * m_aub, :],
                    0, 1, self._elecnt_of_shape(shape_x_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0
                )

                self.vconv(self.data_input_ub_1_fp16_list[idx], self.data_input_ub_1[idx * m_aub:(idx + 1) * m_aub, :],
                           shape_x_ub)

                self.nd_to_zn_3d(self.data_input_ub_1_trans_list[idx], self.data_input_ub_1_fp16_list[idx],
                                 shape_x_ub_trans)

                self.tik_instance.data_move(
                    self.data_input_l1_1[idx * m1_aub:(idx + 1) * m1_aub, :, :], self.data_input_ub_1_trans_list[idx],
                    0, 1, self._elecnt_of_shape(shape_x_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE, 0, 0
                )

        self.zn_to_zz(self.data_input_l0a, self.data_input_l1_1, self.shape_x_l0a)

    def _matmul_gm_to_l0b_tail(self, tensor_b_gm):
        """
        move tensor_b_tail: gm -> ub -> l1 -> l0b

        Parameters:
        -------------
        tensor_b_gm: tensor_b_tail in gm

        Returns:
        -------------
        None
        """
        n_bub, k_bub = tensor_b_gm.shape
        shape_y_ub = (n_bub, k_bub)
        n1_bub = self._ceil(n_bub, 16)
        shape_y_ub_trans = (n1_bub, k_bub, 16)
        k1_bub = self._ceil(k_bub, 16)
        shape_y_l0b = (k1_bub, n1_bub, 16, 16)
        # release ub buffer of tensor_b when tensor_b moves to l1
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_b_ub()
            self.tik_instance.data_move(self.data_input_ub_2[:n_bub, :], tensor_b_gm,
                                        0, 1, self._elecnt_of_shape(shape_y_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0)

            self.vconv(self.data_input_ub_2_fp16[:n_bub, :], self.data_input_ub_2[:n_bub, :], shape_y_ub)

            self.nd_to_zn_3d(self.data_input_ub_2_trans[:n1_bub, :, :], self.data_input_ub_2_fp16[:n_bub, :],
                             shape_y_ub_trans)

            self.tik_instance.data_move(self.data_input_l1_2[:n1_bub, :, :], self.data_input_ub_2_trans[:n1_bub, :, :],
                                        0, 1, self._elecnt_of_shape(shape_y_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE,
                                        0, 0)

        self.nz_to_zn(self.data_input_l0b[:, :n1_bub, :, :], self.data_input_l1_2[:n1_bub, :, :], shape_y_l0b)

    def _matmul_gm_to_l0b_db(self, tensor_b_gm):
        """
        move tensor_b: (gm -> ub -> l1) * double_buffer -> l0b

        Parameters:
        -------------
        tensor_b_gm: tensor_b in gm

        Returns:
        -------------
        None
        """
        double_buffer = self.tiling["manual_pingpong_buffer"]["BUB_pbuffer"]
        n_bub, k_bub = tensor_b_gm.shape
        n_bub //= double_buffer
        shape_y_ub = (n_bub, k_bub)
        n1_bub = self._ceil(n_bub, 16)
        shape_y_ub_trans = (n1_bub, k_bub, 16)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            self._init_matmul_tensor_b_ub_db(shape_y_ub, shape_y_ub_trans, double_buffer)
            for idx in range(double_buffer):
                self.tik_instance.data_move(
                    self.data_input_ub_2[idx * n_bub:(idx + 1) * n_bub, :],
                    tensor_b_gm[idx * n_bub:(idx + 1) * n_bub, :],
                    0, 1, self._elecnt_of_shape(shape_y_ub) * FP32_TYPE // UB_BLOCK_SIZE, 0, 0
                )

                self.vconv(self.data_input_ub_2_fp16_list[idx], self.data_input_ub_2[idx * n_bub:(idx + 1) * n_bub, :],
                           shape_y_ub)

                self.nd_to_zn_3d(self.data_input_ub_2_trans_list[idx], self.data_input_ub_2_fp16_list[idx],
                                 shape_y_ub_trans)

                self.tik_instance.data_move(
                    self.data_input_l1_2[idx * n1_bub:(idx + 1) * n1_bub, :, :], self.data_input_ub_2_trans_list[idx],
                    0, 1, self._elecnt_of_shape(shape_y_ub_trans) * FP16_TYPE // UB_BLOCK_SIZE, 0, 0
                )

        self.nz_to_zn(self.data_input_l0b, self.data_input_l1_2, self.shape_y_l0b)

    def _mmad(self, n_gm_idx, is_n_tail=False):
        """
        mmad: A x B = C

        Parameters:
        -------------
        n_gm_idx: global index of n axis in gm, expr
        is_n_tail: whether axis n has tail in each core, bool

        Returns:
        -------------
        None
        """
        self._init_tensor_ub()
        mmad_m = self.shape_x_l0a[0] * self.shape_x_l0a[2]
        mmad_k = self.shape_x_l0a[1] * self.shape_x_l0a[3]
        mmad_n = self.shape_y_l0b[1] * self.shape_y_l0b[2]
        self.tik_instance.mmad(self.data_output_l0c, self.data_input_l0a, self.data_input_l0b,
                               mmad_m, mmad_k, mmad_n, 0)

        nc_factor, mc_factor, m0, n0 = self.tiling["CUB_matrix"][:4]
        if is_n_tail:
            if self.n_tiling_cub_loop > 0:
                with self.tik_instance.for_range(0, self.n_tiling_cub_loop) as ntc_idx:
                    start_gm = n_gm_idx + ntc_idx * nc_factor * n0
                    start_l0c = ntc_idx * nc_factor
                    self._matmul_l0c_to_ub(start_l0c, self.shape_z_ub)
                    self._argmin(start_gm)
            if self.n_tiling_cub_left > 0:
                start_gm = n_gm_idx + self.n_tiling_cub_loop * nc_factor * n0
                start_l0c = self.n_tiling_cub_loop * nc_factor
                shape_z_ub = (self._ceil(self.n_tiling_cub_left, n0), mc_factor * m0, n0)
                self._matmul_l0c_to_ub(start_l0c, shape_z_ub, is_n_tail=True)
                self._argmin(start_gm, is_n_tail=True)
        else:
            with self.tik_instance.for_range(0, self.n_tiling_ub_loop) as ntu_idx:
                start_gm = n_gm_idx + ntu_idx * nc_factor * n0
                start_l0c = ntu_idx * nc_factor
                self._matmul_l0c_to_ub(start_l0c, self.shape_z_ub)
                self._argmin(start_gm)

    def _matmul_l0c_to_ub(self, n_l0c_idx, shape_z_ub, is_n_tail=False):
        """
        move tensor_c: l0c -> ub

        Parameters:
        -------------
        n_l0c_idx: local index of axis n1 in l0c, expr
        shape_z_ub: input shape of tensor_c, tuple
        is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        nc_factor, _, n0 = shape_z_ub
        self.tik_instance.data_move(self.matmul_output_ub, self.data_output_l0c[n_l0c_idx, 0, 0],
                                    0, 1, self._elecnt_of_shape(shape_z_ub) * FP32_TYPE // L0C_BLOCK_SIZE, 0, 0)
        self._nz_to_nd(self.matmul_output_ub_nd[:, :nc_factor * n0],
                       self.matmul_output_ub[:nc_factor, :, :], shape_z_ub, is_n_tail=is_n_tail)

    def _argmin(self, n_gm_idx, is_n_tail=False):
        """
        compute argmin in ub buffer

        Parameters:
        -------------
        n_gm_idx: global index of n axis in gm, expr
        is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        self.data_input_ub_4_broadcast = self.matmul_output_ub.reshape(self.shape_broadcast_ub)
        self.data_input_ub_4 = self.tik_instance.Tensor(self.input_dtype, (1, self.n_tiling),
                                                        name="data_input_ub_4", scope=tik.scope_ubuf)
        self.scalar_index_offset.set_as(n_gm_idx)
        # move sum_square_centroid from gm to ub
        mv_burst = self.tik_instance.Scalar(INDEX_DTYPE)
        if is_n_tail:
            n_tail = self.n_tiling_cub_left
            mv_burst.set_as(n_tail // self.ub_min_num)
        else:
            mv_burst.set_as(self.n_tiling // self.ub_min_num)
        with self.tik_instance.if_scope(mv_burst > 0):
            self.tik_instance.data_move(self.data_input_ub_4, self.data_input_gm_4[0, n_gm_idx],
                                        0, 1, mv_burst, 0, 0)

        self._broadcast()

        self._monocular_operator(self.matmul_output_ub_nd, self.matmul_output_ub_nd, self.scalar_two, "vmuls")

        self._binary_operator(self.data_input_ub_4_broadcast, self.data_input_ub_4_broadcast,
                              self.matmul_output_ub_nd, "vsub")

        if self.soc_version.find("Ascend710") != -1:
            self._vcmin_v200()
        else:
            self._vcmin(is_n_tail=is_n_tail)

    def _broadcast(self):
        """ broadcast sum_square_sample from (1, n) to (m, n)
        """
        vadds_repeat = min(VECTOR_REPEAT_MAX, self.m_tiling)
        vadds_repeat_tail = self.m_tiling - VECTOR_REPEAT_MAX
        mask = min(self.n_tiling, FP32_MASK)
        mask_tail = self.n_tiling - FP32_MASK
        vadds_dst_rpt_stride = self.n_tiling // 8

        self.tik_instance.vadds(mask, self.data_input_ub_4_broadcast, self.data_input_ub_4,
                                self.scalar_zero, vadds_repeat, 1, 1, vadds_dst_rpt_stride, 0)
        if mask_tail > 0:
            self.tik_instance.vadds(mask_tail, self.data_input_ub_4_broadcast[0, mask], self.data_input_ub_4[0, mask],
                                    self.scalar_zero, vadds_repeat, 1, 1, vadds_dst_rpt_stride, 0)
        if vadds_repeat_tail > 0:
            self.tik_instance.vadds(mask, self.data_input_ub_4_broadcast[VECTOR_REPEAT_MAX, 0],
                                    self.data_input_ub_4, self.scalar_zero, vadds_repeat_tail, 1, 1,
                                    vadds_dst_rpt_stride, 0)
            if mask_tail > 0:
                self.tik_instance.vadds(mask_tail, self.data_input_ub_4_broadcast[VECTOR_REPEAT_MAX, mask],
                                        self.data_input_ub_4[0, mask], self.scalar_zero, vadds_repeat_tail, 1, 1,
                                        vadds_dst_rpt_stride, 0)

    def _vcmin_v200(self):
        """ optimized vcmin v200
        """
        # Use vcmin to get the minimum value of each row in m_tiling rows
        vcmin_repeat = min(VECTOR_REPEAT_MAX, self.m_tiling)
        vcmin_repeat_tail = self.m_tiling - VECTOR_REPEAT_MAX
        vcmin_src_rpt_stride = self.n_tiling // self.ub_min_num
        self.tik_instance.vcmin(self.n_tiling, self.min_distance_ub, self.data_input_ub_4_broadcast[0, 0],
                                vcmin_repeat, 1, 1, vcmin_src_rpt_stride)

        if vcmin_repeat_tail > 0:
            self.tik_instance.vcmin(self.n_tiling, self.min_distance_ub[VECTOR_REPEAT_MAX, 0],
                                    self.data_input_ub_4_broadcast[VECTOR_REPEAT_MAX, 0],
                                    vcmin_repeat_tail, 1, 1, vcmin_src_rpt_stride)
        # Obtain the minimum and minimum indexes from the results of vcmin, respectively
        vr_repeat = (self.m_tiling * 2) // FP32_MASK
        self.tik_instance.vreduce(FP32_MASK, self.local_min_distance_ub, self.min_distance_ub,
                                  1, vr_repeat, 1, 8, 8, 0, None, "normal")
        self.tik_instance.vreduce(FP32_MASK, self.local_min_index_ub, self.min_distance_ub,
                                  2, vr_repeat, 1, 8, 8, 0, None, "normal")
        # Compare local and global minimums and update
        vmin_repeat = (self.m_tiling) // FP32_MASK
        self.tik_instance.vmin(FP32_MASK, self.global_min_distance_ub, self.local_min_distance_ub,
                               self.global_min_distance_ub, vmin_repeat, 1, 1, 1, 8, 8, 8)
        local_min_index_ub_int32 = self.local_min_index_ub.reinterpret_cast_to(INDEX_DTYPE)
        # Local minimum index adds tiling offset
        vadds_rpt = (self.m_tiling) // FP32_MASK
        self.tik_instance.vadds(FP32_MASK, local_min_index_ub_int32, local_min_index_ub_int32,
                                self.scalar_index_offset, vadds_rpt, 1, 1, 8, 8)
        self.local_min_index_ub = local_min_index_ub_int32.reinterpret_cast_to("float32")
        # Update the global minimum index
        update_loop = (self.m_tiling) // FP32_MASK
        with self.tik_instance.for_range(0, update_loop) as u_idx:
            cmp_mask = self.tik_instance.vcmp_eq(FP32_MASK, self.global_min_distance_ub[u_idx * FP32_MASK],
                                                 self.local_min_distance_ub[u_idx * FP32_MASK], 1, 1)
            self.tik_instance.vsel(FP32_MASK, 0, self.global_min_index_ub[u_idx * FP32_MASK], cmp_mask,
                                   self.local_min_index_ub[u_idx * FP32_MASK],
                                   self.global_min_index_ub[u_idx * FP32_MASK],
                                   1, 1, 1, 1, 8, 8, 8)

    def _vcmin(self, is_n_tail=False):
        """ optimized vcmin
        """
        vmin_rpt = self.tik_instance.Scalar(INDEX_DTYPE)
        # The number of rows processed at one time is 8
        row_batch = 8
        # The number of cols processed at one time is 8
        col_batch = 8
        get_min_loop = self.m_tiling // row_batch
        with self.tik_instance.for_range(0, get_min_loop) as m_idx:
            self.tik_instance.vector_dup(FP32_MASK, self.ub_min_8, SCALAR_MAX_FP32, 1, 1, 8)
            if is_n_tail:
                n_tail = self.n_tiling_cub_left
                vmin_rpt.set_as(n_tail // col_batch)
            else:
                vmin_rpt.set_as(self.n_tiling // col_batch)
            vmin_blk_stride = self.n_tiling // col_batch
            # Get the minimum 8 values in each of 8 rows at a time
            self.tik_instance.vmin(FP32_MASK, self.ub_min_8, self.data_input_ub_4_broadcast[m_idx * row_batch, 0],
                                   self.ub_min_8, vmin_rpt, 1, vmin_blk_stride, 1, 0, 1, 0)
            self._set_init_index(m_idx, row_batch, vmin_rpt, vmin_blk_stride)
            # Get the minimum value and the minimum index of the eight values in the 8 rows
            min_value = self.tik_instance.Scalar(self.input_dtype)
            min_index = self.tik_instance.Scalar(INDEX_DTYPE)
            with self.tik_instance.for_range(0, row_batch) as r_idx:
                min_value.set_as(self.ub_min_8[r_idx, 0])
                min_index.set_as(self.ub_index_int32[r_idx, 0])
                # Get the minimum value and the minimum index of the eight values in 1 row
                with self.tik_instance.for_range(1, col_batch) as c_idx:
                    min_cmp_value = self.tik_instance.Scalar(self.input_dtype)
                    min_cmp_index = self.tik_instance.Scalar(INDEX_DTYPE)
                    min_cmp_value.set_as(self.ub_min_8[r_idx, c_idx])
                    min_cmp_index.set_as(self.ub_index_int32[r_idx, c_idx])
                    with self.tik_instance.if_scope(min_cmp_value < min_value):
                        min_value.set_as(self.ub_min_8[r_idx, c_idx])
                        min_index.set_as(min_cmp_index + c_idx)
                    with self.tik_instance.if_scope(tik.all(min_cmp_value == min_value,
                                                            min_cmp_index + c_idx < min_index)):
                        min_value.set_as(self.ub_min_8[r_idx, c_idx])
                        min_index.set_as(min_cmp_index + c_idx)
                # Compare local and global minimums and update
                global_value = self.tik_instance.Scalar(self.input_dtype)
                global_index = self.tik_instance.Scalar(INDEX_DTYPE)
                global_value.set_as(self.global_min_distance_ub[m_idx * 8 + r_idx])
                global_index.set_as(self.global_min_index_ub[m_idx * 8 + r_idx])
                with self.tik_instance.if_scope(min_value < global_value):
                    self.global_min_distance_ub[m_idx * 8 + r_idx].set_as(min_value)
                    self.global_min_index_ub[m_idx * 8 + r_idx].set_as(min_index + self.scalar_index_offset)

    def _set_init_index(self, m_idx, row_batch, rpt, blk_stride):
        """
        Assign an initial value to the index with the smallest 8 values in each row

        Parameters:
        -------------
        m_idx: row index
        row_batch: Number of rows processed at a time
        rpt: repeat of vcmpv_eq
        blk_stride: block stride of vcmpv_eq

        Returns:
        -------------
        None
        """
        self.tik_instance.vcmpv_eq(self.cmp_mask_ub, self.ub_min_8,
                                    self.data_input_ub_4_broadcast[m_idx * row_batch, 0],
                                    rpt, 1, blk_stride, 0, 1)
        self.tik_instance.vector_dup(FP32_MASK, self.ub_index_int32, 0, 1, 1, 8)
        with self.tik_instance.for_range(0, rpt) as update_idx:
            index = rpt - 1 - update_idx
            mask_l = self.tik_instance.Scalar(MASK_DTYPE)
            mask_h = self.tik_instance.Scalar(MASK_DTYPE)
            mask_l.set_as(self.cmp_mask_ub[index])
            mask_h.set_as(0)
            with self.tik_instance.if_scope(mask_l != 0):
                self.tik_instance.vector_dup([mask_h, mask_l], self.ub_index_int32,
                                                index * 8, 1, 1, 8)

    def _unsorted_segment_sum(self, m_gm_idx, is_last_core=False, is_m_tail=False):
        """
        unsorted segment sum,
            sum and count distance result of each cluster

        Parameters:
        -------------
        m_gm_idx: global index of m axis in gm, expr
        is_last_core: whether is last core, bool
        is_m_tail: whether axis m has tail in each core, bool

        Returns:
        -------------
        None
        """
        shape_x1 = self.input_x1.get("shape")
        d_gm = shape_x1[1]
        cur_m = self.m_tiling
        if is_m_tail:
            if is_last_core:
                cur_m = self.m_last_tiling_left
            else:
                cur_m = self.m_tiling_left

        self.tik_instance.set_atomic_add(1)
        self.tik_instance.vector_dup(8, self.output_count_ub, self.scalar_zero, 1, 1, 8)
        self.tik_instance.vector_dup(8, self.output_total_distance_ub, self.scalar_zero, 1, 1, 8)
        self.output_count_ub[0].set_as(self.scalar_one)

        if self.use_actual_distance:
            self._calc_actual_distance(cur_m, m_gm_idx)

        self._output_loss(cur_m)

        min_index_to_gm = self.tik_instance.Scalar(dtype=INDEX_DTYPE)
        if self.soc_version.find("Ascend710") != -1:
            global_min_index_ub_int32 = self.global_min_index_ub.reinterpret_cast_to("int32")
            once_m = cur_m
            once_sample_dma_burst = (d_gm * once_m) // self.ub_min_num
            once_sample = self.tik_instance.Tensor(self.input_dtype, (once_m, d_gm),
                                                   name="once_sample", scope=tik.scope_ubuf)
            self.tik_instance.data_move(once_sample, self.data_input_gm_1[m_gm_idx, 0],
                                        0, 1, once_sample_dma_burst, 0, 0)
        else:
            global_min_index_ub_int32 = self.global_min_index_ub
            once_sample = self.tik_instance.Tensor(self.input_dtype, (d_gm,),
                                                   name="once_sample", scope=tik.scope_ubuf)
        once_sample_out_dma_burst = d_gm // self.ub_min_num
        with self.tik_instance.for_range(0, cur_m) as m_idx:
            min_index_to_gm.set_as(global_min_index_ub_int32[m_idx])
            if self.soc_version.find("Ascend710") != -1:
                self.tik_instance.data_move(self.data_output_gm_1[min_index_to_gm, 0], once_sample[m_idx, 0],
                                            0, 1, once_sample_out_dma_burst, 0, 0)
            else:
                self.tik_instance.data_move(once_sample, self.data_input_gm_1[m_gm_idx + m_idx, 0],
                                            0, 1, once_sample_out_dma_burst, 0, 0)
                self.tik_instance.data_move(self.data_output_gm_1[min_index_to_gm, 0], once_sample,
                                            0, 1, once_sample_out_dma_burst, 0, 0)
            self.tik_instance.data_move(self.data_output_gm_2[min_index_to_gm, 0], self.output_count_ub,
                                        0, 1, 1, 0, 0)
        self.tik_instance.set_atomic_add(0)

    def _calc_actual_distance(self, cur_m, m_gm_idx):
        """
        calculate the true minimum distance

        Parameters:
        -------------
        cur_m: actual size of M currently processed
        m_gm_idx: global index of m axis in gm

        Returns:
        -------------
        None
        """
        cur_m_align = self._ceil(cur_m, self.ub_min_num) * self.ub_min_num
        input_3_dma_burst = cur_m_align // self.ub_min_num
        self.shape_input_3_ub = (cur_m_align, 1)
        data_input_ub_3 = self.tik_instance.Tensor(self.input_dtype, self.shape_input_3_ub,
                                                   name="data_input_ub_3", scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_input_ub_3, self.data_input_gm_3[m_gm_idx, 0],
                                    0, 1, input_3_dma_burst, 0, 0)
        vadd_rpt = cur_m // FP32_MASK
        vadd_left = cur_m % FP32_MASK
        if vadd_rpt > 0:
            self.tik_instance.vadd(FP32_MASK, self.global_min_distance_ub, self.global_min_distance_ub,
                                   data_input_ub_3, vadd_rpt, 1, 1, 1, 8, 8, 8)
        if vadd_left > 0:
            self.tik_instance.vadd(vadd_left, self.global_min_distance_ub[vadd_rpt * FP32_MASK],
                                   self.global_min_distance_ub[vadd_rpt * FP32_MASK],
                                   data_input_ub_3[vadd_rpt * FP32_MASK], 1, 1, 1, 1, 8, 8, 8)

    def _output_loss(self, cur_m):
        """
        output the sum of the minimum distance

        Parameters:
        -------------
        cur_m: actual size of M currently processed

        Returns:
        -------------
        None
        """
        vcadd_loop = cur_m // FP32_MASK
        vcadd_left = cur_m % FP32_MASK
        if vcadd_loop > 0:
            with self.tik_instance.for_range(0, vcadd_loop) as vca_idx:
                self.tik_instance.vcadd(FP32_MASK, self.output_total_distance_ub,
                                        self.global_min_distance_ub[vca_idx * FP32_MASK],
                                        1, 1, 1, 8)
                self.tik_instance.data_move(self.data_output_gm_3,
                                            self.output_total_distance_ub,
                                            0, 1, 1, 0, 0)
        if vcadd_left > 0:
            self.tik_instance.vcadd(vcadd_left, self.output_total_distance_ub,
                                    self.global_min_distance_ub[vcadd_loop * FP32_MASK],
                                    1, 1, 1, 8)
            self.tik_instance.data_move(self.data_output_gm_3,
                                        self.output_total_distance_ub,
                                        0, 1, 1, 0, 0)

    def vconv(self, dst, src, shape_t, src_dtype="float32"):
        """
        transfer data type fp32 to fp16, vice versa

        Parameters:
        -------------
        dst: dst tensor, fp16
        src: src tensor, fp32
        shape_t: tensor shape, tuple
        src_dtype: default fp32

        Returns:
        -------------
        None
        """
        m, n = shape_t
        size = m * n
        repeat = size // FP32_MASK
        left = size % FP32_MASK
        repeat_loop = repeat // VECTOR_REPEAT_MAX
        repeat_left = repeat % VECTOR_REPEAT_MAX
        round_mode = ""
        if src_dtype == "float32":
            dst_rep_stride = 4
            src_rep_stride = 8
        else:
            dst_rep_stride = 8
            src_rep_stride = 4
        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * VECTOR_REPEAT_MAX * FP32_MASK
                self.tik_instance.vconv(FP32_MASK, round_mode, dst[offset // n, offset % n],
                                        src[offset // n, offset % n], VECTOR_REPEAT_MAX,
                                        1, 1, dst_rep_stride, src_rep_stride)

        if repeat_left > 0:
            offset = repeat_loop * VECTOR_REPEAT_MAX * FP32_MASK
            self.tik_instance.vconv(FP32_MASK, round_mode, dst[offset // n, offset % n], src[offset // n, offset % n],
                                    repeat_left, 1, 1, dst_rep_stride, src_rep_stride)

        if left > 0:
            offset = repeat * FP32_MASK
            self.tik_instance.vconv(left, round_mode, dst[offset // n, offset % n], src[offset // n, offset % n],
                                    1, 1, 1, dst_rep_stride, src_rep_stride)

    def nd_to_zn_3d(self, dst, src, shape_x_trans):
        """
        reshape tensor from format ND to Zn/Nz in ub buffer using vnchwconv

        Parameters:
        -------------
        dst: tensor (m1, k, m0), fp16
        src: tensor (m, k), fp16
        shape_x_trans: result shape, tuple

        Returns:
        -------------
        None
        """
        dst_high_half = False
        src_high_half = False
        m1 = shape_x_trans[0]
        k = shape_x_trans[1]
        repeat_times = self._elecnt_of_shape(shape_x_trans) // m1 // 256
        dst_rep_stride = 16
        src_rep_stride = 1
        with self.tik_instance.for_range(0, m1) as m1_idx:
            dst_list = [dst[m1_idx * repeat_times * 256 + 16 * i] for i in range(16)]
            src_list = [src[m1_idx * k * 16 + k * i] for i in range(16)]
            self.tik_instance.vnchwconv(dst_high_half, src_high_half, dst_list, src_list,
                                        repeat_times, dst_rep_stride, src_rep_stride)

    def zn_to_zz(self, dst, src, shape_x_l0a):
        """
        reshape tensor_a from format Zn to Zz using load2d

        Parameters:
        -------------
        dst: tensor (m1, k1, m0, k0), fp16
        src: tensor (m1, k, m0), fp16
        shape_x_l0a: result shape, tuple

        Returns:
        -------------
        None
        """
        index = 0
        repeat_times = self._elecnt_of_shape(shape_x_l0a) // 256
        dst_gap = 0
        src_stride = 1
        sid = 0
        if_transpose = True
        if self.soc_version.find("Ascend710") != -1:
            self.tik_instance.load2dv2(dst, src, index, repeat_times,
                                       dst_gap, src_stride, sid, if_transpose)
        else:
            self.tik_instance.load2dv1(dst, src, index, repeat_times,
                                       src_stride, sid, if_transpose)

    def nz_to_zn(self, dst, src, shape_y_l0b):
        """
        reshape tensor_b from format Nz to Zn using load2d

        Parameters:
        -------------
        dst: tensor (k1, n1, n0, k0), fp16
        src: tensor (n1, k, n0), fp16
        shape_y_l0b: result shape, tuple

        Returns:
        -------------
        None
        """
        k1 = shape_y_l0b[0]
        repeat_times = self._elecnt_of_shape(shape_y_l0b) // k1 // 256
        dst_gap = 0
        src_stride = k1
        sid = 0
        if_transpose = True
        if self.soc_version.find("Ascend710") != -1:
            with self.tik_instance.for_range(0, k1) as index:
                self.tik_instance.load2dv2(dst[index, :, :, :], src,
                                           index, repeat_times, dst_gap, src_stride, sid, if_transpose)
        else:
            with self.tik_instance.for_range(0, k1) as index:
                self.tik_instance.load2dv1(dst[index, :, :, :], src,
                                           index, repeat_times, src_stride, sid, if_transpose)

    def _nz_to_nd(self, dst, src, shape_z_ub, is_n_tail=False):
        """
        reshape tensor_c from format Nz to ND in ub buffer using vadds

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (n1, m, n0), fp32
        shape_z_ub: input shape, tuple
        is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        n1 = shape_z_ub[0]
        # mask belongs to [1,64] when data type is fp32
        repeat_loop = n1 * 16 // FP32_MASK
        repeat_left = n1 * 16 % FP32_MASK
        # repeat_times belongs to [1,255]
        m1_255 = shape_z_ub[1] // VECTOR_REPEAT_MAX
        m1_255_left = shape_z_ub[1] % VECTOR_REPEAT_MAX

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                if m1_255 > 0:
                    with self.tik_instance.for_range(0, m1_255) as m1_255_idx:
                        self._vadds(dst, src, 32, m1_255_idx, rpt_idx,
                                    VECTOR_REPEAT_MAX, shape_z_ub, is_n_tail=is_n_tail)
                if m1_255_left > 0:
                    self._vadds(dst, src, 32, m1_255, rpt_idx,
                                m1_255_left, shape_z_ub, is_n_tail=is_n_tail)
        if repeat_left > 0:
            if m1_255 > 0:
                with self.tik_instance.for_range(0, m1_255) as m1_255_idx:
                    self._vadds(dst, src, repeat_left // 2, m1_255_idx, repeat_loop,
                                VECTOR_REPEAT_MAX, shape_z_ub, is_n_tail=is_n_tail)
            if m1_255_left > 0:
                self._vadds(dst, src, repeat_left // 2, m1_255, repeat_loop,
                            m1_255_left, shape_z_ub, is_n_tail=is_n_tail)

    def _vadds(self, dst, src, mask, m1_255_idx, rpt_idx,
               repeat_times, shape_z_ub, is_n_tail=False):
        """
        vadds entity

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (n1, m, n0), fp32
        mask: numbers of one order, int, max 64 for fp32
        m1_255_idx: start index in m1 axis, expr, max 255
        rpt_idx: start index of repeat loop, expr
        repeat_times: repeat times, int
        shape_z_ub: input shape, tuple
        is_n_tail: whether axis n has tail in cub, bool

        Returns:
        -------------
        None
        """
        # default params
        scalar = 0
        dst_blk_stride = 2
        src_blk_stride = shape_z_ub[1] * 2
        dst_rep_stride = shape_z_ub[0] * 2
        if is_n_tail:
            dst_rep_stride = self.shape_z_ub[0] * 2
        src_rep_stride = 2

        dst_part1 = dst[m1_255_idx * VECTOR_REPEAT_MAX, rpt_idx * 64]
        src_part1 = src[rpt_idx * 4, m1_255_idx * VECTOR_REPEAT_MAX, 0]
        dst_part2 = dst[m1_255_idx * VECTOR_REPEAT_MAX, rpt_idx * 64 + 8]
        src_part2 = src[rpt_idx * 4, m1_255_idx * VECTOR_REPEAT_MAX, 8]
        self.tik_instance.vadds(mask, dst_part1, src_part1, scalar, repeat_times,
                                dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride)
        self.tik_instance.vadds(mask, dst_part2, src_part2, scalar, repeat_times,
                                dst_blk_stride, src_blk_stride, dst_rep_stride, src_rep_stride)

    def _binary_operator(self, dst, src0, src1, operator):
        """
        binary operator vsub or vadd in ub buffer

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src0: tensor (m, n), fp32
        src1: tensor (m, n), fp32
        operator: binary operator type

        Returns:
        -------------
        None
        """
        m, n = self.shape_z_ub_nd

        binary_operator_dict = {
            "vsub": self.tik_instance.vsub,
            "vadd": self.tik_instance.vadd
        }
        unit = 64  # for fp32
        func = binary_operator_dict[operator]
        size = m * n
        repeat = size // unit
        left = size % unit
        repeat_loop = repeat // VECTOR_REPEAT_MAX
        repeat_left = repeat % VECTOR_REPEAT_MAX

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * VECTOR_REPEAT_MAX * unit
                func(unit, dst[offset // n, offset % n],
                     src0[offset // n, offset % n],
                     src1[offset // n, offset % n], VECTOR_REPEAT_MAX, 1, 1, 1, 8, 8, 8)

        if repeat_left > 0:
            offset = repeat_loop * VECTOR_REPEAT_MAX * unit
            func(unit, dst[offset // n, offset % n],
                 src0[offset // n, offset % n],
                 src1[offset // n, offset % n], repeat_left, 1, 1, 1, 8, 8, 8)

        if left > 0:
            offset = repeat * unit
            func(left, dst[offset // n, offset % n],
                 src0[offset // n, offset % n],
                 src1[offset // n, offset % n], 1, 1, 1, 1, 8, 8, 8)

    def _monocular_operator(self, dst, src, scalar, operator):
        """
        monocular operator vmuls in ub buffer

        Parameters:
        -------------
        dst: tensor (m, n), fp32
        src: tensor (m, n), fp32
        scalar: scalar, fp32
        operator: monocular operator type

        Returns:
        -------------
        None
        """
        m, n = self.shape_z_ub_nd

        monocular_operator_dict = {
            "vmuls": self.tik_instance.vmuls
        }

        unit = 64  # for fp32
        func = monocular_operator_dict[operator]
        size = m * n
        repeat = size // unit
        left = size % unit
        repeat_loop = repeat // VECTOR_REPEAT_MAX
        repeat_left = repeat % VECTOR_REPEAT_MAX

        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as rpt_idx:
                offset = rpt_idx * VECTOR_REPEAT_MAX * unit
                func(unit, dst[offset // n, offset % n],
                     src[offset // n, offset % n], scalar, VECTOR_REPEAT_MAX, 1, 1, 8, 8)

        if repeat_left > 0:
            offset = repeat_loop * VECTOR_REPEAT_MAX * unit
            func(unit, dst[offset // n, offset % n],
                 src[offset // n, offset % n], scalar, repeat_left, 1, 1, 8, 8)

        if left > 0:
            offset = repeat * unit
            func(left, dst[offset // n, offset % n],
                 src[offset // n, offset % n], scalar, 1, 1, 1, 8, 8)


def _shape_check(
    x,
    y,
    sum_square_x,
    sum_square_y,
    segment_sum,
    segment_count,
    kmean_total_sum,
):
    """
    shape and dtype check

    Parameters:
    -------------
    x: dict
    y: dict
    sum_square_x: dict
    sum_square_y: dict
    segment_sum: dict
    segment_count: dict
    kmean_total_sum: dict

    Returns:
    -------------
    None
    """
    shape_x1 = x.get("shape", tuple())
    shape_x2 = y.get("shape", tuple())
    shape_x4 = sum_square_y.get("shape", tuple())
    len_shape_x1 = len(shape_x1)
    len_shape_x2 = len(shape_x2)
    len_shape_x4 = len(shape_x4)

    if sum_square_x:
        # sum_square_x maybe is None
        shape_x3 = sum_square_x.get("shape", tuple())
        len_shape_x3 = len(shape_x3)
        if len_shape_x3 != INPUT_LENGTH:
            reason = ("Shape length of sum_square_x must be equal to 2, " +
                      "but recently is %d." % len_shape_x3)
            error_manager_cube.raise_err_message_cube("k_means_centroids", reason)

    if len_shape_x1 != INPUT_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape length of x must be equal to 2, " +
                                                  "but recently is %d." % len_shape_x1)
    if len_shape_x2 != INPUT_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape length of y must be equal to 2, " +
                                                  "but recently is %d." % len_shape_x2)
    if len_shape_x4 != INPUT_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Shape length of sum_square_y must be equal to 2, " +
                                                  "but recently is %d." % len_shape_x4)
    if shape_x1[1] != VECTOR_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Dimension of each sample must be 128, " +
                                                  "but recently is %d." % shape_x1[1])
    if shape_x2[1] != VECTOR_LENGTH:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  "Dimension of each centroid must be 128, " +
                                                  "but recently is %d." % shape_x2[1])

    support_dtype = ("float32",)
    input1_dtype = x.get("dtype")
    input2_dtype = y.get("dtype")
    input4_dtype = sum_square_y.get("dtype")
    output1_dtype = segment_sum.get("dtype")
    output2_dtype = segment_count.get("dtype")
    output3_dtype = kmean_total_sum.get("dtype")

    if sum_square_x:
        input3_dtype = sum_square_x.get("dtype")
        if input3_dtype not in support_dtype:
            reason = (("Input3 dtype only support %s, " % (support_dtype,)) +
                      ("but recently is %s." % input3_dtype))
            error_manager_cube.raise_err_message_cube("k_means_centroids", reason)

    if input1_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Input1 dtype only support %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % input1_dtype))
    if input2_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Input2 dtype only support %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % input2_dtype))
    if input4_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Input4 dtype only support %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % input4_dtype))
    if output1_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Output1 dtype only support %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % output1_dtype))
    if output2_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Output2 dtype only support %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % output2_dtype))
    if output3_dtype not in support_dtype:
        error_manager_cube.raise_err_message_cube("k_means_centroids",
                                                  ("Output3 dtype only support %s, " % (support_dtype,)) +
                                                  ("but recently is %s." % output3_dtype))


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.KERNEL_NAME,
)
def k_means_centroids(
    x,
    y,
    sum_square_y,
    sum_square_x,
    segment_sum,
    segment_count,
    kmean_total_sum,
    use_actual_distance=False,
    kernel_name="k_means_centroids",
):
    """
    algorithm k_means_centroids

    Parameters:
    -------------
    x: dict
        samples, shape (m, d), fp32

    y: dict
        centroids, shape (n, d), fp32

    sum_square_x: dict
        sum of squares of samples, shape (m, 1), fp32

    sum_square_y: dict
        sum of squares of centroids, shape (1, n), fp32

    segment_sum: dict
        sum of distance result in each cluster, shape (n, d), fp32

    segment_count: dict
        count of distance result in each cluster, shape (n,), fp32

    kmean_total_sum: dict
        sum of all samples' distance to centroids, shape (1,), fp32

    use_actual_distance: bool
        whether to use actual distance

    kernel_name: str

    Returns:
    -------------
    tik_instance: tik instance
    """
    _shape_check(x, y, sum_square_x, sum_square_y, segment_sum, segment_count,
                 kmean_total_sum)

    kmeans = KMeansCentroids(x, y, sum_square_x, sum_square_y, segment_sum, segment_count,
                             kmean_total_sum, use_actual_distance, kernel_name)

    tik_instance = kmeans.k_means_centroids_compute()

    return tik_instance
