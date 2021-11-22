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
dynamic bounding_box_decode
"""
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl import constant_util as constant

# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUMBER_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_EIGHT = 8
    # the number of blocks skipped per repeat
    STRIDE_FOUR = 4
    # the number of blocks skipped per repeat
    STRIDE_ONE = 1
    # the number of blocks per transposition
    LIST_NUMBER = 16
    # the number of transposes per repeat
    NUMBER_TWO = 2
    # max int64
    MAX_INT64 = 2 ** 63 - 1
    # ting param num
    TILING_ARG_NUM = 16


# 'pylint: disable=useless-object-inheritance,too-many-instance-attributes
class BoundingBoxDecode(object):
    """
       Function: use to store BoundingBoxDecode base parameters
    """

    # 'pylint: disable=too-many-arguments,invalid-name
    def __init__(self, rois, deltas, means, stds, max_shape, wh_ratio_clip,
                 kernel_name):
        """
        Init BoundingBoxDecode base parameters

        Parameters
        ----------
        rois : dict
            shape and dtype of input rois
        deltas : dict
            shape and dtype of input deltas
        means : list
            the result of the calculation is normalized, default is [0,0,0,0]
        stds : list
            the result of the calculation is normalized, default is [1,1,1,1]
        max_shape : list or tuple
            max_shape of bboxes, default is None
        wh_ratio_clip : scalar
            limit the size of deltas[:,4] and deltas[:,3] between negative
            wh_ratio_clip and positive wh_ratio_clip, default is 0.016
        kernel_name : str
            kernel name, default value is "bounding_box_decode"

        Returns
        -------
        None
        """
        MAX_UB_ELEMENT_NUMBER_FP16 = 8704
        MAX_UB_ELEMENT_NUMBER_FP32 = 4608
        BYTE_SIZE = 8
        BLOCK_NUMBER_FP16 = 32
        BLOCK_NUMBER_FP32 = 64
        self.tik_instance = tik.Tik()

        self.wh_ratio_clip = wh_ratio_clip
        self.max_shape = max_shape
        self.means = means
        self.stds = stds
        self.rois_dtype = rois.get("dtype").lower()
        self.deltas_dtype = deltas.get("dtype").lower()
        self.kernel_name = kernel_name

        self.rois_dtype_bytes_size = tbe_platform.get_bit_len(self.rois_dtype) // BYTE_SIZE
        self.rois_data_each_block = constant.BLOCK_SIZE // self.rois_dtype_bytes_size
        self.core_num = 32
        self.each_repeat_block_number = BLOCK_NUMBER_FP16
        self.ub_max_size = MAX_UB_ELEMENT_NUMBER_FP16
        if self.rois_dtype == "float32":
            self.each_repeat_block_number = BLOCK_NUMBER_FP32
            self.ub_max_size = MAX_UB_ELEMENT_NUMBER_FP32

        # init gm data
        self.rois_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                name="rois_gm", scope=tik.scope_gm)
        self.deltas_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                  name="deltas_gm", scope=tik.scope_gm)
        self.bboxes_out_gm = self.tik_instance.Tensor(self.rois_dtype, [Constant.MAX_INT64],
                                                      name="bboxes_out_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)
        # init tiling data
        self.each_core_start_address = self.tik_instance.Scalar("int32", name="each_core_start_address")
        self.loop_cycle = self.tik_instance.Scalar("int32", name="loop_cycle")
        self.start_block_addrss = self.tik_instance.Scalar("int32", name="start_block_addrss")
        self.block_number = self.tik_instance.Scalar("int32", name="block_number")
        self.repeat_times = self.tik_instance.Scalar("int32", name="repeat_times")

    def set_meanstds_scalar(self, means, stds):
        """
        init means and stds scalar and assign value to it

        Parameters
        ----------

        Returns
        -------
        scalar_list: the scalar tuple of means and stds
        """
        dtype = "float16"
        # set means value [0, 0, 0, 0]
        means_zero_scalar = self.tik_instance.Scalar(
            dtype, name="means_zero_scalar")
        means_zero_scalar.set_as(means[0])
        means_one_scalar = self.tik_instance.Scalar(
            dtype, name="means_one_scalar")
        means_one_scalar.set_as(means[1])
        means_two_scalar = self.tik_instance.Scalar(
            dtype, name="means_two_scalar")
        means_two_scalar.set_as(means[2])
        means_three_scalar = self.tik_instance.Scalar(
            dtype, name="means_three_scalar")
        means_three_scalar.set_as(means[3])
        # set stds value [1, 1, 1, 1]
        stds_zero_scalar = self.tik_instance.Scalar(
            dtype, name="stds_zero_scalar")
        stds_zero_scalar.set_as(stds[0])
        stds_one_scalar = self.tik_instance.Scalar(
            dtype, name="stds_one_scalar")
        stds_one_scalar.set_as(stds[1])
        stds_two_scalar = self.tik_instance.Scalar(
            dtype, name="stds_two_scalar")
        stds_two_scalar.set_as(stds[2])
        stds_three_scalar = self.tik_instance.Scalar(
            dtype, name="stds_three_scalar")
        stds_three_scalar.set_as(stds[3])
        scalar_list = (means_zero_scalar, means_one_scalar, means_two_scalar,
                       means_three_scalar, stds_zero_scalar, stds_one_scalar,
                       stds_two_scalar, stds_three_scalar)
        return scalar_list

    def _tiling_args(self, tiling_ub):
        """
        get runtime tiling params from tiling

        Parameters
        ----------

        Returns
        -------
        None
        """
        # read tiling int32 scalar
        self.each_core_start_address.set_as(tiling_ub[0])
        self.loop_cycle.set_as(tiling_ub[1])
        self.start_block_addrss.set_as(tiling_ub[2])
        self.block_number.set_as(tiling_ub[3])
        self.repeat_times.set_as(tiling_ub[4])

    # 'pylint: disable=too-many-arguments
    def calculate_denorm_delta(self, scalar_list, deltas_src_ub, deltas_dst_ub,
                               denorm_detal_dst_ub, repeat_times):
        """
        calculate denorm_delta using formula:
        dx = delta[..., 0]*target_stds[0] + target_means[0]
        dy = delta[..., 1]*target_stds[1] + target_means[1]
        dw = delta[..., 2]*target_stds[2] + target_means[2]
        dh = delta[..., 3]*target_stds[3] + target_means[3]

        Parameters
        ----------
        scalar_list: the scalar list of means and stds
        deltas_src_ub: the ub tensor with deltas
        deltas_dst_ub: the temporary ub tensor for calculation
        denorm_detal_dst_ub: the ub tensor with denorm_delta
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_detal_dst_ub: the ub tensor with denorm_delta
        """
        deltas_src_ub_16 = deltas_src_ub[16]
        deltas_src_ub_32 = deltas_src_ub[32]
        deltas_src_ub_48 = deltas_src_ub[48]
        self.tik_instance.vmuls(
            constant.MASK128, deltas_src_ub, deltas_dst_ub,
            scalar_list[4], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_detal_dst_ub, deltas_src_ub,
            scalar_list[0], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vmuls(
            constant.MASK128, deltas_src_ub_16, deltas_dst_ub[16],
            scalar_list[5], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_detal_dst_ub[16], deltas_src_ub_16,
            scalar_list[1], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vmuls(
            constant.MASK128, deltas_src_ub_32, deltas_dst_ub[32],
            scalar_list[6], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_detal_dst_ub[32], deltas_src_ub_32,
            scalar_list[2], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vmuls(
            constant.MASK128, deltas_src_ub_48, deltas_dst_ub[48],
            scalar_list[7], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_detal_dst_ub[48], deltas_src_ub_48,
            scalar_list[3], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        return denorm_detal_dst_ub

    def clamp_denorm_detal(self, denorm_detal_dst_ub, repeat_times):
        """
        clamp denorm_delta using formula:
        max_ratio = abs(log(max_ratio))
        dw = -max_ratio<= dw <= max_ratio
        dh = -max_ratio<= dh <= max_ratio

        Parameters
        ----------
        denorm_detal_dst_ub: the ub tensor with denorm_delta
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_detal_dst_ub: the ub tensor with denorm_delta
        """
        denorm_detal_dst_ub_48 = denorm_detal_dst_ub[48]
        denorm_detal_dst_ub_32 = denorm_detal_dst_ub[32]
        max_ratio_fp16_ub = \
            self.tik_instance.Tensor("float16",
                                     (self.ub_max_size / Constant.NUMBER_FOUR,),
                                     name="max_ratio_fp16_ub",
                                     scope=tik.scope_ubuf)

        self.tik_instance.vector_dup(constant.MASK128, max_ratio_fp16_ub,
                                     self.wh_ratio_clip, repeat_times,
                                     Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
        self.tik_instance.vln(constant.MASK128, max_ratio_fp16_ub,
                              max_ratio_fp16_ub, repeat_times, Constant.STRIDE_ONE,
                              Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)
        self.tik_instance.vabs(constant.MASK128, max_ratio_fp16_ub,
                               max_ratio_fp16_ub, repeat_times, Constant.STRIDE_ONE,
                               Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)

        self.tik_instance.vmin(constant.MASK128, denorm_detal_dst_ub_32,
                               denorm_detal_dst_ub_32, max_ratio_fp16_ub,
                               repeat_times, Constant.STRIDE_FOUR * repeat_times,
                               Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                               Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
        self.tik_instance.vmin(constant.MASK128, denorm_detal_dst_ub_48,
                               denorm_detal_dst_ub_48, max_ratio_fp16_ub,
                               repeat_times, Constant.STRIDE_FOUR * repeat_times,
                               Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                               Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
        self.tik_instance.vmuls(
            constant.MASK128, max_ratio_fp16_ub, max_ratio_fp16_ub, -1.0,
            repeat_times, Constant.STRIDE_ONE, Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_EIGHT)
        self.tik_instance.vmax(constant.MASK128, denorm_detal_dst_ub_32,
                               denorm_detal_dst_ub_32, max_ratio_fp16_ub,
                               repeat_times, Constant.STRIDE_FOUR * repeat_times,
                               Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                               Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
        self.tik_instance.vmax(constant.MASK128, denorm_detal_dst_ub_48,
                               denorm_detal_dst_ub_48, max_ratio_fp16_ub,
                               repeat_times, Constant.STRIDE_FOUR * repeat_times,
                               Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                               Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)

        return denorm_detal_dst_ub

    def calculate_denorm_rois(self, rois_src_ub, rois_dst_ub,
                              denorm_rois_dst_ub, repeat_times):
        """
        calculate denorm_rois using formula:
        px = (rois[..., 2] + rois[..., 0])*0.5
        py = (rois[..., 3] + rois[..., 1])*0.5
        pw = rois[..., 2] - rois[..., 0] + 1
        ph = rois[..., 3] - rois[..., 1] + 1

        Parameters
        ----------
        rois_src_ub: the ub tensor with rois
        rois_dst_ub: the temporary ub tensor for calculation
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        """
        rois_src_ub_32 = rois_src_ub[32]
        rois_dst_ub_32 = rois_dst_ub[32]
        rois_src_ub_48 = rois_src_ub[48]
        rois_src_ub_16 = rois_src_ub[16]
        rois_dst_ub_16 = rois_dst_ub[16]
        rois_dst_ub_48 = rois_dst_ub[48]

        # calculate denorm_rois_dst_ub == (px, py, pw, ph) ==>(px, py)
        self.tik_instance.vadd(
            constant.MASK128, rois_src_ub, rois_dst_ub, rois_dst_ub_32,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vmuls(
            constant.MASK128, denorm_rois_dst_ub, rois_src_ub, 0.5,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vadd(
            constant.MASK128, rois_src_ub_16, rois_dst_ub_16, rois_dst_ub_48,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vmuls(
            constant.MASK128, denorm_rois_dst_ub[16], rois_src_ub_16, 0.5,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        # calculate denorm_rois_dst_ub == (px, py, pw, ph) ==>(pw, ph)
        self.tik_instance.vsub(
            constant.MASK128, rois_src_ub_32, rois_dst_ub_32, rois_dst_ub,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_rois_dst_ub[32], rois_src_ub_32, 1.0,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vsub(
            constant.MASK128, rois_src_ub_48, rois_dst_ub_48, rois_dst_ub_16,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_rois_dst_ub[48], rois_src_ub_48, 1.0,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        return denorm_rois_dst_ub

    def addcmul_demorm_rois(self, denorm_detal_dst_ub, denorm_rois_dst_ub,
                            repeat_times):
        """
        addcmul denorm_rois using formula:
        gx = dx * pw + px
        gy = dy * ph + py
        gw = exp(dw)*pw
        gh = exp(dh)*ph

        Parameters
        ----------
        denorm_detal_dst_ub: the ub tensor with denorm_rois
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_detal_dst_ub: the ub tensor with denorm_rois
        """
        denorm_detal_dst_ub_48 = denorm_detal_dst_ub[48]
        denorm_rois_dst_ub_32 = denorm_rois_dst_ub[32]
        denorm_rois_dst_ub_48 = denorm_rois_dst_ub[48]
        denorm_detal_dst_ub_32 = denorm_detal_dst_ub[32]
        denorm_detal_dst_ub_16 = denorm_detal_dst_ub[16]
        # calculate denorm_rois_dst_ub == (gx, gy, gw, gh) ==>gw, gh
        self.tik_instance.vexp(
            constant.MASK128, denorm_detal_dst_ub_32, denorm_detal_dst_ub_32,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vmul(
            constant.MASK128, denorm_detal_dst_ub_32, denorm_detal_dst_ub_32,
            denorm_rois_dst_ub_32, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vexp(
            constant.MASK128, denorm_detal_dst_ub_48, denorm_detal_dst_ub_48,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vmul(
            constant.MASK128, denorm_detal_dst_ub_48, denorm_detal_dst_ub_48,
            denorm_rois_dst_ub_48, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        # calculate denorm_rois_dst_ub == (gx, gy, gw, gh) ==>gx, gy
        self.tik_instance.vmul(
            constant.MASK128, denorm_detal_dst_ub, denorm_detal_dst_ub,
            denorm_rois_dst_ub_32, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadd(
            constant.MASK128, denorm_detal_dst_ub, denorm_detal_dst_ub,
            denorm_rois_dst_ub, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vmul(
            constant.MASK128, denorm_detal_dst_ub_16, denorm_detal_dst_ub_16,
            denorm_rois_dst_ub_48, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadd(
            constant.MASK128, denorm_detal_dst_ub_16, denorm_detal_dst_ub_16,
            denorm_rois_dst_ub[16], repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        return denorm_detal_dst_ub

    def calculate_result(self, denorm_rois_dst_ub, denorm_detal_dst_ub,
                         repeat_times):
        """
        calculate the result using formula:
        x1 = gx - gw*0.5 + 0.5
        y1 = gy - gh*0.5 + 0.5
        x2 = gx + gw*0.5 - 0.5
        y2 = gy + gh*0.5 - 0.5

        Parameters
        ----------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        denorm_detal_dst_ub: the ub tensor with denorm_detal
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        """
        denorm_rois_dst_ub_32 = denorm_rois_dst_ub[32]
        denorm_detal_dst_ub_48 = denorm_detal_dst_ub[48]
        denorm_rois_dst_ub_16 = denorm_rois_dst_ub[16]
        denorm_rois_dst_ub_48 = denorm_rois_dst_ub[48]
        denorm_detal_dst_ub_32 = denorm_detal_dst_ub[32]
        denorm_detal_dst_ub_16 = denorm_detal_dst_ub[16]
        # calculate (x1, y1, x2, y2) ==>x1, y1
        self.tik_instance.vmuls(
            constant.MASK128, denorm_rois_dst_ub, denorm_detal_dst_ub_32,
            -0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadd(
            constant.MASK128, denorm_rois_dst_ub, denorm_detal_dst_ub,
            denorm_rois_dst_ub, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_rois_dst_ub, denorm_rois_dst_ub, 0.5,
            repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vmuls(
            constant.MASK128, denorm_rois_dst_ub_16, denorm_detal_dst_ub_48,
            -0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadd(
            constant.MASK128, denorm_rois_dst_ub_16, denorm_detal_dst_ub_16,
            denorm_rois_dst_ub_16, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_rois_dst_ub_16, denorm_rois_dst_ub_16,
            0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        # calculate (x1, y1, x2, y2) ==>x2, y2
        self.tik_instance.vmuls(
            constant.MASK128, denorm_rois_dst_ub_32, denorm_detal_dst_ub_32,
            0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadd(
            constant.MASK128, denorm_rois_dst_ub_32, denorm_detal_dst_ub,
            denorm_rois_dst_ub_32, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_rois_dst_ub_32, denorm_rois_dst_ub_32,
            -0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        self.tik_instance.vmuls(
            constant.MASK128, denorm_rois_dst_ub_48, denorm_detal_dst_ub_48,
            0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadd(
            constant.MASK128, denorm_rois_dst_ub_48, denorm_detal_dst_ub_16,
            denorm_rois_dst_ub_48, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR,
            Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)
        self.tik_instance.vadds(
            constant.MASK128, denorm_rois_dst_ub_48, denorm_rois_dst_ub_48,
            -0.5, repeat_times, Constant.STRIDE_FOUR * repeat_times,
            Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_FOUR, Constant.STRIDE_FOUR)

        return denorm_rois_dst_ub

    def clamp_result(self, denorm_rois_dst_ub, repeat_times):
        """
        clamp the result using formula if max_shape is not none:
        x1 = 0 <= x1 <= max_shape[1] - 1
        y1 = 0 <= y1 <= max_shape[0] - 1
        x2 = 0 <= x2 <= max_shape[1] - 1
        y2 = 0 <= y2 <= max_shape[0] - 1

        Parameters
        ----------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        repeat_times: the vector calculation repeat times

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor with denorm_rois
        """
        denorm_rois_dst_ub_16 = denorm_rois_dst_ub[16]
        denorm_rois_dst_ub_32 = denorm_rois_dst_ub[32]
        denorm_rois_dst_ub_48 = denorm_rois_dst_ub[48]
        if self.max_shape is not None:
            max_shape_one_ub = \
                self.tik_instance.Tensor("float16",
                                         (self.ub_max_size / Constant.NUMBER_FOUR,),
                                         name="max_shape_one_ub",
                                         scope=tik.scope_ubuf)
            max_shape_ub = \
                self.tik_instance.Tensor("float16",
                                         (self.ub_max_size / Constant.NUMBER_FOUR,),
                                         name="max_shape_ub",
                                         scope=tik.scope_ubuf)
            zero_ub = \
                self.tik_instance.Tensor("float16",
                                         (self.ub_max_size / Constant.NUMBER_FOUR,),
                                         name="zero_ub",
                                         scope=tik.scope_ubuf)

            self.tik_instance.vector_dup(constant.MASK128, max_shape_one_ub,
                                         self.max_shape[0] - 1, repeat_times,
                                         Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
            self.tik_instance.vector_dup(constant.MASK128, max_shape_ub,
                                         self.max_shape[1] - 1, repeat_times,
                                         Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
            self.tik_instance.vector_dup(constant.MASK128, zero_ub, 0.0,
                                         repeat_times, Constant.STRIDE_ONE, Constant.STRIDE_EIGHT)
            self.tik_instance.vmin(constant.MASK128, denorm_rois_dst_ub,
                                   denorm_rois_dst_ub, max_shape_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmin(constant.MASK128, denorm_rois_dst_ub_16,
                                   denorm_rois_dst_ub_16, max_shape_one_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmin(constant.MASK128, denorm_rois_dst_ub_32,
                                   denorm_rois_dst_ub_32, max_shape_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmin(constant.MASK128, denorm_rois_dst_ub_48,
                                   denorm_rois_dst_ub_48, max_shape_one_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmax(constant.MASK128, denorm_rois_dst_ub,
                                   denorm_rois_dst_ub, zero_ub,
                                   repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmax(constant.MASK128, denorm_rois_dst_ub_16,
                                   denorm_rois_dst_ub_16, zero_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmax(constant.MASK128, denorm_rois_dst_ub_32,
                                   denorm_rois_dst_ub_32, zero_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)
            self.tik_instance.vmax(constant.MASK128, denorm_rois_dst_ub_48,
                                   denorm_rois_dst_ub_48, zero_ub,
                                   repeat_times, Constant.STRIDE_FOUR * repeat_times,
                                   Constant.STRIDE_FOUR * repeat_times, Constant.STRIDE_ONE,
                                   Constant.STRIDE_FOUR, Constant.STRIDE_FOUR, Constant.STRIDE_EIGHT)

        return denorm_rois_dst_ub

    def data_move_mte2_function(self, loop_input, block_num):
        """
        move data of rois/deltas from gm to ub with each pinpang of each core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        block_num: the number of block of ub data

        Returns
        -------
        rois_src_ub: the ub tensor with rois
        deltas_src_ub: the ub tensor with deltas
        """
        rois_src_ub = self.tik_instance.Tensor(
            self.rois_dtype, (self.ub_max_size,),
            name="rois_src_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(rois_src_ub, self.rois_gm[loop_input],
                                    constant.SID, constant.DEFAULT_NBURST,
                                    block_num, constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)
        deltas_src_ub = self.tik_instance.Tensor(
            self.deltas_dtype, (self.ub_max_size,),
            name="deltas_src_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(deltas_src_ub, self.deltas_gm[loop_input],
                                    constant.SID, constant.DEFAULT_NBURST,
                                    block_num, constant.STRIDE_ZERO,
                                    constant.STRIDE_ZERO)

        return rois_src_ub, deltas_src_ub

    def data_move_mte3_function(self, loop_input, block_num,
                                denorm_rois_dst_ub):
        """
        move output data of bboxes from gm to ub with each pinpang of each core

        Parameters
        ----------
        loop_input: the starting address of the gm data
        block_num: the number of block of ub data
        denorm_rois_dst_ub: the ub tensor of output data of bboxes

        Returns
        -------
        None
        """
        self.tik_instance.data_move(self.bboxes_out_gm[loop_input],
                                    denorm_rois_dst_ub, constant.SID,
                                    constant.DEFAULT_NBURST, block_num,
                                    constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    # 'pylint: disable=too-many-locals
    def bounding_box_decode_compute(self, scalar_list, repeat_times,
                                    rois_src_ub, deltas_src_ub):
        """
        describe the bounding_box_decode calculation process

        Parameters
        ----------
        scalar_list: the scalar list of means and stds
        repeat_times: the vector calculation repeat times
        rois_src_ub: the ub tensor with rois
        deltas_src_ub: the ub tensor with deltas

        Returns
        -------
        denorm_rois_dst_ub: the ub tensor for output data of bboxes
        """
        if self.rois_dtype == "float32":
            deltas_src_ub_vconv = \
                self.tik_instance.Tensor("float16", (self.ub_max_size,),
                                         name="deltas_src_ub_vconv",
                                         scope=tik.scope_ubuf)

            rois_src_ub_vconv = \
                self.tik_instance.Tensor("float16", (self.ub_max_size,),
                                         name="rois_src_ub_vconv",
                                         scope=tik.scope_ubuf)

            self.tik_instance.vconv(constant.MASK64, '', rois_src_ub_vconv,
                                    rois_src_ub, repeat_times * Constant.STRIDE_EIGHT,
                                    Constant.STRIDE_ONE, Constant.STRIDE_ONE, Constant.STRIDE_FOUR,
                                    Constant.STRIDE_EIGHT)
            self.tik_instance.vconv(constant.MASK64, '', deltas_src_ub_vconv,
                                    deltas_src_ub, repeat_times * Constant.STRIDE_EIGHT,
                                    Constant.STRIDE_ONE, Constant.STRIDE_ONE, Constant.STRIDE_FOUR,
                                    Constant.STRIDE_EIGHT)
        else:
            deltas_src_ub_vconv = deltas_src_ub
            rois_src_ub_vconv = rois_src_ub

        deltas_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="deltas_dst_ub",
            scope=tik.scope_ubuf)
        rois_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="rois_dst_ub",
            scope=tik.scope_ubuf)
        denorm_rois_dst_ub = self.tik_instance.Tensor(
            "float16", (self.ub_max_size,),
            name="denorm_rois_dst_ub",
            scope=tik.scope_ubuf)
        denorm_detal_dst_ub = \
            self.tik_instance.Tensor("float16", (self.ub_max_size,),
                                     name="denorm_detal_dst_ub",
                                     scope=tik.scope_ubuf)

        # transform rois and deltas data with 16x16
        rois_src_list = [
            rois_src_ub_vconv[Constant.LIST_NUMBER * i] for i in range(Constant.LIST_NUMBER)
        ]
        rois_dst_list = [
            rois_dst_ub[Constant.LIST_NUMBER * i] for i in range(Constant.LIST_NUMBER)
        ]
        deltas_src_list = [
            deltas_src_ub_vconv[Constant.LIST_NUMBER * i] for i in range(Constant.LIST_NUMBER)
        ]
        deltas_dst_list = [
            deltas_dst_ub[Constant.LIST_NUMBER * i] for i in range(Constant.LIST_NUMBER)
        ]
        self.tik_instance.vnchwconv(True, True, rois_dst_list, rois_src_list,
                                    repeat_times * Constant.NUMBER_TWO, Constant.LIST_NUMBER,
                                    Constant.LIST_NUMBER)
        self.tik_instance.vnchwconv(True, True, deltas_dst_list,
                                    deltas_src_list, repeat_times * Constant.NUMBER_TWO,
                                    Constant.LIST_NUMBER, Constant.LIST_NUMBER)

        denorm_detal_dst_ub = self.calculate_denorm_delta(
            scalar_list, deltas_src_ub_vconv, deltas_dst_ub,
            denorm_detal_dst_ub, repeat_times)
        denorm_detal_dst_ub = self.clamp_denorm_detal(denorm_detal_dst_ub,
                                                      repeat_times)
        denorm_rois_dst_ub = self.calculate_denorm_rois(
            rois_src_ub_vconv, rois_dst_ub, denorm_rois_dst_ub, repeat_times)
        denorm_detal_dst_ub = self.addcmul_demorm_rois(
            denorm_detal_dst_ub, denorm_rois_dst_ub, repeat_times)
        denorm_rois_dst_ub = self.calculate_result(
            denorm_rois_dst_ub, denorm_detal_dst_ub, repeat_times)
        denorm_rois_dst_ub = self.clamp_result(denorm_rois_dst_ub,
                                               repeat_times)

        res_list = [
            denorm_rois_dst_ub[Constant.LIST_NUMBER * i] for i in range(Constant.LIST_NUMBER)
        ]
        self.tik_instance.vnchwconv(True, True, res_list, res_list,
                                    repeat_times * Constant.NUMBER_TWO, Constant.LIST_NUMBER,
                                    Constant.LIST_NUMBER)

        if self.rois_dtype == "float32":
            self.tik_instance.vconv(constant.MASK64, '', rois_src_ub,
                                    denorm_rois_dst_ub,
                                    repeat_times * Constant.STRIDE_EIGHT, Constant.STRIDE_ONE,
                                    Constant.STRIDE_ONE, Constant.STRIDE_EIGHT, Constant.STRIDE_FOUR)
            denorm_rois_dst_ub = rois_src_ub

        return denorm_rois_dst_ub


    # 'pylint: disable=invalid-name
    def calculation_process(self, block_id):
        """
        decide whether to enable pingpang according to different loop_cycle of
        the core

        Parameters
        ----------
        block_id: identifies the number of cores

        Returns
        -------
        None
        """
        THREAD_NUM = 2
        scalar_list = self.set_meanstds_scalar(self.means, self.stds)
        with self.tik_instance.if_scope(self.loop_cycle == 1):
            loop_input = block_id * self.each_core_start_address
            rois_src_ub, deltas_src_ub = self.data_move_mte2_function(
                loop_input, self.block_number)
            denorm_rois_dst_ub = \
                self.bounding_box_decode_compute(scalar_list,
                                                 self.repeat_times,
                                                 rois_src_ub,
                                                 deltas_src_ub)

            self.data_move_mte3_function(loop_input, self.block_number,
                                         denorm_rois_dst_ub)
        with self.tik_instance.else_scope():
            loop_input = block_id * self.each_core_start_address
            with self.tik_instance.for_range(0, self.loop_cycle, thread_num=THREAD_NUM) as cycle:
                loop_input = loop_input + cycle * self.start_block_addrss * self.rois_data_each_block
                rois_src_ub, deltas_src_ub = self.data_move_mte2_function(
                    loop_input, self.block_number)
                denorm_rois_dst_ub = self.bounding_box_decode_compute(
                    scalar_list, self.repeat_times, rois_src_ub, deltas_src_ub)
                self.data_move_mte3_function(loop_input, self.block_number, denorm_rois_dst_ub)

    def tik_instance_function(self):
        """
        the entry of bounding_box_decode calculation

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            with self.tik_instance.new_stmt_scope():
                tiling_ub = self.tik_instance.Tensor("int32", (Constant.TILING_ARG_NUM,),
                                                     name="tiling_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 2, 0, 0)
                self._tiling_args(tiling_ub)
            self.calculation_process(block_id)
        opt_config = {"out_of_bound_sync_check":True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.rois_gm, self.deltas_gm],
                                   outputs=[self.bboxes_out_gm],
                                   flowtable=[self.tiling_gm], config=opt_config)
        tbe_context.get_context().add_compile_info("vars",
                                                   {"core_num": self.core_num,
                                                    "rois_data_each_block": self.rois_data_each_block,
                                                    "each_repeat_block_number": self.each_repeat_block_number,
                                                    "ub_max_size": self.ub_max_size})
        return self.tik_instance


# 'pylint: disable=too-many-arguments, too-many-instance-attributes
# 'pylint: disable=unused-argument, too-many-locals, too-many-lines
@register_operator("BoundingBoxDecode")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_LIST_FLOAT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def bounding_box_decode(rois,
                        deltas,
                        bboxes,
                        means=(0.0, 0.0, 0.0, 0.0),
                        stds=(1.0, 1.0, 1.0, 1.0),
                        max_shape=None,
                        wh_ratio_clip=0.016,
                        kernel_name="bounding_box_decode"):
    """
    calculating data

    Parameters
    ----------
    rois : dict
        shape and dtype of input rois
    deltas : dict
        shape and dtype of input deltas
    bboxes : dict
        shape and dtype of output, should be same shape and type as input
    means : list
        the result of the calculation is normalized, default is [0,0,0,0]
    stds : list
        the result of the calculation is normalized, default is [1,1,1,1]
    max_shape : list or tuple
        max_shape of bboxes, default is None
    wh_ratio_clip : scalar
        limit the size of deltas[:,4] and deltas[:,3] between negative
        wh_ratio_clip and positive wh_ratio_clip, default is 0.016
    kernel_name : str
        kernel name, default value is "bounding_box_decode"

    Returns
    -------
    None
    """
    rois_dtype = rois.get("dtype").lower()
    deltas_dtype = deltas.get("dtype").lower()
    para_check.check_dtype(rois_dtype, ["float16", "float32"])
    para_check.check_dtype(deltas_dtype, ["float16", "float32"])
    if rois_dtype != deltas_dtype:
        error_detail = "dtype of rois and deltas should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "rois", "deltas", error_detail)

    bboxes_instance = BoundingBoxDecode(rois, deltas, means, stds, max_shape,
                                        wh_ratio_clip, kernel_name)
    instance = bboxes_instance.tik_instance_function()
    return instance
