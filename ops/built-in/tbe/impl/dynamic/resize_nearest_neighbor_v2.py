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
resize_nearest_neighbor_v2.py
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_tik_comm_func

# max uint16
MAX_UINT16 = 2 ** 16 - 1
# max int64
MAX_INT64 = 2 ** 63 - 1
# ting param num
TILING_ARG_NUM = 16
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024


# pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class ResizeNearestNeighbor:
    """
    Function: use to store ResizeNearestNeighbor base parameters
    Modify: 2021-01-15
    """

    def __init__(self, images, size, y, align_corners, half_pixel_centers, kernel_name):
        self.tik_instance = tik.Tik()
        self.images_dtype = images.get("dtype").lower()
        self.size_dtype = size.get("dtype").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        # check dtype
        para_check.check_dtype(self.size_dtype, ("int64", "int32"), param_name="size")
        para_check.check_dtype(self.images_dtype, ("float32", "float16"), param_name="images")

        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE)

        self.elememts_vector_fp16 = tbe_platform.ELEMENTS_VECTOR_OP_FP16

        self.block_num = 16 if self.images_dtype in ("float16",) else 8
        self.vcetor_num = self.block_num * 8
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.block_num

        self.images_shape_c0 = 16
        self.height_idx_sigment_num = 512
        self.weight_idx_sigment_num = 512
        self.tiling_gm = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)

        self.images_gm = self.tik_instance.Tensor(self.images_dtype, [MAX_INT64],
                                                  name="images_gm", scope=tik.scope_gm)
        self.size_gm = self.tik_instance.Tensor(self.size_dtype, (2,),
                                                name="size_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.images_dtype, [MAX_INT64],
                                               name="out_gm", scope=tik.scope_gm)
        self.stride_threshold = MAX_UINT16 if self.images_dtype in ("float16",) else MAX_UINT16 // 2
        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", "float32")
        # init tiling data
        self.resize_scale_h = self.tik_instance.Scalar("float32", name="resize_scale_h")
        self.resize_scale_w = self.tik_instance.Scalar("float32", name="resize_scale_w")
        self.scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_key")
        self.tiling_batch = self.tik_instance.Scalar("int64", name="tiling_batch")
        self.tiling_c1 = self.tik_instance.Scalar("int64", name="tiling_c1")
        self.tiling_in_height = self.tik_instance.Scalar("int64", name="tiling_in_height")
        self.tiling_in_weight = self.tik_instance.Scalar("int64", name="tiling_in_weight")
        self.tiling_out_height = self.tik_instance.Scalar("int64", name="tiling_out_height")
        self.tiling_out_weight = self.tik_instance.Scalar("int64", name="tiling_out_weight")
        self.tiling_bc1_cut_num = self.tik_instance.Scalar("int64", name="tiling_bc1_cut_num")
        self.tiling_height_cut_num = self.tik_instance.Scalar("int64", name="tiling_height_cut_num")
        self.tiling_weight_cut_num = self.tik_instance.Scalar("int64", name="tiling_weight_cut_num")

        # init scaler for each core
        # nc1 start addr offset for per core
        self.core_nc_start = self.tik_instance.Scalar("int64", name="core_nc_start")
        # h start addr offset for per core
        self.core_height_start = self.tik_instance.Scalar("int64", name="core_height_start")
        # w start addr offset for per core
        self.core_weight_start = self.tik_instance.Scalar("int64", name="core_weight_start")
        # nc1 process len for per core
        self.core_nc_num = self.tik_instance.Scalar("int64", name="core_nc_num")
        # h process len for per core
        self.core_height_num = self.tik_instance.Scalar("int64", name="core_height_num")
        # w process len for per core
        self.core_weight_num = self.tik_instance.Scalar("int64", name="core_weight_num")
        self.cut_weight_num = None
        self.cut_height_num = None

        # init ub
        self.height_idx_ub = None
        self.weight_idx_ub = None
        self.idx_ub_fp32 = None
        self.idx_cb_fp32 = None
        self.image_out_ub = None
        self.image_in_cb_ping = None
        self.image_out_ub = None
        self.image_in_cb_ping = None

    def _tiling_args(self, tiling_ub, mode="read"):
        """
        get runtime tiling parameters from tiling
        """
        if mode == "read":
            # read tiling data
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_batch.set_as(tiling_ub[1])
            self.tiling_c1.set_as(tiling_ub[2])
            self.tiling_in_height.set_as(tiling_ub[3])
            self.tiling_in_weight.set_as(tiling_ub[4])
            self.tiling_out_height.set_as(tiling_ub[5])
            self.tiling_out_weight.set_as(tiling_ub[6])
            self.tiling_bc1_cut_num.set_as(tiling_ub[7])
            self.tiling_height_cut_num.set_as(tiling_ub[8])
            self.tiling_weight_cut_num.set_as(tiling_ub[9])

    def _core_scalar_args(self, _core_idx):
        """
        get runtime tiling parameters from tiling
        """
        self.core_nc_start.set_as(_core_idx // (self.tiling_height_cut_num * self.tiling_weight_cut_num))
        self.core_height_start.set_as(
            (_core_idx % (self.tiling_height_cut_num * self.tiling_weight_cut_num)) // self.tiling_weight_cut_num)
        self.core_weight_start.set_as(
            (_core_idx % (self.tiling_height_cut_num * self.tiling_weight_cut_num)) % self.tiling_weight_cut_num)
        # h process len for per core
        self.cut_height_num = self.tik_instance.Scalar("int64", name="cut_height_num")
        # w process len for per core
        self.cut_weight_num = self.tik_instance.Scalar("int64", name="cut_weight_num")
        self.cut_height_num.set_as(self.tiling_out_height)
        self.cut_weight_num.set_as(self.tiling_out_weight)
        with self.tik_instance.if_scope(self.tiling_key == 111000):
            # when tiling_key is 111000, will cut by input
            self.cut_height_num.set_as(self.tiling_in_height)
            self.cut_weight_num.set_as(self.tiling_in_weight)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            self.cut_weight_num.set_as(self.tiling_in_weight)

        nc_sigment = (self.tiling_batch * self.tiling_c1 + self.tiling_bc1_cut_num - 1) // self.tiling_bc1_cut_num
        h_sigment = (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num
        w_sigment = (self.cut_weight_num + self.tiling_weight_cut_num - 1) // self.tiling_weight_cut_num
        self.core_nc_start.set_as(
            (_core_idx // (self.tiling_height_cut_num * self.tiling_weight_cut_num)) * nc_sigment)
        self.core_height_start.set_as(
            ((_core_idx % (self.tiling_height_cut_num * self.tiling_weight_cut_num))
             // self.tiling_weight_cut_num) * h_sigment)
        self.core_weight_start.set_as(
            ((_core_idx % (self.tiling_height_cut_num * self.tiling_weight_cut_num))
             % self.tiling_weight_cut_num) * w_sigment)
        self.core_nc_num.set_as(nc_sigment)
        self.core_height_num.set_as(h_sigment)
        self.core_weight_num.set_as(w_sigment)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            # when tiling_key is 101000, w start will start from align_num*n
            align_num = self.tiling_out_weight // self.tiling_in_weight
            self.core_weight_num.set_as(self.core_weight_num * align_num)
            self.core_weight_start.set_as(self.core_weight_start * align_num)
            self.cut_weight_num.set_as(self.tiling_in_weight * align_num)

        with self.tik_instance.if_scope(
                self.core_nc_start + self.core_nc_num >= self.tiling_batch * self.tiling_c1):
            self.core_nc_num.set_as(self.tiling_batch * self.tiling_c1 - self.core_nc_start)
        with self.tik_instance.if_scope(
                self.core_height_start + self.core_height_num >= self.cut_height_num):
            self.core_height_num.set_as(self.cut_height_num - self.core_height_start)
        with self.tik_instance.if_scope(
                self.core_weight_start + self.core_weight_num >= self.cut_weight_num):
            self.core_weight_num.set_as(self.cut_weight_num - self.core_weight_start)
        core_used = self.tiling_weight_cut_num * self.tiling_height_cut_num * self.tiling_bc1_cut_num
        with self.tik_instance.if_scope(_core_idx >= core_used):
            self.core_nc_num.set_as(0)
            self.core_height_num.set_as(0)
            self.core_weight_num.set_as(0)

    def _init_ub_tensor_for_idx(self, height_idx_len=0, weight_idx_len=0):
        """
        compute the ub size of tensors
        """
        height_idx_len = self.height_idx_sigment_num if height_idx_len == 0 else height_idx_len
        weight_idx_len = self.weight_idx_sigment_num if weight_idx_len == 0 else weight_idx_len
        idx_max_len = max(height_idx_len, weight_idx_len)
        self.height_idx_ub = self.tik_instance.Tensor("int32", (height_idx_len,),
                                                      name="height_idx", scope=tik.scope_ubuf)
        self.weight_idx_ub = self.tik_instance.Tensor("int32", (weight_idx_len,),
                                                      name="weight_idx", scope=tik.scope_ubuf)
        self.idx_ub_fp32 = self.tik_instance.Tensor("float32", (idx_max_len,),
                                                    name="idx_ub_fp32", scope=tik.scope_ubuf)
        self.idx_cb_fp32 = self.tik_instance.Tensor("float32", (idx_max_len,),
                                                    name="idx_cb_fp32", scope=tik.scope_cbuf)
        avail_bytes = self.ub_size_bytes - (height_idx_len + weight_idx_len + idx_max_len) * 4
        avail_block = avail_bytes // 32 // 2
        self.ub_max_num = avail_block * self.block_num

    def _init_ub_tensor_for_images(self, mode="all"):
        """
        _init_ub_tensor_for_images
        """
        if mode in ("all",):
            self.image_out_ub = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num,),
                                                         name="image_out_ub", scope=tik.scope_ubuf)
            self.image_in_cb_ping = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num,),
                                                             name="image_in_cb_ping", scope=tik.scope_cbuf)
        if mode in ("l1",):
            self.image_in_cb_ping = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num,),
                                                             name="image_in_cb_ping", scope=tik.scope_cbuf)
        if mode in ("ub",):
            self.image_out_ub = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num,),
                                                         name="image_out_ub", scope=tik.scope_ubuf)

    def scalar_vconv_int32_to_fp32(self, int32_value, float32_value):
        """
        vconv one scalar from int32 to fp32 usr vector
        """
        with self.tik_instance.new_stmt_scope():
            idx_int32_tmp = self.tik_instance.Tensor("int32", (64,),
                                                     name="idx_int32_tmp", scope=tik.scope_ubuf)
            idx_fp32_tmp = self.tik_instance.Tensor("float32", (64,),
                                                    name="idx_fp32_tmp", scope=tik.scope_ubuf)
            idx_int32_tmp[0].set_as(int32_value)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, idx_fp32_tmp, idx_int32_tmp, 1)
            float32_value.set_as(idx_fp32_tmp[0])

    def calcu_out_in_idx(self, scale, des_idx_ub, src_idx_fp_ub, idx_num):
        """
        if not self.align_corners and self.half_pixel_centers:
            # vconv_f322s32f((idx + 0.5) * scale)
        if not self.align_corners and not self.half_pixel_centers:
            # vconv_f322s32f(idx * scale)
        if self.align_corners and not self.half_pixel_centers:
            # vconv_f322s32r(idx * scale)
        if self.align_corners and self.half_pixel_centers:
            # vconv_f322s32r((idx + 0.5) * scale)
        """
        with self.tik_instance.new_stmt_scope():
            calcu_out_in_idx_tmp_ub = self.tik_instance.Tensor(src_idx_fp_ub.dtype, src_idx_fp_ub.shape,
                                                               name="calcu_out_in_idx_tmp_ub", scope=tik.scope_ubuf)
            vector_repeat_num = (idx_num + 63) // 64
            if self.half_pixel_centers:
                # calcu: (idx + 0.5) * scale
                self.tik_instance.vadds(64, calcu_out_in_idx_tmp_ub, src_idx_fp_ub, 0.5,
                                        vector_repeat_num, 1, 1, 8, 8)
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub, scale,
                                        vector_repeat_num, 1, 1, 8, 8)
            else:
                # calcu: idx * scale
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, src_idx_fp_ub, scale,
                                        vector_repeat_num, 1, 1, 8, 8)
            if self.align_corners:
                # will use vconv_f322s32r to cast to int32
                util_tik_comm_func.tik_func_vconv(self.tik_instance, des_idx_ub, calcu_out_in_idx_tmp_ub,
                                                  vector_repeat_num * 64, mode="round")
            else:
                # will use vconv_f322s32f to cast to int32
                util_tik_comm_func.tik_func_vconv(self.tik_instance, des_idx_ub, calcu_out_in_idx_tmp_ub,
                                                  vector_repeat_num * 64, mode="floor")

    def _function_default(self, is_src_stride_copy=False, is_dst_stride_copy=False,
                          is_w_algin=False, is_big_to_small=True):
        """
        _function_default, run this
        """
        if is_w_algin:
            is_big_to_small = False
        self.height_idx_sigment_num = 64
        self.weight_idx_sigment_num = 128
        # cut by output h and output w
        self._init_ub_tensor_for_idx()

        # gen 0-511 to l1 fp32
        with self.tik_instance.new_stmt_scope():
            fill_index_in_ub(self.tik_instance, self.idx_ub_fp32, self.weight_idx_sigment_num)

        # calcu is_src_stride_copy and is_dst_stride_copy use scalar
        scalar_is_src_stride = self.tik_instance.Scalar("int32", name="scalar_is_src_stride", init_value=1)
        scalar_is_dst_stride = self.tik_instance.Scalar("int32", name="scalar_is_dst_stride", init_value=1)

        with self.tik_instance.if_scope(self.tiling_in_height * self.tiling_in_weight > self.stride_threshold):
            scalar_is_src_stride.set_as(0)
        with self.tik_instance.if_scope(self.tiling_out_height * self.tiling_out_weight > self.stride_threshold):
            scalar_is_dst_stride.set_as(0)
        # calcu is_src_stride_copy and is_dst_stride_copy use scalar end

        # init a scalar for w sigment one time
        w_loop_sigment = self.tik_instance.Scalar("int32", name="w_loop_sigment",
                                                  init_value=self.weight_idx_sigment_num)
        if is_w_algin:
            # if weight is input_w resize to n*input_w, one sigment must be n algin
            # exp: 24 resize to 48, one sigment of weight must be 2*n
            with self.tik_instance.new_stmt_scope():
                algin_num_scalar = self.tik_instance.Scalar("int32", name="algin_num_scalar")
                algin_num_scalar.set_as(self.tiling_out_weight // self.tiling_in_weight)
                w_loop_sigment.set_as(w_loop_sigment // algin_num_scalar * algin_num_scalar)

        with self.tik_instance.if_scope(tik.all(self.core_weight_num < w_loop_sigment, self.core_weight_num > 0)):
            w_loop_sigment.set_as(self.core_weight_num)

        w_loop_num = self.core_weight_num // w_loop_sigment
        w_tail_num = self.core_weight_num % w_loop_sigment
        nc_max_segment = self.ub_max_num // (w_loop_sigment * self.images_shape_c0)
        nc_loop = self.core_nc_num // nc_max_segment
        nc_tail = self.core_nc_num % nc_max_segment

        if is_big_to_small:
            # mean: if input_w // output_w > 4, the input_w can not save in l1
            # will modify w_loop_sigment base on input
            w_input_output_rate = (self.tiling_in_weight + self.tiling_out_weight - 1) // self.tiling_out_weight
            w_loop_input_sigment = (w_loop_sigment - 1) * w_input_output_rate + 1
            _max_nc_w_in_l1 = self.ub_max_num * 4 // self.images_shape_c0
            with self.tik_instance.if_scope(self.tiling_out_weight * 4 < self.tiling_in_weight):
                with self.tik_instance.if_scope(tik.any(w_loop_input_sigment * nc_tail > _max_nc_w_in_l1,
                                                        w_loop_input_sigment * nc_loop > _max_nc_w_in_l1)):
                    w_loop_sigment.set_as(w_loop_sigment * 4 // w_input_output_rate)
                    with self.tik_instance.if_scope(w_loop_sigment < 1):
                        w_loop_sigment.set_as(1)
                    w_loop_num = self.core_weight_num // w_loop_sigment
                    w_tail_num = self.core_weight_num % w_loop_sigment

        scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        # vconv start idx from int32 scalar to fp32 scalar
        self.scalar_vconv_int32_to_fp32(self.core_weight_start, scalar_idx_fp32)
        # do vadds 0,1,2,3,4 + fp32_scalar
        self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                (w_loop_sigment + 63) // 64, 1, 1, 8, 8)
        self.scalar_vconv_int32_to_fp32(w_loop_sigment, scalar_idx_fp32)

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = w_loop_idx * w_loop_sigment + self.core_weight_start
            self.calcu_out_in_idx(self.resize_scale_w, self.weight_idx_ub,
                                  self.idx_ub_fp32, self.weight_idx_sigment_num)
            self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                    (self.weight_idx_sigment_num + 63) // 64, 1, 1, 8, 8)

            scalar_w_start_idx = self.tik_instance.Scalar("int32", name="scalar_w_start_idx")
            scalar_w_end_idx = self.tik_instance.Scalar("int32", name="scalar_w_end_idx")
            scalar_w_start_idx.set_as(self.weight_idx_ub[0])
            scalar_w_end_idx.set_as(self.weight_idx_ub[w_do_len - 1])
            input_w_len = scalar_w_end_idx - scalar_w_start_idx + 1

            # one sigment h and one sigment w

            def _do_single_nc(do_nc_num, _nc_loop_idx):
                def _do_one_height(h_idx, output_ub, input_l1):
                    h_gm_offset = h_idx + h_loop_offset
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    height_idx_ub = self.tik_instance.Tensor("int32", (64,),
                                                             name="height_idx", scope=tik.scope_ubuf)
                    height_idx_ub_fp32 = self.tik_instance.Tensor("float32", (64,),
                                                                  name="height_idx_ub_fp32", scope=tik.scope_ubuf)
                    self.tik_instance.vector_dup(64, height_idx_ub_fp32, 0, 1, 1, 8)
                    util_tik_comm_func.tik_func_vector(self.tik_instance, height_idx_ub,
                                                       h_idx + self.core_height_start, 64)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, height_idx_ub_fp32,
                                                      height_idx_ub, 64)

                    self.calcu_out_in_idx(self.resize_scale_h, height_idx_ub,
                                          height_idx_ub_fp32, 1)

                    scalar_in_h_idx.set_as(height_idx_ub[0])
                    with self.tik_instance.if_scope(scalar_is_src_stride == 0):
                        with self.tik_instance.for_range(0, do_nc_num) as _sigment_idx:
                            data_move_cbuf_offset = (input_w_len * self.images_shape_c0) * _sigment_idx
                            nc_gm_input_offset = \
                                (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                * self.tiling_in_weight * self.tiling_in_height
                            data_move_gm_offset = \
                                nc_gm_input_offset + scalar_in_h_idx * self.tiling_in_weight + scalar_w_start_idx
                            self.tik_instance.data_move(input_l1[data_move_cbuf_offset],
                                                        self.images_gm[data_move_gm_offset * self.images_shape_c0],
                                                        0, 1,
                                                        input_w_len * self.images_shape_c0 // self.block_num, 0, 0)
                    with self.tik_instance.else_scope():
                        data_move_cbuf_offset = 0
                        nc_gm_input_offset = \
                            (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                            * self.tiling_in_weight * self.tiling_in_height
                        data_move_gm_offset = \
                            nc_gm_input_offset + scalar_in_h_idx * self.tiling_in_weight + scalar_w_start_idx
                        data_move_burst_num = do_nc_num
                        data_move_burst_len = input_w_len * self.images_shape_c0 // self.block_num
                        data_move_src_stride = \
                            (self.tiling_in_weight * self.tiling_in_height - input_w_len) \
                            * self.images_shape_c0 // self.block_num
                        self.tik_instance.data_move(input_l1[data_move_cbuf_offset],
                                                    self.images_gm[data_move_gm_offset * self.images_shape_c0],
                                                    0,
                                                    data_move_burst_num,
                                                    data_move_burst_len,
                                                    data_move_src_stride, 0)

                    if not is_w_algin:
                        with self.tik_instance.for_range(0, w_do_len) as w_idx:
                            scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                            scalar_in_w_idx.set_as(self.weight_idx_ub[w_idx])
                            nc_cbuf_offset = input_w_len * self.images_shape_c0
                            burst_num = do_nc_num
                            burst_len = self.images_shape_c0 // self.block_num
                            cbuf_burst_stride = nc_cbuf_offset // self.block_num - burst_len
                            ub_out_burst_strde = w_do_len * self.images_shape_c0 // self.block_num - burst_len

                            self.tik_instance.data_move(output_ub[w_idx * self.images_shape_c0],
                                                        input_l1[(scalar_in_w_idx - scalar_w_start_idx)
                                                                 * self.images_shape_c0],
                                                        0, burst_num, burst_len,
                                                        cbuf_burst_stride, ub_out_burst_strde)
                    else:
                        # input_w_len
                        scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                        scalar_in_w_idx.set_as(self.weight_idx_ub[0])
                        w_algin_num = self.tiling_out_weight // self.tiling_in_weight
                        with self.tik_instance.for_range(0, input_w_len) as w_input_idx:
                            nc_cbuf_offset = input_w_len * self.images_shape_c0
                            burst_num = do_nc_num
                            burst_len = self.images_shape_c0 // self.block_num
                            cbuf_burst_stride = nc_cbuf_offset // self.block_num - burst_len
                            ub_out_burst_strde = w_do_len * self.images_shape_c0 // self.block_num - burst_len

                            self.tik_instance.data_move(output_ub[w_input_idx * w_algin_num * self.images_shape_c0],
                                                        input_l1[w_input_idx * self.images_shape_c0],
                                                        0, burst_num, burst_len,
                                                        cbuf_burst_stride, ub_out_burst_strde)
                        # datamove to all
                        burst_num = do_nc_num * input_w_len
                        burst_len = self.images_shape_c0 // self.block_num
                        with self.tik_instance.for_range(1, w_algin_num) as copy_num:
                            data_move_src_offset = 0
                            data_move_dst_offset = self.images_shape_c0 * copy_num
                            data_move_src_stride = (w_algin_num - 1) * self.images_shape_c0 // self.block_num
                            data_move_dst_stride = data_move_src_stride
                            self.tik_instance.data_move(output_ub[data_move_dst_offset:],
                                                        output_ub[data_move_src_offset:],
                                                        0, burst_num, burst_len,
                                                        data_move_src_stride, data_move_dst_stride)

                    with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                        with self.tik_instance.for_range(0, do_nc_num) as _sigment_idx:
                            nc_gm_offset = \
                                (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                * self.tiling_out_weight * self.tiling_out_height
                            output_gm_offset = \
                                nc_gm_offset + h_gm_offset * self.tiling_out_weight + w_gm_offset
                            ub_output_offset = w_do_len * self.images_shape_c0 * _sigment_idx
                            self.tik_instance.data_move(self.out_gm[output_gm_offset * self.images_shape_c0:],
                                                        output_ub[ub_output_offset:], 0, 1,
                                                        w_do_len * self.images_shape_c0 // self.block_num,
                                                        0, 0)
                    with self.tik_instance.else_scope():
                        nc_gm_offset = \
                            (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                            * self.tiling_out_weight * self.tiling_out_height
                        output_gm_offset = nc_gm_offset + h_gm_offset * self.tiling_out_weight + w_gm_offset
                        data_move_ub_offset = 0
                        data_move_burst_num = do_nc_num
                        data_move_burst_len = w_do_len * self.images_shape_c0 // self.block_num
                        data_move_dst_stride = \
                            (self.tiling_out_weight * self.tiling_out_height - w_do_len) \
                            * self.images_shape_c0 // self.block_num
                        self.tik_instance.data_move(self.out_gm[output_gm_offset * self.images_shape_c0:],
                                                    output_ub[data_move_ub_offset:], 0,
                                                    data_move_burst_num,
                                                    data_move_burst_len,
                                                    0, data_move_dst_stride)

                image_out_ub_ping = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num,),
                                                             name="image_out_ub_ping", scope=tik.scope_ubuf)
                image_out_ub_pang = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num,),
                                                             name="image_out_ub_pang", scope=tik.scope_ubuf)
                image_in_cb_ping = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num * 4,),
                                                            name="image_in_cb_ping", scope=tik.scope_cbuf)
                image_in_cb_pang = self.tik_instance.Tensor(self.images_dtype, (self.ub_max_num * 4,),
                                                            name="image_in_cb_pang", scope=tik.scope_cbuf)
                with self.tik_instance.for_range(0, h_do_len // 2) as _h_idx:
                    _do_one_height(_h_idx * 2, image_out_ub_ping, image_in_cb_ping)
                    _do_one_height(_h_idx * 2 + 1, image_out_ub_pang, image_in_cb_pang)
                with self.tik_instance.if_scope(h_do_len % 2 == 1):
                    _do_one_height(h_do_len - 1, image_out_ub_ping, image_in_cb_ping)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            h_loop_sigment_start = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_gm_offset = h_loop_sigment_start
            # calcu h idx

            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, w_loop_sigment, h_gm_offset, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_gm_offset, h_do_len)

        _run_h_loop_default(0, self.core_height_num)

    def _function_hw_to_nhnw_resize(self, is_w_equal=False):
        """
        _function_hw_to_nhnw_resize, when tiling key = 111000, run this
        """
        # h boardcast base input_h cut
        size_h_n = self.tiling_out_height // self.tiling_in_height
        size_w_n = self.tiling_out_weight // self.tiling_in_weight
        output_w_size = self.core_weight_num * size_w_n
        w_output_size_one_line = self.tik_instance.Scalar("int64", name="input_w_size", init_value=0)
        w_output_size_one_line.set_as(output_w_size)

        with self.tik_instance.if_scope(
                tik.all(self.ub_max_num < output_w_size * self.images_shape_c0,
                        self.core_weight_num > 0)):
            w_output_size_one_line.set_as((self.ub_max_num // self.images_shape_c0 // size_w_n) * size_w_n)
        with self.tik_instance.if_scope(w_output_size_one_line == 0):
            w_output_size_one_line.set_as((self.ub_max_num // self.images_shape_c0 // size_w_n) * size_w_n)
        _w_loop_num = self.core_weight_num // (w_output_size_one_line // size_w_n)
        _w_tail_num = self.core_weight_num % (w_output_size_one_line // size_w_n)
        _segment_h_num = self.ub_max_num // self.images_shape_c0 // w_output_size_one_line
        _h_loop_num = self.core_height_num // _segment_h_num
        _h_tail_num = self.core_height_num % _segment_h_num

        def _run_h_loop(h_loop_idx, h_do_len, w_start_offset, w_do_len, nc_idx):
            h_sigment_start = h_loop_idx * _segment_h_num + self.core_height_start
            nc_sigment_start = nc_idx + self.core_nc_start
            self._init_ub_tensor_for_images("l1")
            self._init_ub_tensor_for_images("ub")
            aicore_mem = self.image_out_ub if is_w_equal else self.image_in_cb_ping

            # copy h * w input to l1
            data_move_gm_offset = \
                nc_sigment_start * self.tiling_in_height * self.tiling_in_weight + \
                h_sigment_start * self.tiling_in_weight + w_start_offset
            data_move_burst_num = h_do_len
            data_move_burst_len = w_do_len * self.images_shape_c0 // self.block_num
            data_move_src_stride = (self.tiling_in_weight - w_do_len) * self.images_shape_c0 // self.block_num
            data_move_dst_stride = 0
            self.tik_instance.data_move(aicore_mem,
                                        self.images_gm[data_move_gm_offset * self.images_shape_c0],
                                        0,
                                        data_move_burst_num,
                                        data_move_burst_len,
                                        data_move_src_stride,
                                        data_move_dst_stride)
            if not is_w_equal:
                # boardcast w from l1 to ub
                data_move_burst_num = h_do_len * w_do_len
                data_move_burst_len = self.images_shape_c0 // self.block_num
                data_move_src_stride = 0
                data_move_dst_stride = (size_w_n - 1) * self.images_shape_c0 // self.block_num
                self.tik_instance.data_move(self.image_out_ub,
                                            aicore_mem,
                                            0,
                                            data_move_burst_num,
                                            data_move_burst_len,
                                            data_move_src_stride,
                                            data_move_dst_stride)
                # ub to ub
                with self.tik_instance.for_range(1, size_w_n) as _w_idx:
                    data_move_dst_offset = _w_idx * self.images_shape_c0
                    data_move_burst_num = h_do_len * w_do_len
                    data_move_burst_len = self.images_shape_c0 // self.block_num
                    data_move_src_stride = (size_w_n - 1) * self.images_shape_c0 // self.block_num
                    data_move_dst_stride = (size_w_n - 1) * self.images_shape_c0 // self.block_num
                    self.tik_instance.data_move(self.image_out_ub[data_move_dst_offset],
                                                self.image_out_ub,
                                                0,
                                                data_move_burst_num,
                                                data_move_burst_len,
                                                data_move_src_stride,
                                                data_move_dst_stride)
            with self.tik_instance.for_range(0, size_h_n) as _h_idx:
                # copy output one h by one h
                data_move_src_offset = 0
                data_move_dst_offset = \
                    nc_sigment_start * self.tiling_out_height * self.tiling_out_weight + \
                    h_sigment_start * size_h_n * self.tiling_out_weight + w_start_offset * size_w_n + \
                    _h_idx * self.tiling_out_weight
                data_move_burst_num = h_do_len
                data_move_burst_len = w_do_len * size_w_n * self.images_shape_c0 // self.block_num
                data_move_src_stride = 0
                data_move_dst_stride = \
                    (size_h_n * self.tiling_out_weight - w_do_len * size_w_n) \
                    * self.images_shape_c0 // self.block_num
                self.tik_instance.data_move(self.out_gm[data_move_dst_offset * self.images_shape_c0:],
                                            self.image_out_ub[data_move_src_offset:],
                                            0,
                                            data_move_burst_num,
                                            data_move_burst_len,
                                            data_move_src_stride,
                                            data_move_dst_stride)

        def _run_w_loop(w_loop_idx, input_w_len):
            w_sigment_start = w_loop_idx * (w_output_size_one_line // size_w_n) + self.core_weight_start
            with self.tik_instance.for_range(0, self.core_nc_num) as nc_idx:
                with self.tik_instance.for_range(0, _h_loop_num, thread_num=2) as _h_loop_idx:
                    _run_h_loop(_h_loop_idx, _segment_h_num, w_sigment_start, input_w_len, nc_idx)
                with self.tik_instance.if_scope(_h_tail_num != 0):
                    _run_h_loop(_h_loop_num, _h_tail_num, w_sigment_start, input_w_len, nc_idx)

        with self.tik_instance.for_range(0, _w_loop_num) as _w_loop_idx:
            _run_w_loop(_w_loop_idx, w_output_size_one_line // size_w_n)
        with self.tik_instance.if_scope(_w_tail_num != 0):
            _run_w_loop(_w_loop_num, _w_tail_num)

    def calculate_scale(self):
        """
        calculate scale user input h/w and output h/w
        """
        with self.tik_instance.new_stmt_scope():
            height_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="height_input_fp32", scope=tik.scope_ubuf)
            weight_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="weight_input_fp32", scope=tik.scope_ubuf)
            height_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                          name="height_input_int32", scope=tik.scope_ubuf)
            weight_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                          name="weight_input_int32", scope=tik.scope_ubuf)
            height_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                          name="height_output_fp32", scope=tik.scope_ubuf)
            weight_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                          name="weight_output_fp32", scope=tik.scope_ubuf)

            height_input_int32[0].set_as(self.tiling_in_height)
            weight_input_int32[0].set_as(self.tiling_in_weight)
            height_input_int32[self.block_num].set_as(self.tiling_out_height)
            weight_input_int32[self.block_num].set_as(self.tiling_out_weight)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_input_fp32,
                                              height_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, weight_input_fp32,
                                              weight_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_output_fp32,
                                              height_input_int32[self.block_num:], 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, weight_output_fp32,
                                              weight_input_int32[self.block_num:], 1)

            with self.tik_instance.if_scope(tik.all(self.align_corners, self.tiling_out_height > 1)):
                self.tik_instance.vadds(1, height_output_fp32, height_output_fp32, -1.0, 1, 1, 1, 8, 8)
                self.tik_instance.vadds(1, height_input_fp32, height_input_fp32, -1.0, 1, 1, 1, 8, 8)

            self.tik_instance.vrec(1, height_output_fp32[self.block_num:], height_output_fp32, 1, 1, 1, 8, 8)
            _tik_fuc_vrec_newton(self.tik_instance, height_output_fp32[self.block_num:], height_output_fp32,
                                 1, block_num=self.block_num)
            self.tik_instance.vmul(1, height_input_fp32, height_input_fp32, height_output_fp32[self.block_num:],
                                   1, 1, 1, 1, 8, 8, 8)
            self.resize_scale_h.set_as(height_input_fp32[0])

            with self.tik_instance.if_scope(tik.all(self.align_corners, self.tiling_out_weight > 1)):
                self.tik_instance.vadds(1, weight_output_fp32, weight_output_fp32, -1.0, 1, 1, 1, 8, 8)
                self.tik_instance.vadds(1, weight_input_fp32, weight_input_fp32, -1.0, 1, 1, 1, 8, 8)
            self.tik_instance.vrec(1, weight_output_fp32[self.block_num:], weight_output_fp32, 1, 1, 1, 8, 8)
            _tik_fuc_vrec_newton(self.tik_instance, weight_output_fp32[self.block_num:], weight_output_fp32,
                                 1, block_num=self.block_num)
            self.tik_instance.vmul(1, weight_input_fp32, weight_input_fp32, weight_output_fp32[self.block_num:],
                                   1, 1, 1, 1, 8, 8, 8)
            self.resize_scale_w.set_as(weight_input_fp32[0])

    def _do_resize_base_tiling_key(self):
        """
        tiling braches
        """
        # calcu scale for h and w
        self.calculate_scale()

        # tiling_key format: 000000
        # 1. Reserved, default 1
        # 2. h align flag, 0: h -> x.x*h, 1: h -> nh, 2: nh -> h, 3: h = h
        # 3. w align flag, 0: w -> x.x*w, 1: w -> nw, 2: nw -> w, 3: w = w
        # 4. src stride flag, 0: can not copy with stride 1: can copy with stride
        # 5. des stride flag, 0: can not copy with stride 1: can copy with stride
        # 6. Reserved, default 0
        with self.tik_instance.if_scope(self.tiling_key == 100000):
            with self.tik_instance.new_stmt_scope():
                self._function_default(is_src_stride_copy=False, is_dst_stride_copy=False, is_w_algin=False)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            with self.tik_instance.new_stmt_scope():
                self._function_default(is_src_stride_copy=False, is_dst_stride_copy=False, is_w_algin=True)
        with self.tik_instance.if_scope(self.tiling_key == 111000):
            # tiling_key is 111000, mean: h,w resize to nh, mw
            with self.tik_instance.new_stmt_scope():
                self._function_hw_to_nhnw_resize()
        with self.tik_instance.if_scope(self.tiling_key == 113000):
            # tiling_key is 113000, mean: h,w resize to nh, mw, and m = 1
            with self.tik_instance.new_stmt_scope():
                self._function_hw_to_nhnw_resize(is_w_equal=True)

    def _do_resize(self):
        """
        main process of _do_resize
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as _core_idx:
            with self.tik_instance.new_stmt_scope():
                tiling_ub = self.tik_instance.Tensor("int64", (TILING_ARG_NUM,),
                                                     name="tiling_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
                self._tiling_args(tiling_ub, "read")
            self._core_scalar_args(_core_idx)
            self._do_resize_base_tiling_key()

    def resize_nearest_neighbor_v2_operator(self):
        """
        resize_nearest_neighbor_v2_operator
        """
        self._do_resize()
        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.images_gm, self.size_gm),
                                   outputs=(self.out_gm,),
                                   flowtable=(self.tiling_gm,), config=opt_config)

        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.ai_core_num,
                                                            "max_w_len": self.ub_max_num // self.images_shape_c0,
                                                            "align_corners": int(self.align_corners),
                                                            "half_pixel_centers": int(self.half_pixel_centers)})

        return self.tik_instance


def _tik_fuc_vrec_newton(tik_instance, vrec_ub, origin_ub, do_len, newton_iteration=6, block_num=16):
    """
    only do newton for vrec result

    Parameters
    ----------
    tik_instance: class
        tik_instance
    vrec_ub: ub
        the result of vrec
    origin_ub: ub
        the origin input for vrec
    do_len: int
        vrec num
    newton_iteration: int
        do newton iteration
    block_num: int
        num in one block

    Returns
    -------
    None
    """
    with tik_instance.new_stmt_scope():
        vrec_newton_1 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_1", scope=tik.scope_ubuf)
        vrec_newton_2 = tik_instance.Tensor(
            vrec_ub.dtype, (((do_len + block_num - 1) // block_num) * block_num,),
            name="vrec_newton_2", scope=tik.scope_ubuf)

        def _one_newton():
            tik_instance.vmul(1, vrec_newton_1, vrec_ub, origin_ub, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vmuls(1, vrec_newton_2, vrec_newton_1, -1, 1, 1, 1, 8, 8)
            tik_instance.vadds(1, vrec_newton_1, vrec_newton_2, 2, 1, 1, 1, 8, 8)
            tik_instance.vmul(1, vrec_ub, vrec_newton_1, vrec_ub, 1, 1, 1, 1, 8, 8, 8)

        for _ in range(newton_iteration):
            _one_newton()


def fill_index_in_ub(tik_instance, idx_ub, idx_num, vector_num=64):
    """
    fill 0,1,2  .... (idx_num -1) in idx_ub
    when the idx_num is less than 16, fill it one by one
    when the type is not int32, will fill in int32 ub and cast to idx_ub dtype
    when the type is int32, will fill in int32 one by one
    """
    # when the idx_num is less than 16, fill it one by one
    _idx_scalar = tik_instance.Scalar(dtype=idx_ub.dtype)

    vector_num_ub = tik_instance.Tensor(idx_ub.dtype, (vector_num,),
                                        name="vector_num_ub", scope=tik.scope_ubuf)
    for _idx in range(vector_num // 8):
        _idx_scalar.set_as(_idx)
        idx_ub[_idx].set_as(_idx_scalar)
    tik_instance.vector_dup(vector_num, vector_num_ub, vector_num // 8, 1, 1, 8)
    with tik_instance.for_range(1, 8) as add_idx:
        add_offset = add_idx * vector_num // 8
        tik_instance.vadd(vector_num // 8, idx_ub[add_offset:], vector_num_ub,
                          idx_ub[add_offset - (vector_num // 8):],
                          1, 1, 1, 1, 8, 0, 8)

    tik_instance.vector_dup(vector_num, vector_num_ub, vector_num, 1, 1, 8)
    idx_vector_num = (idx_num + vector_num - 1) // vector_num
    with tik_instance.for_range(1, idx_vector_num) as add_idx:
        add_offset = add_idx * vector_num
        tik_instance.vadd(vector_num, idx_ub[add_offset:], vector_num_ub, idx_ub[add_offset - vector_num:],
                          1, 1, 1, 1, 8, 0, 8)


@register_operator("ResizeNearestNeighborV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def resize_nearest_neighbor_v2(images, size, y, align_corners=False, half_pixel_centers=False,
                               kernel_name="resize_nearest_neighbor_v2"):
    """Resize `images` to `size` using nearest neighbor interpolation.

    Parameters
    ----------
    images: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    size: dict
        the dict of input, the height and width of output tensor
        only support 5HD and dtype supports 'float16', 'float32'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float16', 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_nearest_neighbor`

    Returns
    -------
    tik_instance
    """
    obj = ResizeNearestNeighbor(images, size, y, align_corners, half_pixel_centers,
                                kernel_name)

    return obj.resize_nearest_neighbor_v2_operator()
