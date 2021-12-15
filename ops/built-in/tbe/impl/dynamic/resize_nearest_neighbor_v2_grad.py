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
resize_nearest_neighbor_v2_grad.py
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util import util_tik_comm_func


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # max uint16
    MAX_UINT16 = 2 ** 16 - 1
    # max int64
    MAX_INT64 = 2 ** 63 - 1
    # tiling param num
    TILING_ARG_NUM = 16
    # reserved ub size
    RESERVED_UB_SIZE = 8 * 1024
    # The max repeat_times of vadd instruction
    VADD_MAX_REPEAT_TIMES = 255


# 'pylint: disable=too-many-instance-attributes,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,too-many-statements,unused-argument,invalid-name
class ResizeNearestNeighborV2Grad:
    """
    Function: use to store ResizeNearestNeighborV2Grad base parameters
    Modify: 2021-03-10
    """
    def __init__(self, grads, size, y, align_corners, half_pixel_centers, kernel_name):
        self.tik_instance = tik.Tik()
        self.grads_dtype = grads.get("dtype").lower()
        self.grads_shape = grads.get("shape")
        self.size_dtype = size.get("dtype").lower()
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        # check dtype
        para_check.check_dtype(self.size_dtype, ("int64", "int32"), param_name="size")
        para_check.check_dtype(self.grads_dtype, ("float32"), param_name="grads")

        self.kernel_name = kernel_name
        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB_SIZE)

        self.elememts_vector_fp16 = tbe_platform.ELEMENTS_VECTOR_OP_FP16

        self.block_num = 16 if self.grads_dtype in ("float16",) else 8
        self.vcetor_num = self.block_num * 8
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.block_num

        self.grads_shape_c0 = 16
        self.height_idx_segment_num = 512
        self.width_idx_segment_num = 512
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm", scope=tik.scope_gm)

        self.grads_gm = self.tik_instance.Tensor(self.grads_dtype, [Constant.MAX_INT64],
                                                 name="grads_gm", scope=tik.scope_gm)
        self.size_gm = self.tik_instance.Tensor(self.size_dtype, (2,),
                                                name="size_gm", scope=tik.scope_gm)
        self.out_gm = self.tik_instance.Tensor(self.grads_dtype, [Constant.MAX_INT64],
                                               name="out_gm", scope=tik.scope_gm, is_atomic_add=True)
        self.stride_threshold = Constant.MAX_UINT16 if self.grads_dtype in ("float16",) else Constant.MAX_UINT16 // 2
        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", "float32")
        # init tiling data
        self.resize_scale_h = self.tik_instance.Scalar("float32", name="resize_scale_h")
        self.resize_scale_w = self.tik_instance.Scalar("float32", name="resize_scale_w")
        self.scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_key")
        self.tiling_batch = self.tik_instance.Scalar("int64", name="tiling_batch")
        self.tiling_c1 = self.tik_instance.Scalar("int64", name="tiling_c1")
        self.tiling_in_height = self.tik_instance.Scalar("int64", name="tiling_in_height")
        self.tiling_in_width = self.tik_instance.Scalar("int64", name="tiling_in_width")
        self.tiling_out_height = self.tik_instance.Scalar("int64", name="tiling_out_height")
        self.tiling_out_width = self.tik_instance.Scalar("int64", name="tiling_out_width")
        self.tiling_bc1_cut_num = self.tik_instance.Scalar("int64", name="tiling_bc1_cut_num")
        self.tiling_height_cut_num = self.tik_instance.Scalar("int64", name="tiling_height_cut_num")
        self.tiling_width_cut_num = self.tik_instance.Scalar("int64", name="tiling_width_cut_num")

        # init scaler for each core
        # nc1 start addr offset for per core
        self.core_nc_start = self.tik_instance.Scalar("int64", name="core_nc_start")
        # h start addr offset for per core
        self.core_height_start = self.tik_instance.Scalar("int64", name="core_height_start")
        # w start addr offset for per core
        self.core_width_start = self.tik_instance.Scalar("int64", name="core_width_start")
        # nc1 process len for per core
        self.core_nc_num = self.tik_instance.Scalar("int64", name="core_nc_num")
        # h process len for per core
        self.core_height_num = self.tik_instance.Scalar("int64", name="core_height_num")
        # w process len for per core
        self.core_width_num = self.tik_instance.Scalar("int64", name="core_width_num")
        self.cut_width_num = None
        self.cut_height_num = None

        # init ub
        self.height_idx_ub = None
        self.width_idx_ub = None
        self.idx_ub_fp32 = None
        self.grad_out_ub = None
        self.grad_in_gm_ping = None
        self.grad_in_gm_ping = None

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
            self.tiling_in_width.set_as(tiling_ub[4])
            self.tiling_out_height.set_as(tiling_ub[5])
            self.tiling_out_width.set_as(tiling_ub[6])
            self.tiling_bc1_cut_num.set_as(tiling_ub[7])
            self.tiling_height_cut_num.set_as(tiling_ub[8])
            self.tiling_width_cut_num.set_as(tiling_ub[9])

    def _core_scalar_args(self, _core_idx):
        """
        get runtime tiling parameters from tiling
        """
        self.core_nc_start.set_as(_core_idx // (self.tiling_height_cut_num * self.tiling_width_cut_num))
        self.core_height_start.set_as(
            (_core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num)) // self.tiling_width_cut_num)
        self.core_width_start.set_as(
            (_core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num)) % self.tiling_width_cut_num)
        # h process len for per core
        self.cut_height_num = self.tik_instance.Scalar("int64", name="cut_height_num")
        # w process len for per core
        self.cut_width_num = self.tik_instance.Scalar("int64", name="cut_width_num")
        self.cut_height_num.set_as(self.tiling_in_height)
        self.cut_width_num.set_as(self.tiling_in_width)

        nc_segment = (self.tiling_batch * self.tiling_c1 + self.tiling_bc1_cut_num - 1) // self.tiling_bc1_cut_num
        h_segment = (self.cut_height_num + self.tiling_height_cut_num - 1) // self.tiling_height_cut_num
        w_segment = (self.cut_width_num + self.tiling_width_cut_num - 1) // self.tiling_width_cut_num
        self.core_nc_start.set_as(
            (_core_idx // (self.tiling_height_cut_num * self.tiling_width_cut_num)) * nc_segment)
        self.core_height_start.set_as(
            ((_core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num))
             // self.tiling_width_cut_num) * h_segment)
        self.core_width_start.set_as(
            ((_core_idx % (self.tiling_height_cut_num * self.tiling_width_cut_num))
             % self.tiling_width_cut_num) * w_segment)
        self.core_nc_num.set_as(nc_segment)
        self.core_height_num.set_as(h_segment)
        self.core_width_num.set_as(w_segment)
        with self.tik_instance.if_scope(
                self.core_nc_start + self.core_nc_num >= self.tiling_batch * self.tiling_c1):
            self.core_nc_num.set_as(self.tiling_batch * self.tiling_c1 - self.core_nc_start)
        with self.tik_instance.if_scope(
                self.core_height_start + self.core_height_num >= self.cut_height_num):
            self.core_height_num.set_as(self.cut_height_num - self.core_height_start)
        with self.tik_instance.if_scope(
                self.core_width_start + self.core_width_num >= self.cut_width_num):
            self.core_width_num.set_as(self.cut_width_num - self.core_width_start)
        core_used = self.tiling_width_cut_num * self.tiling_height_cut_num * self.tiling_bc1_cut_num
        with self.tik_instance.if_scope(_core_idx >= core_used):
            self.core_nc_num.set_as(0)
            self.core_height_num.set_as(0)
            self.core_width_num.set_as(0)

    def _init_ub_tensor_for_idx(self, height_idx_len=0, width_idx_len=0):
        """
        compute the ub size of tensors
        """
        height_idx_len = self.height_idx_segment_num if height_idx_len == 0 else height_idx_len
        width_idx_len = self.width_idx_segment_num if width_idx_len == 0 else width_idx_len
        idx_max_len = max(height_idx_len, width_idx_len)
        self.height_idx_ub = self.tik_instance.Tensor("int32", (height_idx_len,),
                                                      name="height_idx", scope=tik.scope_ubuf)
        self.width_idx_ub = self.tik_instance.Tensor("int32", (width_idx_len,),
                                                     name="width_idx", scope=tik.scope_ubuf)

        self.idx_ub_fp32 = self.tik_instance.Tensor("float32", (idx_max_len,),
                                                    name="idx_ub_fp32", scope=tik.scope_ubuf)
        avail_bytes = self.ub_size_bytes - (height_idx_len + width_idx_len + idx_max_len) * 4
        avail_block = avail_bytes // 32 // 2
        self.ub_max_num = avail_block * self.block_num

    def _init_ub_tensor_for_grads(self, mode="all"):
        """
        _init_ub_tensor_for_grads
        """
        if mode in ("all",):
            self.grad_out_ub = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                        name="grad_out_ub", scope=tik.scope_ubuf)
            self.grad_in_gm_ping = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                            name="grad_in_gm_ping", scope=tik.scope_gm)
        if mode in ("ub",):
            self.grad_out_ub = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                        name="grad_out_ub", scope=tik.scope_ubuf)

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
                # `calcu: (idx + 0.5) * scale`
                self.tik_instance.vadds(64, calcu_out_in_idx_tmp_ub, src_idx_fp_ub, 0.5,
                                        vector_repeat_num, 1, 1, 8, 8)
                self.tik_instance.vmuls(64, calcu_out_in_idx_tmp_ub, calcu_out_in_idx_tmp_ub, scale,
                                        vector_repeat_num, 1, 1, 8, 8)
            else:
                # `calcu: idx * scale`
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
                          is_w_align=False, is_big_to_small=False):
        """
        _function_default, run this
        """
        self.tik_instance.set_atomic_add(1)

        self.height_idx_segment_num = 64
        self.width_idx_segment_num = 128
        # cut by output h and output w
        self._init_ub_tensor_for_idx()

        # gen 0-511 to ub fp32
        with self.tik_instance.new_stmt_scope():
            fill_index_in_ub(self.tik_instance, self.idx_ub_fp32, self.width_idx_segment_num)

        # calcu is_src_stride_copy and is_dst_stride_copy use scalar
        scalar_is_src_stride = self.tik_instance.Scalar("int32", name="scalar_is_src_stride", init_value=1)
        scalar_is_dst_stride = self.tik_instance.Scalar("int32", name="scalar_is_dst_stride", init_value=1)

        with self.tik_instance.if_scope(self.tiling_in_height * self.tiling_in_width > self.stride_threshold):
            scalar_is_src_stride.set_as(0)
        with self.tik_instance.if_scope(self.tiling_out_height * self.tiling_out_width > self.stride_threshold):
            scalar_is_dst_stride.set_as(0)
        # calcu is_src_stride_copy and is_dst_stride_copy use scalar end

        # init a scalar for w segment one time
        w_loop_segment = self.tik_instance.Scalar("int32", name="w_loop_segment",
                                                  init_value=self.width_idx_segment_num)
        if is_big_to_small:
            if is_w_align:
                with self.tik_instance.new_stmt_scope():
                    align_num_scalar = self.tik_instance.Scalar("int32", name="align_num_scalar")
                    align_num_scalar.set_as(self.tiling_in_width // self.tiling_out_width)
                    w_loop_segment.set_as(w_loop_segment // align_num_scalar * align_num_scalar)
            else:
                w_loop_segment.set_as(127)
        else:
            if is_w_align:
                # if width is input_w resize to n*input_w, one segment must be n align
                # exp: 24 resize to 48, one segment of width must be 2*n
                with self.tik_instance.new_stmt_scope():
                    align_num_scalar = self.tik_instance.Scalar("int32", name="align_num_scalar")
                    align_num_scalar.set_as(self.tiling_out_width // self.tiling_in_width)
                    w_loop_segment.set_as(w_loop_segment // align_num_scalar * align_num_scalar)

        with self.tik_instance.if_scope(tik.all(self.core_width_num < w_loop_segment, self.core_width_num > 0)):
            w_loop_segment.set_as(self.core_width_num)

        w_loop_num = self.core_width_num // w_loop_segment
        w_tail_num = self.core_width_num % w_loop_segment
        nc_max_segment = self.tik_instance.Scalar("int64", name="nc_max_segment")
        nc_max_segment.set_as(self.ub_max_num // (w_loop_segment * self.grads_shape_c0))
        if is_big_to_small:
            self.tik_instance.scalar_min(nc_max_segment, nc_max_segment, Constant.VADD_MAX_REPEAT_TIMES)
        nc_loop = self.core_nc_num // nc_max_segment
        nc_tail = self.core_nc_num % nc_max_segment

        scalar_idx_fp32 = self.tik_instance.Scalar("float32", name="scalar_idx_fp32")
        # vconv start idx from int32 scalar to fp32 scalar
        self.scalar_vconv_int32_to_fp32(self.core_width_start, scalar_idx_fp32)
        # do vadds 0,1,2,3,4 + fp32_scalar
        self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                (w_loop_segment + 63) // 64, 1, 1, 8, 8)
        self.scalar_vconv_int32_to_fp32(w_loop_segment, scalar_idx_fp32)

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_offset, h_do_len):
            w_gm_offset = w_loop_idx * w_loop_segment + self.core_width_start
            self.calcu_out_in_idx(self.resize_scale_w, self.width_idx_ub,
                                  self.idx_ub_fp32, self.width_idx_segment_num)
            self.tik_instance.vadds(64, self.idx_ub_fp32, self.idx_ub_fp32, scalar_idx_fp32,
                                    (self.width_idx_segment_num + 63) // 64, 1, 1, 8, 8)

            # one segment h and one segment w
            def _do_single_nc(do_nc_num, _nc_loop_idx):
                def _do_one_height(h_idx, output_ub):
                    h_gm_offset = h_idx + h_loop_offset
                    scalar_in_h_idx = self.tik_instance.Scalar("int32", name="scalar_in_h_idx")
                    height_idx_ub = self.tik_instance.Tensor("int32", (64,), name="height_idx", scope=tik.scope_ubuf)
                    height_idx_ub_fp32 = self.tik_instance.Tensor("float32", (64,),
                                                                  name="height_idx_ub_fp32", scope=tik.scope_ubuf)
                    self.tik_instance.vector_dup(64, height_idx_ub_fp32, 0, 1, 1, 8)
                    util_tik_comm_func.tik_func_vector(self.tik_instance, height_idx_ub,
                                                       h_idx + self.core_height_start, 64)
                    util_tik_comm_func.tik_func_vconv(self.tik_instance, height_idx_ub_fp32,
                                                      height_idx_ub, 64)

                    self.calcu_out_in_idx(self.resize_scale_h, height_idx_ub, height_idx_ub_fp32, 1)
                    scalar_in_h_idx.set_as(height_idx_ub[0])

                    with self.tik_instance.if_scope(scalar_is_src_stride == 0):
                        with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                            data_move_ubuf_offset = (w_do_len * self.grads_shape_c0) * _segment_idx
                            nc_gm_input_offset = \
                                (_nc_loop_idx * nc_max_segment + self.core_nc_start + _segment_idx) \
                                * self.tiling_in_width * self.tiling_in_height
                            data_move_gm_offset = \
                                nc_gm_input_offset + h_gm_offset * self.tiling_in_width + w_gm_offset
                            self.tik_instance.data_move(output_ub[data_move_ubuf_offset],
                                                        self.grads_gm[data_move_gm_offset * self.grads_shape_c0],
                                                        0, 1,
                                                        w_do_len * self.grads_shape_c0 // self.block_num, 0, 0)
                    with self.tik_instance.else_scope():
                        data_move_ubuf_offset = 0
                        nc_gm_input_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                                             * self.tiling_in_width * self.tiling_in_height
                        data_move_gm_offset = nc_gm_input_offset + h_gm_offset * self.tiling_in_width + w_gm_offset
                        data_move_burst_num = do_nc_num
                        data_move_burst_len = w_do_len * self.grads_shape_c0 // self.block_num
                        data_move_src_stride = (self.tiling_in_width * self.tiling_in_height - w_do_len) \
                                               * self.grads_shape_c0 // self.block_num
                        self.tik_instance.data_move(output_ub[data_move_ubuf_offset],
                                                    self.grads_gm[data_move_gm_offset * self.grads_shape_c0],
                                                    0,
                                                    data_move_burst_num,
                                                    data_move_burst_len,
                                                    data_move_src_stride, 0)

                    # When Win > Wout, set is_big_to_small as "True", default set it as "False"
                    if is_big_to_small:
                        burst_len = self.grads_shape_c0 // self.block_num
                        # When the result of Win/Wout is integer and less than 120, set is_w_align as "True", default
                        # set it as "False"
                        if not is_w_align:
                            scalar_out_w_idx = self.tik_instance.Scalar("int32", name="scalar_out_w_idx",
                                                                        init_value=0)

                            with self.tik_instance.if_scope(w_do_len == 1):
                                scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                                scalar_in_w_idx.set_as(self.width_idx_ub[0])
                                nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start)\
                                                * self.tiling_out_width * self.tiling_out_height
                                output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                   self.tiling_out_width + scalar_in_w_idx
                                burst_num = do_nc_num
                                ubuf_burst_stride = 0
                                gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height - 1) \
                                                      * self.grads_shape_c0 // self.block_num
                                self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                        * self.grads_shape_c0],
                                                            output_ub[0],
                                                            0, burst_num, burst_len, ubuf_burst_stride,
                                                            gm_out_burst_stride)

                            with self.tik_instance.else_scope():
                                with self.tik_instance.for_range(0, w_do_len - 1) as w_idx:
                                    scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                                    scalar_in_w_idx_next = self.tik_instance.Scalar("int32",
                                                                                    name="scalar_in_w_idx_next")
                                    scalar_in_w_idx.set_as(self.width_idx_ub[w_idx])
                                    scalar_in_w_idx_next.set_as(self.width_idx_ub[w_idx + 1])

                                    with self.tik_instance.if_scope(scalar_in_w_idx == scalar_in_w_idx_next):
                                        rep_stride = w_do_len * self.grads_shape_c0 // self.block_num
                                        self.tik_instance.vadd(self.grads_shape_c0, \
                                                               output_ub[scalar_out_w_idx * self.grads_shape_c0],\
                                                               output_ub[scalar_out_w_idx * self.grads_shape_c0], \
                                                               output_ub[(w_idx + 1) * self.grads_shape_c0],
                                                               do_nc_num, 1, 1, 1, rep_stride, rep_stride, rep_stride)
                                        with self.tik_instance.if_scope(w_idx == w_do_len - 2):
                                            with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                                                with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                                    burst_num = 1
                                                    nc_ubuf_offset = w_do_len * _segment_idx + scalar_out_w_idx
                                                    nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                                    + _segment_idx) * self.tiling_out_width \
                                                                    * self.tiling_out_height
                                                    output_gm_offset = nc_gm_offset + scalar_in_h_idx \
                                                                       * self.tiling_out_width + scalar_in_w_idx

                                                    self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                                * self.grads_shape_c0],
                                                                                output_ub[nc_ubuf_offset \
                                                                                * self.grads_shape_c0],
                                                                                0, burst_num, burst_len, 0, 0)
                                            with self.tik_instance.else_scope():
                                                nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start)\
                                                                * self.tiling_out_width * self.tiling_out_height
                                                output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                                   self.tiling_out_width + scalar_in_w_idx
                                                burst_num = do_nc_num
                                                ubuf_burst_stride = (w_do_len - 1) * self.grads_shape_c0 \
                                                                    // self.block_num
                                                gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height \
                                                                       - 1) * self.grads_shape_c0 // self.block_num
                                                self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                                        * self.grads_shape_c0],
                                                                            output_ub[scalar_out_w_idx \
                                                                                      * self.grads_shape_c0],
                                                                            0, burst_num, burst_len, ubuf_burst_stride,
                                                                            gm_out_burst_stride)
                                    with self.tik_instance.else_scope():
                                        with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                                            with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                                burst_num = 1
                                                nc_ubuf_offset = w_do_len * _segment_idx + scalar_out_w_idx
                                                nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                                + _segment_idx) * self.tiling_out_width \
                                                                * self.tiling_out_height
                                                output_gm_offset = nc_gm_offset + scalar_in_h_idx \
                                                                   * self.tiling_out_width + scalar_in_w_idx

                                                self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                            * self.grads_shape_c0],
                                                                            output_ub[nc_ubuf_offset \
                                                                            * self.grads_shape_c0],
                                                                            0, burst_num, burst_len, 0, 0)
                                            scalar_out_w_idx.set_as(w_idx + 1)
                                            with self.tik_instance.if_scope(w_idx == w_do_len - 2):
                                                with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                                    burst_num = 1
                                                    nc_ubuf_offset = w_do_len * _segment_idx + scalar_out_w_idx
                                                    nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                                    + _segment_idx) * self.tiling_out_width \
                                                                    * self.tiling_out_height
                                                    output_gm_offset = nc_gm_offset + scalar_in_h_idx \
                                                                       * self.tiling_out_width + scalar_in_w_idx_next

                                                    self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                                * self.grads_shape_c0],
                                                                                output_ub[nc_ubuf_offset \
                                                                                * self.grads_shape_c0],
                                                                                0, burst_num, burst_len, 0, 0)
                                        with self.tik_instance.else_scope():
                                            nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                                                           * self.tiling_out_width * self.tiling_out_height
                                            output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                               self.tiling_out_width + scalar_in_w_idx
                                            burst_num = do_nc_num
                                            ubuf_burst_stride = (w_do_len - 1) * self.grads_shape_c0 // self.block_num
                                            gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height - 1) \
                                                                  * self.grads_shape_c0 // self.block_num
                                            self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                                    * self.grads_shape_c0],
                                                                        output_ub[scalar_out_w_idx \
                                                                                  * self.grads_shape_c0],
                                                                        0, burst_num, burst_len, ubuf_burst_stride,
                                                                        gm_out_burst_stride)
                                            scalar_out_w_idx.set_as(w_idx + 1)
                                            with self.tik_instance.if_scope(w_idx == w_do_len - 2):
                                                output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                                   self.tiling_out_width + scalar_in_w_idx_next
                                                self.tik_instance.data_move(self.out_gm[output_gm_offset * \
                                                                                        self.grads_shape_c0],
                                                                            output_ub[scalar_out_w_idx \
                                                                            * self.grads_shape_c0],
                                                                            0, burst_num, burst_len, ubuf_burst_stride,
                                                                            gm_out_burst_stride)

                        else:
                            w_align_num = self.tiling_in_width // self.tiling_out_width
                            repeat_times = w_do_len // w_align_num
                            with self.tik_instance.for_range(0, repeat_times) as repeat_idx:
                                with self.tik_instance.for_range(0, w_align_num - 1) as w_align_idx:
                                    # The range of rep_stride is [0, 255] in vadd instruction
                                    output_ub_offset = repeat_idx * w_align_num
                                    rep_stride = w_do_len * self.grads_shape_c0 // self.block_num
                                    self.tik_instance.vadd(self.grads_shape_c0, output_ub[output_ub_offset \
                                                           * self.grads_shape_c0],
                                                           output_ub[output_ub_offset * self.grads_shape_c0],
                                                           output_ub[(output_ub_offset + w_align_idx + 1) \
                                                                      * self.grads_shape_c0],
                                                           do_nc_num, 1, 1, 1, rep_stride, rep_stride, rep_stride)

                                with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                                    with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                        nc_ubuf_offset = w_do_len * _segment_idx + repeat_idx * w_align_num
                                        nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                        + _segment_idx) * self.tiling_out_width * self.tiling_out_height
                                        output_gm_offset = nc_gm_offset + scalar_in_h_idx * self.tiling_out_width \
                                                           + w_gm_offset // w_align_num + repeat_idx
                                        burst_num = 1
                                        self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                    output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                    0, burst_num, burst_len,
                                                                    0, 0)
                                with self.tik_instance.else_scope():
                                    nc_ubuf_offset = repeat_idx * w_align_num
                                    nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                                                   * self.tiling_out_width * self.tiling_out_height
                                    output_gm_offset = nc_gm_offset + scalar_in_h_idx * self.tiling_out_width \
                                                       + w_gm_offset // w_align_num + repeat_idx
                                    burst_num = do_nc_num
                                    ubuf_burst_stride = (w_do_len - 1) * self.grads_shape_c0 // self.block_num
                                    gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height - 1) \
                                                           * self.grads_shape_c0 // self.block_num
                                    self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                0, burst_num, burst_len,
                                                                ubuf_burst_stride, gm_out_burst_stride)
                    else:
                        # When the result of Wout/Win is integer, set is_w_align as "True", default set it as "False"
                        if not is_w_align:
                            with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                                with self.tik_instance.for_range(0, w_do_len) as w_idx:
                                    scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                                    scalar_in_w_idx.set_as(self.width_idx_ub[w_idx])
                                    burst_num = 1
                                    burst_len = self.grads_shape_c0 // self.block_num
                                    with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                        nc_ubuf_offset = w_do_len * _segment_idx + w_idx
                                        nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                        + _segment_idx) * self.tiling_out_width * self.tiling_out_height
                                        output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                           self.tiling_out_width + scalar_in_w_idx

                                        self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                    output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                    0, burst_num, burst_len,
                                                                    0, 0)
                            with self.tik_instance.else_scope():
                                with self.tik_instance.for_range(0, w_do_len) as w_idx:
                                    scalar_in_w_idx = self.tik_instance.Scalar("int32", name="scalar_in_w_idx")
                                    scalar_in_w_idx.set_as(self.width_idx_ub[w_idx])
                                    nc_ubuf_offset = w_do_len * self.grads_shape_c0
                                    burst_num = do_nc_num
                                    burst_len = self.grads_shape_c0 // self.block_num
                                    ubuf_burst_stride = nc_ubuf_offset // self.block_num - burst_len
                                    gm_out_burst_stride = self.tiling_out_width * self.tiling_out_height * \
                                                          self.grads_shape_c0 // self.block_num - burst_len
                                    nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                                                   * self.tiling_out_width * self.tiling_out_height
                                    output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                       self.tiling_out_width + scalar_in_w_idx

                                    self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                output_ub[w_idx * self.grads_shape_c0],
                                                                0, burst_num, burst_len,
                                                                ubuf_burst_stride, gm_out_burst_stride)

                        else:
                            w_align_num = self.tiling_out_width // self.tiling_in_width

                            with self.tik_instance.if_scope(w_align_num == 1):
                                with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                                    with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                        burst_num = 1
                                        burst_len = w_do_len * self.grads_shape_c0 // self.block_num
                                        nc_ubuf_offset = w_do_len * _segment_idx
                                        nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                        + _segment_idx) * self.tiling_out_width * self.tiling_out_height
                                        output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                           self.tiling_out_width + self.core_width_start + w_do_len \
                                                           * _segment_idx

                                        self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                    output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                    0, burst_num, burst_len,
                                                                    0, 0)

                                with self.tik_instance.else_scope():
                                    burst_num = do_nc_num
                                    burst_len = w_do_len * self.grads_shape_c0 // self.block_num
                                    nc_ubuf_offset = 0
                                    nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                                                   * self.tiling_out_width * self.tiling_out_height
                                    output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                       self.tiling_out_width + self.core_width_start
                                    ubuf_burst_stride = 0
                                    gm_out_burst_stride = (self.tiling_out_width * self.tiling_out_height - w_do_len) \
                                                          * self.grads_shape_c0 // self.block_num

                                    self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                0, burst_num, burst_len,
                                                                ubuf_burst_stride, gm_out_burst_stride)

                            with self.tik_instance.else_scope():
                                with self.tik_instance.if_scope(scalar_is_dst_stride == 0):
                                    with self.tik_instance.for_range(0, w_do_len) as w_input_idx:
                                        with self.tik_instance.for_range(0, do_nc_num) as _segment_idx:
                                            burst_num = 1
                                            burst_len = self.grads_shape_c0 // self.block_num
                                            nc_ubuf_offset = w_do_len * _segment_idx + w_input_idx
                                            nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start \
                                                            + _segment_idx) * self.tiling_out_width \
                                                            * self.tiling_out_height
                                            output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                               self.tiling_out_width + (self.core_width_start \
                                                               + w_do_len * _segment_idx + w_input_idx) * w_align_num

                                            self.tik_instance.data_move(self.out_gm[output_gm_offset \
                                                                                    * self.grads_shape_c0],
                                                                        output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                        0, burst_num, burst_len,
                                                                        0, 0)

                                with self.tik_instance.else_scope():
                                    with self.tik_instance.for_range(0, w_do_len) as w_input_idx:
                                        burst_num = do_nc_num
                                        burst_len = self.grads_shape_c0 // self.block_num
                                        nc_ubuf_offset = w_input_idx
                                        ubuf_burst_stride = w_do_len * self.grads_shape_c0 // self.block_num - burst_len
                                        gm_out_burst_stride = self.tiling_out_width * self.tiling_out_height * \
                                                              self.grads_shape_c0 // self.block_num - burst_len

                                        nc_gm_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start) \
                                                       * self.tiling_out_width * self.tiling_out_height
                                        output_gm_offset = nc_gm_offset + scalar_in_h_idx * \
                                                           self.tiling_out_width + (self.core_width_start + \
                                                           w_input_idx) * w_align_num
                                        self.tik_instance.data_move(self.out_gm[output_gm_offset * self.grads_shape_c0],
                                                                    output_ub[nc_ubuf_offset * self.grads_shape_c0],
                                                                    0, burst_num, burst_len,
                                                                    ubuf_burst_stride, gm_out_burst_stride)

                grad_out_ub_ping = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                            name="grad_out_ub_ping", scope=tik.scope_ubuf)
                grad_out_ub_pang = self.tik_instance.Tensor(self.grads_dtype, (self.ub_max_num,),
                                                            name="grad_out_ub_pang", scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, h_do_len // 2) as _h_idx:
                    _do_one_height(_h_idx * 2, grad_out_ub_ping)
                    _do_one_height(_h_idx * 2 + 1, grad_out_ub_pang)
                with self.tik_instance.if_scope(h_do_len % 2 == 1):
                    _do_one_height(h_do_len - 1, grad_out_ub_ping)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            h_loop_segment_start = h_loop_idx * self.height_idx_segment_num + self.core_height_start
            h_gm_offset = h_loop_segment_start
            # calcu h idx

            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, w_loop_segment, h_gm_offset, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_gm_offset, h_do_len)

        _run_h_loop_default(0, self.core_height_num)
        self.tik_instance.set_atomic_add(0)

    def _function_hw_to_nhnw_resize(self):
        """
        _function_hw_to_nhnw_resize, when `tiling key = 111000, run this`
        """
        # h boardcast base input_h cut
        size_h_n = self.tiling_out_height // self.tiling_in_height
        size_w_n = self.tiling_out_width // self.tiling_in_width
        output_w_size = self.core_width_num * size_w_n
        w_output_size_one_line = self.tik_instance.Scalar("int64", name="input_w_size", init_value=0)
        w_output_size_one_line.set_as(output_w_size)

        with self.tik_instance.if_scope(
                tik.all(self.ub_max_num < output_w_size * self.grads_shape_c0,
                        self.core_width_num > 0)):
            w_output_size_one_line.set_as((self.ub_max_num // self.grads_shape_c0 // size_w_n) * size_w_n)
        with self.tik_instance.if_scope(w_output_size_one_line == 0):
            w_output_size_one_line.set_as((self.ub_max_num // self.grads_shape_c0 // size_w_n) * size_w_n)
        _w_loop_num = self.core_width_num // (w_output_size_one_line // size_w_n)
        _w_tail_num = self.core_width_num % (w_output_size_one_line // size_w_n)
        _segment_h_num = self.ub_max_num // self.grads_shape_c0 // w_output_size_one_line
        _h_loop_num = self.core_height_num // _segment_h_num
        _h_tail_num = self.core_height_num % _segment_h_num

        def _run_h_loop(h_loop_idx, h_do_len, w_start_offset, w_do_len, nc_idx):
            h_segment_start = h_loop_idx * _segment_h_num + self.core_height_start
            nc_segment_start = nc_idx + self.core_nc_start
            self._init_ub_tensor_for_grads("ub")

            # copy h * w input to ub
            data_move_gm_offset = \
                nc_segment_start * self.tiling_in_height * self.tiling_in_width + \
                h_segment_start * self.tiling_in_width + w_start_offset
            data_move_burst_num = h_do_len
            data_move_burst_len = w_do_len * self.grads_shape_c0 // self.block_num
            data_move_src_stride = (self.tiling_in_width - w_do_len) * self.grads_shape_c0 // self.block_num
            data_move_dst_stride = 0

            self.tik_instance.data_move(self.grad_out_ub,
                                        self.grads_gm[data_move_gm_offset * self.grads_shape_c0],
                                        0,
                                        data_move_burst_num,
                                        data_move_burst_len,
                                        data_move_src_stride,
                                        data_move_dst_stride)

            with self.tik_instance.for_range(0, h_do_len) as _h_idx:
                # copy output according to h one by one
                data_move_src_offset = _h_idx * w_do_len * self.grads_shape_c0
                data_move_dst_offset = \
                    nc_segment_start * self.tiling_out_height * self.tiling_out_width + \
                    h_segment_start * size_h_n * self.tiling_out_width + w_start_offset * size_w_n + \
                    _h_idx * size_h_n * self.tiling_out_width
                data_move_burst_num = w_do_len
                data_move_burst_len = self.grads_shape_c0 // self.block_num
                data_move_src_stride = 0
                data_move_dst_stride = (size_w_n - 1) * self.grads_shape_c0 // self.block_num

                self.tik_instance.data_move(self.out_gm[data_move_dst_offset * self.grads_shape_c0:],
                                            self.grad_out_ub[data_move_src_offset:],
                                            0,
                                            data_move_burst_num,
                                            data_move_burst_len,
                                            data_move_src_stride,
                                            data_move_dst_stride)

        def _run_w_loop(w_loop_idx, input_w_len):
            w_segment_start = w_loop_idx * (w_output_size_one_line // size_w_n) + self.core_width_start
            with self.tik_instance.for_range(0, self.core_nc_num) as nc_idx:
                with self.tik_instance.for_range(0, _h_loop_num, thread_num=2) as _h_loop_idx:
                    _run_h_loop(_h_loop_idx, _segment_h_num, w_segment_start, input_w_len, nc_idx)
                with self.tik_instance.if_scope(_h_tail_num != 0):
                    _run_h_loop(_h_loop_num, _h_tail_num, w_segment_start, input_w_len, nc_idx)

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
            width_input_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                        name="width_input_fp32", scope=tik.scope_ubuf)
            height_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                          name="height_input_int32", scope=tik.scope_ubuf)
            width_input_int32 = self.tik_instance.Tensor("int32", (self.block_num * 2,),
                                                         name="width_input_int32", scope=tik.scope_ubuf)
            height_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                          name="height_output_fp32", scope=tik.scope_ubuf)
            width_output_fp32 = self.tik_instance.Tensor("float32", (self.block_num * 2,),
                                                         name="width_output_fp32", scope=tik.scope_ubuf)

            height_input_int32[0].set_as(self.tiling_out_height)
            width_input_int32[0].set_as(self.tiling_out_width)
            height_input_int32[self.block_num].set_as(self.tiling_in_height)
            width_input_int32[self.block_num].set_as(self.tiling_in_width)

            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_input_fp32,
                                              height_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, width_input_fp32,
                                              width_input_int32, 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, height_output_fp32,
                                              height_input_int32[self.block_num:], 1)
            util_tik_comm_func.tik_func_vconv(self.tik_instance, width_output_fp32,
                                              width_input_int32[self.block_num:], 1)

            with self.tik_instance.if_scope(tik.all(self.align_corners, self.tiling_out_height > 1)):
                self.tik_instance.vadds(1, height_output_fp32, height_output_fp32, -1.0, 1, 1, 1, 8, 8)
                self.tik_instance.vadds(1, height_input_fp32, height_input_fp32, -1.0, 1, 1, 1, 8, 8)

            self.tik_instance.vdiv(1, height_input_fp32, height_input_fp32, height_output_fp32,
                                   1, 1, 1, 1, 8, 8, 8)
            self.resize_scale_h.set_as(height_input_fp32[0])

            with self.tik_instance.if_scope(tik.all(self.align_corners, self.tiling_out_width > 1)):
                self.tik_instance.vadds(1, width_output_fp32, width_output_fp32, -1.0, 1, 1, 1, 8, 8)
                self.tik_instance.vadds(1, width_input_fp32, width_input_fp32, -1.0, 1, 1, 1, 8, 8)
            self.tik_instance.vdiv(1, width_input_fp32, width_input_fp32, width_output_fp32,
                                   1, 1, 1, 1, 8, 8, 8)
            self.resize_scale_w.set_as(width_input_fp32[0])

    def _do_resize_base_tiling_key(self):
        """
        tiling braches
        """
        # calcu scale for h and w
        self.calculate_scale()

        # tiling_key format: 000000
        # 1. Reserved, default 1
        # 2. h align flag, 0: `h -> x.x*h, 1: h -> nh, 2: nh -> h, 3: h = h`
        # 3. w align flag, 0: `w -> x.x*w, 1: w -> nw, 2: nw -> w, 3: w = w`
        # 4. src stride flag, 0: can not copy with stride 1: can copy with stride
        # 5. des stride flag, 0: can not copy with stride 1: can copy with stride
        # 6. Reserved, default 0

        with self.tik_instance.if_scope(self.tiling_key == 100000):
            with self.tik_instance.new_stmt_scope():
                self._function_default(is_src_stride_copy=False, is_dst_stride_copy=False, is_w_align=False)
        with self.tik_instance.if_scope(self.tiling_key == 101000):
            with self.tik_instance.new_stmt_scope():
                self._function_default(is_src_stride_copy=False, is_dst_stride_copy=False, is_w_align=True)
        with self.tik_instance.if_scope(self.tiling_key == 100010):
            with self.tik_instance.new_stmt_scope():
                self._function_default(is_src_stride_copy=False, is_dst_stride_copy=False, is_w_align=False,
                                       is_big_to_small=True)
        with self.tik_instance.if_scope(self.tiling_key == 100110):
            with self.tik_instance.new_stmt_scope():
                self._function_default(is_src_stride_copy=False, is_dst_stride_copy=False, is_w_align=True,
                                       is_big_to_small=True)
        with self.tik_instance.if_scope(self.tiling_key == 111000):
            # tiling_key is 111000, mean: h,w resize to nh, mw
            with self.tik_instance.new_stmt_scope():
                self._function_hw_to_nhnw_resize()

    def _do_resize(self):
        """
        main process of _do_resize
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as _core_idx:
            with self.tik_instance.new_stmt_scope():
                tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                     name="tiling_ub", scope=tik.scope_ubuf)
                self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
                self._tiling_args(tiling_ub, "read")
            self._core_scalar_args(_core_idx)
            self._do_resize_base_tiling_key()

    def resize_nearest_neighbor_v2_grad_operator(self):
        """
        resize_nearest_neighbor_v2_grad_operator
        """
        self._do_resize()
        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True
                     }

        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.ai_core_num,
                                                            "max_w_len": self.ub_max_num // self.grads_shape_c0,
                                                            "align_corners": int(self.align_corners),
                                                            "half_pixel_centers": int(self.half_pixel_centers)})

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.grads_gm, self.size_gm),
                                   outputs=(self.out_gm,),
                                   flowtable=(self.tiling_gm,), config=opt_config)

        return self.tik_instance


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


@register_operator("ResizeNearestNeighborV2Grad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def resize_nearest_neighbor_v2_grad(grads, size, y, align_corners=False, half_pixel_centers=False,
                                    kernel_name="resize_nearest_neighbor_grad"):
    """Resize `grads` to `size` using nearest neighbor interpolation.

    Parameters
    ----------
    grads: dict
        the dict of input, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float32'
    size: list
        the list of input, the height and width of output tensor
        only support 5HD and dtype supports 'int32', 'int64'
    y: dict
        the dict of output, include shape of input_tensor which layout
        only support 5HD and dtype supports 'float32'
    align_corners: bool
        whether align_corners
    half_pixel_centers: bool
        whether half_pixel_centers
    kernel_name: str
        cce kernel name, default value is `resize_nearest_neighbor_grad`

    Returns
    -------
    tik_instance
    """
    obj = ResizeNearestNeighborV2Grad(grads, size, y, align_corners, half_pixel_centers,
                                      kernel_name)
    return obj.resize_nearest_neighbor_v2_grad_operator()
