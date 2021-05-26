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
resize_bilinear_v2.py
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.dynamic.resize_bilinear_v2_1981 import resize_bilinear_v2_with_gatherb

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
class ResizeBilinearV2:
    """
    Function: use to store ResizeBilinearV2 base parameters
    Modify: 2021-01-20
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
        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", self.images_dtype)
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
        self.in_images_ub = None
        self.out_images_ub = None

    class CommomScalar():
        """
        define some scalar
        """

        def __init__(self, tik_instance):
            self.dst_h = tik_instance.Scalar(dtype="float32", name="dst_h")
            self.dst_w = tik_instance.Scalar(dtype="float32", name="dst_w")

            self.src_h_start = tik_instance.Scalar(dtype="int32", name="src_h_start")
            self.src_w_start = tik_instance.Scalar(dtype="int32", name="src_w_start")
            self.src_h_end = tik_instance.Scalar(dtype="int32", name="src_h_end")
            self.src_w_end = tik_instance.Scalar(dtype="int32", name="src_w_end")

            self.src_h = tik_instance.Scalar(dtype="float32", name="src_h")
            self.h_idx = tik_instance.Scalar(dtype="int32", name="h_idx")
            self.h_stride = tik_instance.Scalar(dtype="int32", name="h_stride")
            self.hl_ratio = tik_instance.Scalar(dtype="float32", name="hl_ratio")
            self.hr_ratio = tik_instance.Scalar(dtype="float32", name="hr_ratio")

            self.src_w = tik_instance.Scalar(dtype="float32", name="src_w")
            self.w_idx = tik_instance.Scalar(dtype="int32", name="w_idx")
            self.w_stride = tik_instance.Scalar(dtype="int32", name="w_stride")
            self.wl_ratio = tik_instance.Scalar(dtype="float32", name="wl_ratio")
            self.wr_ratio = tik_instance.Scalar(dtype="float32", name="wr_ratio")
            self.l_ratio = tik_instance.Scalar(dtype="float32", name="l_ratio")
            self.r_ratio = tik_instance.Scalar(dtype="float32", name="r_ratio")

            self.images_idx00 = tik_instance.Scalar(dtype="int32", name="images_idx00", init_value=0)
            self.images_idx01 = tik_instance.Scalar(dtype="int32", name="images_idx01", init_value=0)
            self.images_idx10 = tik_instance.Scalar(dtype="int32", name="images_idx10")
            self.images_idx11 = tik_instance.Scalar(dtype="int32", name="images_idx11")

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
        with self.tik_instance.if_scope(self.tiling_key == 1):
            align_num = self.tiling_out_weight // self.tiling_in_weight
            self.core_weight_num.set_as(
                self.core_weight_num + (self.core_weight_start - (self.core_weight_start // align_num) * align_num))
            self.core_weight_start.set_as((self.core_weight_start // align_num) * align_num)

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

    def get_src_index(self, scale, dst_index_scalar, src_scalar):
        if self.half_pixel_centers:
            src_scalar.set_as(scale * (dst_index_scalar + 0.5) - 0.5)
            with self.tik_instance.if_scope(src_scalar < 0):
                src_scalar.set_as(0)
            with self.tik_instance.else_scope():
                pass
        else:
            src_scalar.set_as(scale * dst_index_scalar)

    def get_ratio(self, src_index, idx_scalar, l_ratio_scalar, r_ratio_scalar):
        r_ratio = src_index - idx_scalar
        l_ratio = 1 - r_ratio
        l_ratio_scalar.set_as(l_ratio)
        r_ratio_scalar.set_as(r_ratio)

    def get_stride(self, idx_scalar, stride_scalar, image_size):
        if idx_scalar < image_size - 1:
            stride_scalar.set_as(1)
        else:
            stride_scalar.set_as(0)

    def calculate_h_start(self, scalar, dst_h_start):
        scalar.dst_h.set_as(dst_h_start)
        self.get_src_index(self.resize_scale_h, scalar.dst_h, scalar.src_h)
        self.tik_instance.scalar_conv("floor", scalar.src_h_start, scalar.src_h)

    def calculate_h_end(self, scalar, dst_h_end):
        scalar.dst_h.set_as(dst_h_end)
        self.get_src_index(self.resize_scale_h, scalar.dst_h, scalar.src_h)
        self.tik_instance.scalar_conv("floor", scalar.h_idx, scalar.src_h)
        self.get_stride(scalar.h_idx, scalar.h_stride, self.tiling_in_height)
        scalar.src_h_end.set_as(scalar.h_idx + scalar.h_stride)

    def calculate_w_start(self, scalar, dst_w_start):
        scalar.dst_w.set_as(dst_w_start)
        self.get_src_index(self.resize_scale_w, scalar.dst_w, scalar.src_w)
        self.tik_instance.scalar_conv("floor", scalar.src_w_start, scalar.src_w)

    def calculate_w_end(self, scalar, dst_w_end):
        scalar.dst_w.set_as(dst_w_end)
        self.get_src_index(self.resize_scale_w, scalar.dst_w, scalar.src_w)
        self.tik_instance.scalar_conv("floor", scalar.w_idx, scalar.src_w)
        self.get_stride(scalar.w_idx, scalar.w_stride, self.tiling_in_weight)
        scalar.src_w_end.set_as(scalar.w_idx + scalar.w_stride)

    def calculate_h(self, scalar, dst_h):
        scalar.dst_h.set_as(dst_h)
        self.get_src_index(self.resize_scale_h, scalar.dst_h, scalar.src_h)
        self.tik_instance.scalar_conv("floor", scalar.h_idx, scalar.src_h)
        self.get_stride(scalar.h_idx, scalar.h_stride, self.tiling_in_height)
        self.get_ratio(scalar.src_h, scalar.h_idx, scalar.hl_ratio, scalar.hr_ratio)

    def calculate_w(self, scalar, dst_w):
        scalar.dst_w.set_as(dst_w)
        self.get_src_index(self.resize_scale_w, scalar.dst_w, scalar.src_w)
        self.tik_instance.scalar_conv("floor", scalar.w_idx, scalar.src_w)
        self.get_stride(scalar.w_idx, scalar.w_stride, self.tiling_in_weight)
        self.get_ratio(scalar.src_w, scalar.w_idx, scalar.wl_ratio, scalar.wr_ratio)

    def _function_default_1(self):
        """
        when tiling_key = 1
        """
        self.height_idx_sigment_num = 8
        self.weight_idx_sigment_num = 64
        h_loop_num = self.core_height_num // self.height_idx_sigment_num
        w_loop_num = self.core_weight_num // self.weight_idx_sigment_num
        h_tail_num = self.core_height_num % self.height_idx_sigment_num
        w_tail_num = self.core_weight_num % self.weight_idx_sigment_num

        nc_total = self.core_nc_num
        input_hw_size = self.tik_instance.Scalar("int64", name="input_hw_size")
        input_hw_size.set_as(self.weight_idx_sigment_num * self.height_idx_sigment_num)
        with self.tik_instance.if_scope(tik.all(self.core_weight_num * self.core_height_num < input_hw_size,
                                                self.core_weight_num * self.core_height_num > 0)):
            input_hw_size.set_as(self.core_weight_num * self.core_height_num)

        input_hw_size = input_hw_size * self.images_shape_c0
        nc_max_segment = self.ub_max_num // input_hw_size
        nc_loop = nc_total // nc_max_segment
        nc_tail = nc_total % nc_max_segment

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_idx, h_do_len):
            h_loop_offset = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_start = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_end = h_loop_sigment_start + h_do_len
            w_loop_offset = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_start = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_end = w_loop_sigment_start + w_do_len

            scalar = self.CommomScalar(self.tik_instance)
            self.calculate_h_start(scalar, h_loop_sigment_start)
            self.calculate_h_end(scalar, h_loop_sigment_end)
            self.calculate_w_start(scalar, w_loop_sigment_start)
            self.calculate_w_end(scalar, w_loop_sigment_end)

            def _do_single_nc(do_nc_num, _nc_loop_idx):
                with self.tik_instance.for_range(0, do_nc_num) as _sigment_idx:
                    nc_gm_images_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                          * self.tiling_in_weight * self.tiling_in_height
                    nc_gm_images_start = self.tik_instance.Scalar("int32", name="nc_gm_images_start")
                    nc_gm_images_start.set_as(nc_gm_images_offset + scalar.src_h_start * self.tiling_in_weight +
                                              scalar.src_w_start)
                    data_move_gm_burst = (scalar.src_w_end - scalar.src_w_start + 1) * self.images_shape_c0
                    data_move_gm_nburst = scalar.src_h_end - scalar.src_h_start + 1
                    data_move_gm_stride = (self.tiling_in_weight + scalar.src_w_start - scalar.src_w_end - 1) * \
                                          self.images_shape_c0
                    self.in_images_ub = self.tik_instance.Tensor("float32", (self.height_idx_sigment_num *
                                                                             self.weight_idx_sigment_num *
                                                                             self.images_shape_c0 * 4,),
                                                                 name="in_images_ub", scope=tik.scope_ubuf)
                    self.out_images_ub = self.tik_instance.Tensor("float32", (self.height_idx_sigment_num *
                                                                              self.weight_idx_sigment_num *
                                                                              self.images_shape_c0,),
                                                                  name="out_images_ub", scope=tik.scope_ubuf)
                    self.tik_instance.data_move(self.in_images_ub, self.images_gm[nc_gm_images_start], 0,
                                                data_move_gm_nburst, data_move_gm_burst, data_move_gm_stride, 0)
                    with self.tik_instance.for_range(0, h_do_len) as h_idx:
                        h_idx_per_nc = h_idx + h_loop_offset
                        self.calculate_h(scalar, h_idx_per_nc)
                        with self.tik_instance.for_range(0, w_do_len) as w_idx:
                            w_idx_per_nc = w_idx + w_loop_offset
                            self.calculate_w(scalar, w_idx_per_nc)
                            scalar.images_idx00.set_as((scalar.src_h - scalar.src_h_start)
                                                       * (scalar.src_w_end - scalar.src_w_start + 1) +
                                                       scalar.src_w - scalar.src_w_start)
                            scalar.images_idx01.set_as(scalar.images_idx00 + scalar.w_stride)
                            scalar.images_idx10.set_as(scalar.images_idx00 + scalar.h_stride *
                                                       (scalar.src_w_end - scalar.src_w_start + 1))
                            scalar.images_idx11.set_as(scalar.images_idx10 + scalar.w_stride)
                            out_images_00_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                        name="out_images_00_ub",
                                                                        scope=tik.scope_ubuf)
                            self.tik_instance.vec_muls(self.images_shape_c0, out_images_00_ub,
                                                       self.in_images_ub[scalar.images_idx00],
                                                       scalar.wl_ratio, 1, 0, 0)
                            out_images_01_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                        name="out_images_01_ub",
                                                                        scope=tik.scope_ubuf)
                            self.tik_instance.vec_muls(self.images_shape_c0, out_images_01_ub,
                                                       self.in_images_ub[scalar.images_idx01],
                                                       scalar.wr_ratio, 1, 0, 0)
                            out_images_10_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                        name="out_images_10_ub",
                                                                        scope=tik.scope_ubuf)
                            self.tik_instance.vec_muls(self.images_shape_c0, out_images_10_ub,
                                                       self.in_images_ub[scalar.images_idx10],
                                                       scalar.wl_ratio, 1, 0, 0)
                            out_images_11_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                        name="out_images_11_ub",
                                                                        scope=tik.scope_ubuf)
                            self.tik_instance.vec_muls(self.images_shape_c0, out_images_11_ub,
                                                       self.in_images_ub[scalar.images_idx11],
                                                       scalar.wr_ratio, 1, 0, 0)
                            out_images_0_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                       name="out_images_0_ub",
                                                                       scope=tik.scope_ubuf)
                            self.tik_instance.vec_add(self.images_shape_c0, out_images_0_ub, out_images_00_ub,
                                                      out_images_01_ub, 1, 0, 0, 0)
                            out_images_1_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                       name="out_images_1_ub",
                                                                       scope=tik.scope_ubuf)
                            self.tik_instance.vec_add(self.images_shape_c0, out_images_1_ub, out_images_10_ub,
                                                      out_images_11_ub, 1, 0, 0, 0)

                            self.tik_instance.vec_muls(self.images_shape_c0, out_images_0_ub, out_images_0_ub,
                                                       scalar.hl_ratio, 1, 0, 0)
                            self.tik_instance.vec_muls(self.images_shape_c0, out_images_1_ub, out_images_1_ub,
                                                       scalar.hr_ratio, 1, 0, 0)
                            self.tik_instance.vec_add(self.images_shape_c0,
                                                      self.out_images_ub[(h_idx * w_do_len + w_idx)
                                                                         * self.images_shape_c0],
                                                      out_images_0_ub, out_images_1_ub, 1, 0, 0, 0)
                    nc_gm_out_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                       * self.tiling_out_weight * self.tiling_out_height
                    nc_gm_out_start = nc_gm_out_offset + h_loop_sigment_start * w_loop_offset + w_loop_sigment_start
                    out_data_move_gm_stride = (self.tiling_out_weight - w_do_len) * self.images_shape_c0
                    self.tik_instance.data_move(self.out_gm[nc_gm_out_start], self.out_images_ub, 0,
                                                data_move_gm_nburst, data_move_gm_burst, out_data_move_gm_stride, 0, 0)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, self.weight_idx_sigment_num, h_loop_idx, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_loop_idx, h_do_len)

        with self.tik_instance.for_range(0, h_loop_num) as _loop_idx:
            _run_h_loop_default(_loop_idx, self.height_idx_sigment_num)
        with self.tik_instance.if_scope(h_tail_num != 0):
            _run_h_loop_default(h_loop_num, h_tail_num)

    def _function_default_2(self):
        """
        when tiling_key = 2
        """
        h_loop_num = self.core_height_num // self.height_idx_sigment_num
        w_loop_num = self.core_weight_num // self.weight_idx_sigment_num
        h_tail_num = self.core_height_num % self.height_idx_sigment_num
        w_tail_num = self.core_weight_num % self.weight_idx_sigment_num

        nc_total = self.core_nc_num
        input_hw_size = self.tik_instance.Scalar("int64", name="input_hw_size")
        input_hw_size.set_as(self.weight_idx_sigment_num * self.height_idx_sigment_num)
        with self.tik_instance.if_scope(tik.all(self.core_weight_num * self.core_height_num < input_hw_size,
                                                self.core_weight_num * self.core_height_num > 0)):
            input_hw_size.set_as(self.core_weight_num * self.core_height_num)

        input_hw_size = input_hw_size * self.images_shape_c0
        nc_max_segment = self.ub_max_num // input_hw_size
        nc_loop = nc_total // nc_max_segment
        nc_tail = nc_total % nc_max_segment

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_idx, h_do_len):
            h_loop_offset = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_start = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_end = h_loop_sigment_start + h_do_len
            w_loop_offset = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_start = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_end = w_loop_sigment_start + w_do_len

            scalar = self.CommomScalar(self.tik_instance)
            self.calculate_h_start(scalar, h_loop_sigment_start)
            self.calculate_h_end(scalar, h_loop_sigment_end)
            self.calculate_w_start(scalar, w_loop_sigment_start)
            self.calculate_w_end(scalar, w_loop_sigment_end)

            def _do_single_nc(do_nc_num, _nc_loop_idx):
                with self.tik_instance.for_range(0, do_nc_num) as _sigment_idx:
                    nc_gm_images_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                          * self.tiling_in_weight * self.tiling_in_height
                    nc_gm_images_start = self.tik_instance.Scalar("int32", name="nc_gm_images_start")
                    nc_gm_images_start.set_as(nc_gm_images_offset + scalar.src_h_start * self.tiling_in_weight
                                              + scalar.src_w_start)
                    with self.tik_instance.for_range(0, h_do_len) as h_idx:
                        data_move_gm_burst = (scalar.src_w_end - scalar.src_w_start + 1) * self.images_shape_c0
                        data_move_gm_nburst = scalar.src_h_end - scalar.src_h_start + 1
                        data_move_gm_stride = (self.tiling_in_weight + scalar.src_w_start - scalar.src_w_end - 1) * \
                                              self.images_shape_c0
                        self.in_images_ub = self.tik_instance.Tensor("float32", (self.weight_idx_sigment_num *
                                                                                 self.images_shape_c0 * 4,),
                                                                     name="in_images_ub", scope=tik.scope_ubuf)
                        self.out_images_ub = self.tik_instance.Tensor("float32", (self.weight_idx_sigment_num *
                                                                                  self.images_shape_c0,),
                                                                      name="out_images_ub", scope=tik.scope_ubuf)
                        self.tik_instance.data_move(self.in_images_ub, self.images_gm[nc_gm_images_start], 0,
                                                    data_move_gm_nburst, data_move_gm_burst, data_move_gm_stride, 0)
                        h_idx_per_nc = h_idx + h_loop_offset
                        self.calculate_h(scalar, h_idx_per_nc)
                        with self.tik_instance.for_range(0, w_do_len) as w_idx:
                            w_idx_per_nc = w_idx + w_loop_offset
                            self.calculate_w(scalar, w_idx_per_nc)
                            scalar.images_idx00.set_as((scalar.src_h - scalar.src_h_start)
                                                       * (scalar.src_w_end - scalar.src_w_start + 1) +
                                                       scalar.src_w - scalar.src_w_start)
                            scalar.images_idx01.set_as(scalar.images_idx00 + scalar.w_stride)
                            scalar.images_idx10.set_as(scalar.images_idx00 + scalar.h_stride *
                                                       (scalar.src_w_end - scalar.src_w_start + 1))
                            scalar.images_idx11.set_as(scalar.images_idx10 + scalar.w_stride)
                            with self.tik_instance.new_stmt_scope():
                                out_images_00_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_00_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_00_ub,
                                                           self.in_images_ub[scalar.images_idx00],
                                                           scalar.wl_ratio, 1, 0, 0)
                                out_images_01_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_01_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_01_ub,
                                                           self.in_images_ub[scalar.images_idx01],
                                                           scalar.wr_ratio, 1, 0, 0)
                                out_images_10_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_10_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_10_ub,
                                                           self.in_images_ub[scalar.images_idx10],
                                                           scalar.wl_ratio, 1, 0, 0)
                                out_images_11_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_11_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_11_ub,
                                                           self.in_images_ub[scalar.images_idx11],
                                                           scalar.wr_ratio, 1, 0, 0)
                                out_images_0_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                           name="out_images_0_ub",
                                                                           scope=tik.scope_ubuf)
                                self.tik_instance.vec_add(self.images_shape_c0, out_images_0_ub, out_images_00_ub,
                                                          out_images_01_ub, 1, 0, 0, 0)
                                out_images_1_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                           name="out_images_1_ub",
                                                                           scope=tik.scope_ubuf)
                                self.tik_instance.vec_add(self.images_shape_c0, out_images_1_ub, out_images_10_ub,
                                                          out_images_11_ub, 1, 0, 0, 0)

                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_0_ub, out_images_0_ub,
                                                           scalar.hl_ratio, 1, 0, 0)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_1_ub, out_images_1_ub,
                                                           scalar.hr_ratio, 1, 0, 0)
                                self.tik_instance.vec_add(self.images_shape_c0,
                                                          self.out_images_ub[(h_idx * w_do_len + w_idx)
                                                                             * self.images_shape_c0],
                                                          out_images_0_ub, out_images_1_ub, 1, 0, 0, 0)
                        nc_gm_out_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                           * self.tiling_out_weight * self.tiling_out_height
                        nc_gm_out_start = nc_gm_out_offset + h_loop_sigment_start * w_loop_offset + w_loop_sigment_start
                        out_data_move_gm_stride = (self.tiling_out_weight - w_do_len) * self.images_shape_c0
                        self.tik_instance.data_move(self.out_gm[nc_gm_out_start], self.out_images_ub, 0,
                                                    data_move_gm_nburst, data_move_gm_burst,
                                                    out_data_move_gm_stride, 0, 0)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, self.weight_idx_sigment_num, h_loop_idx, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_loop_idx, h_do_len)

        with self.tik_instance.for_range(0, h_loop_num) as _loop_idx:
            _run_h_loop_default(_loop_idx, self.height_idx_sigment_num)
        with self.tik_instance.if_scope(h_tail_num != 0):
            _run_h_loop_default(h_loop_num, h_tail_num)

    def _function_default_3(self):
        """
        when tiling_key = 3
        """
        h_loop_num = self.core_height_num // self.height_idx_sigment_num
        w_loop_num = self.core_weight_num // self.weight_idx_sigment_num
        h_tail_num = self.core_height_num % self.height_idx_sigment_num
        w_tail_num = self.core_weight_num % self.weight_idx_sigment_num

        nc_total = self.core_nc_num
        input_hw_size = self.tik_instance.Scalar("int64", name="input_hw_size")
        input_hw_size.set_as(self.weight_idx_sigment_num * self.height_idx_sigment_num)
        with self.tik_instance.if_scope(tik.all(self.core_weight_num * self.core_height_num < input_hw_size,
                                                self.core_weight_num * self.core_height_num > 0)):
            input_hw_size.set_as(self.core_weight_num * self.core_height_num)

        input_hw_size = input_hw_size * self.images_shape_c0
        nc_max_segment = self.ub_max_num // input_hw_size
        nc_loop = nc_total // nc_max_segment
        nc_tail = nc_total % nc_max_segment

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_idx, h_do_len):
            h_loop_offset = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_start = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_end = h_loop_sigment_start + h_do_len
            w_loop_offset = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_start = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_end = w_loop_sigment_start + w_do_len

            scalar = self.CommomScalar(self.tik_instance)
            self.calculate_h_start(scalar, h_loop_sigment_start)
            self.calculate_h_end(scalar, h_loop_sigment_end)
            self.calculate_w_start(scalar, w_loop_sigment_start)
            self.calculate_w_end(scalar, w_loop_sigment_end)

            def _do_single_nc(do_nc_num, _nc_loop_idx):
                with self.tik_instance.for_range(0, do_nc_num) as _sigment_idx:
                    nc_gm_images_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                          * self.tiling_in_weight * self.tiling_in_height
                    nc_gm_images_start = self.tik_instance.Scalar("int32", name="nc_gm_images_start")
                    nc_gm_images_start.set_as(nc_gm_images_offset + scalar.src_h_start * self.tiling_in_weight
                                              + scalar.src_w_start)
                    with self.tik_instance.for_range(0, w_do_len) as w_idx:
                        data_move_gm_burst = (scalar.src_w_end - scalar.src_w_start + 1) * self.images_shape_c0
                        data_move_gm_nburst = scalar.src_h_end - scalar.src_h_start + 1
                        data_move_gm_stride = (self.tiling_in_weight + scalar.src_w_start - scalar.src_w_end - 1) * \
                                              self.images_shape_c0
                        self.in_images_ub = self.tik_instance.Tensor("float32", (self.height_idx_sigment_num *
                                                                                 self.images_shape_c0 * 4,),
                                                                     name="in_images_ub", scope=tik.scope_ubuf)
                        self.out_images_ub = self.tik_instance.Tensor("float32", (self.height_idx_sigment_num *
                                                                                  self.images_shape_c0,),
                                                                      name="out_images_ub", scope=tik.scope_ubuf)
                        self.tik_instance.data_move(self.in_images_ub, self.images_gm[nc_gm_images_start], 0,
                                                    data_move_gm_nburst, data_move_gm_burst, data_move_gm_stride, 0)
                        w_idx_per_nc = w_idx + w_loop_offset
                        self.calculate_w(scalar, w_idx_per_nc)
                        with self.tik_instance.for_range(0, h_do_len) as h_idx:
                            h_idx_per_nc = h_idx + h_loop_offset
                            self.calculate_h(scalar, h_idx_per_nc)
                            scalar.images_idx00.set_as((scalar.src_h - scalar.src_h_start)
                                                       * (scalar.src_w_end - scalar.src_w_start + 1) +
                                                       scalar.src_w - scalar.src_w_start)
                            scalar.images_idx01.set_as(scalar.images_idx00 + scalar.w_stride)
                            scalar.images_idx10.set_as(scalar.images_idx00 + scalar.h_stride *
                                                       (scalar.src_w_end - scalar.src_w_start + 1))
                            scalar.images_idx11.set_as(scalar.images_idx10 + scalar.w_stride)
                            with self.tik_instance.new_stmt_scope():
                                out_images_00_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_00_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_00_ub,
                                                           self.in_images_ub[scalar.images_idx00],
                                                           scalar.wl_ratio, 1, 0, 0)
                                out_images_01_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_01_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_01_ub,
                                                           self.in_images_ub[scalar.images_idx01],
                                                           scalar.wr_ratio, 1, 0, 0)
                                out_images_10_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_10_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_10_ub,
                                                           self.in_images_ub[scalar.images_idx10],
                                                           scalar.wl_ratio, 1, 0, 0)
                                out_images_11_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_11_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_11_ub,
                                                           self.in_images_ub[scalar.images_idx11],
                                                           scalar.wr_ratio, 1, 0, 0)
                                out_images_0_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                           name="out_images_0_ub",
                                                                           scope=tik.scope_ubuf)
                                self.tik_instance.vec_add(self.images_shape_c0, out_images_0_ub, out_images_00_ub,
                                                          out_images_01_ub, 1, 0, 0, 0)
                                out_images_1_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                           name="out_images_1_ub",
                                                                           scope=tik.scope_ubuf)
                                self.tik_instance.vec_add(self.images_shape_c0, out_images_1_ub, out_images_10_ub,
                                                          out_images_11_ub, 1, 0, 0, 0)

                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_0_ub, out_images_0_ub,
                                                           scalar.hl_ratio, 1, 0, 0)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_1_ub, out_images_1_ub,
                                                           scalar.hr_ratio, 1, 0, 0)
                                self.tik_instance.vec_add(self.images_shape_c0,
                                                          self.out_images_ub[(h_idx * w_do_len + w_idx)
                                                                             * self.images_shape_c0],
                                                          out_images_0_ub, out_images_1_ub, 1, 0, 0, 0)
                        nc_gm_out_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                           * self.tiling_out_weight * self.tiling_out_height
                        nc_gm_out_start = nc_gm_out_offset + h_loop_sigment_start * w_loop_offset + w_loop_sigment_start
                        out_data_move_gm_stride = (self.tiling_out_weight - w_do_len) * self.images_shape_c0
                        self.tik_instance.data_move(self.out_gm[nc_gm_out_start], self.out_images_ub, 0,
                                                    data_move_gm_nburst, data_move_gm_burst,
                                                    out_data_move_gm_stride, 0, 0)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, self.weight_idx_sigment_num, h_loop_idx, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_loop_idx, h_do_len)

        with self.tik_instance.for_range(0, h_loop_num) as _loop_idx:
            _run_h_loop_default(_loop_idx, self.height_idx_sigment_num)
        with self.tik_instance.if_scope(h_tail_num != 0):
            _run_h_loop_default(h_loop_num, h_tail_num)

    def _function_default_4(self):
        """
        when tiling_key = 4
        """
        h_loop_num = self.core_height_num // self.height_idx_sigment_num
        w_loop_num = self.core_weight_num // self.weight_idx_sigment_num
        h_tail_num = self.core_height_num % self.height_idx_sigment_num
        w_tail_num = self.core_weight_num % self.weight_idx_sigment_num

        nc_total = self.core_nc_num
        input_hw_size = self.tik_instance.Scalar("int64", name="input_hw_size")
        input_hw_size.set_as(self.weight_idx_sigment_num * self.height_idx_sigment_num)
        with self.tik_instance.if_scope(tik.all(self.core_weight_num * self.core_height_num < input_hw_size,
                                                self.core_weight_num * self.core_height_num > 0)):
            input_hw_size.set_as(self.core_weight_num * self.core_height_num)

        input_hw_size = input_hw_size * self.images_shape_c0
        nc_max_segment = self.ub_max_num // input_hw_size
        nc_loop = nc_total // nc_max_segment
        nc_tail = nc_total % nc_max_segment

        def _run_w_loop_default(w_loop_idx, w_do_len, h_loop_idx, h_do_len):
            h_loop_offset = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_start = h_loop_idx * self.height_idx_sigment_num + self.core_height_start
            h_loop_sigment_end = h_loop_sigment_start + h_do_len
            w_loop_offset = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_start = w_loop_idx * self.weight_idx_sigment_num + self.core_weight_start
            w_loop_sigment_end = w_loop_sigment_start + w_do_len

            scalar = self.CommomScalar(self.tik_instance)
            self.calculate_h_start(scalar, h_loop_sigment_start)
            self.calculate_h_end(scalar, h_loop_sigment_end)
            self.calculate_w_start(scalar, w_loop_sigment_start)
            self.calculate_w_end(scalar, w_loop_sigment_end)

            def _do_single_nc(do_nc_num, _nc_loop_idx):
                with self.tik_instance.for_range(0, do_nc_num) as _sigment_idx:
                    nc_gm_images_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                          * self.tiling_in_weight * self.tiling_in_height
                    nc_gm_images_start = self.tik_instance.Scalar("int32", name="nc_gm_images_start")
                    nc_gm_images_start.set_as(nc_gm_images_offset + scalar.src_h_start * self.tiling_in_weight
                                              + scalar.src_w_start)
                    with self.tik_instance.for_range(0, w_do_len) as w_idx:
                        w_idx_per_nc = w_idx + w_loop_offset
                        self.calculate_w(scalar, w_idx_per_nc)
                        with self.tik_instance.for_range(0, h_do_len) as h_idx:
                            data_move_gm_burst = (scalar.src_w_end - scalar.src_w_start + 1) * self.images_shape_c0
                            data_move_gm_nburst = scalar.src_h_end - scalar.src_h_start + 1
                            data_move_gm_stride = (self.tiling_in_weight + scalar.src_w_start - scalar.src_w_end - 1) \
                                                  * self.images_shape_c0
                            self.in_images_ub = self.tik_instance.Tensor("float32", (self.height_idx_sigment_num *
                                                                                     self.images_shape_c0 * 4,),
                                                                         name="in_images_ub", scope=tik.scope_ubuf)
                            self.out_images_ub = self.tik_instance.Tensor("float32", (self.height_idx_sigment_num *
                                                                                      self.images_shape_c0,),
                                                                          name="out_images_ub", scope=tik.scope_ubuf)
                            self.tik_instance.data_move(self.in_images_ub, self.images_gm[nc_gm_images_start], 0,
                                                        data_move_gm_nburst, data_move_gm_burst,
                                                        data_move_gm_stride, 0, 0)

                            h_idx_per_nc = h_idx + h_loop_offset
                            self.calculate_h(scalar, h_idx_per_nc)
                            scalar.images_idx00.set_as((scalar.src_h - scalar.src_h_start)
                                                       * (scalar.src_w_end - scalar.src_w_start) +
                                                       scalar.src_w - scalar.src_w_start)
                            scalar.images_idx01.set_as(scalar.images_idx00 + scalar.w_stride)
                            scalar.images_idx10.set_as(scalar.images_idx00 + scalar.h_stride *
                                                       (scalar.src_w_end - scalar.src_w_start + 1))
                            scalar.images_idx11.set_as(scalar.images_idx10 + scalar.w_stride)
                            with self.tik_instance.new_stmt_scope():
                                out_images_00_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_00_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_00_ub,
                                                           self.in_images_ub[scalar.images_idx00],
                                                           scalar.wl_ratio, 1, 0, 0)
                                out_images_01_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_01_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_01_ub,
                                                           self.in_images_ub[scalar.images_idx01],
                                                           scalar.wr_ratio, 1, 0, 0)
                                out_images_10_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_10_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_10_ub,
                                                           self.in_images_ub[scalar.images_idx10],
                                                           scalar.wl_ratio, 1, 0, 0)
                                out_images_11_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                            name="out_images_11_ub",
                                                                            scope=tik.scope_ubuf)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_11_ub,
                                                           self.in_images_ub[scalar.images_idx11],
                                                           scalar.wr_ratio, 1, 0, 0)
                                out_images_0_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                           name="out_images_0_ub",
                                                                           scope=tik.scope_ubuf)
                                self.tik_instance.vec_add(self.images_shape_c0, out_images_0_ub, out_images_00_ub,
                                                          out_images_01_ub, 1, 0, 0, 0)
                                out_images_1_ub = self.tik_instance.Tensor("float32", (self.images_shape_c0,),
                                                                           name="out_images_1_ub",
                                                                           scope=tik.scope_ubuf)
                                self.tik_instance.vec_add(self.images_shape_c0, out_images_1_ub, out_images_10_ub,
                                                          out_images_11_ub, 1, 0, 0, 0)

                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_0_ub, out_images_0_ub,
                                                           scalar.hl_ratio, 1, 0, 0)
                                self.tik_instance.vec_muls(self.images_shape_c0, out_images_1_ub, out_images_1_ub,
                                                           scalar.hr_ratio, 1, 0, 0)
                                self.tik_instance.vec_add(self.images_shape_c0,
                                                          self.out_images_ub[(h_idx * w_do_len + w_idx)
                                                                             * self.images_shape_c0],
                                                          out_images_0_ub, out_images_1_ub, 1, 0, 0, 0)
                            nc_gm_out_offset = (_nc_loop_idx * nc_max_segment + self.core_nc_start + _sigment_idx) \
                                               * self.tiling_out_weight * self.tiling_out_height
                            nc_gm_out_start = nc_gm_out_offset + h_loop_sigment_start * w_loop_offset\
                                              + w_loop_sigment_start
                            out_data_move_gm_stride = (self.tiling_out_weight - w_do_len) * self.images_shape_c0
                            self.tik_instance.data_move(self.out_gm[nc_gm_out_start], self.out_images_ub, 0,
                                                        data_move_gm_nburst, data_move_gm_burst,
                                                        out_data_move_gm_stride, 0, 0)

            with self.tik_instance.for_range(0, nc_loop) as nc_loop_idx:
                _do_single_nc(nc_max_segment, nc_loop_idx)
            with self.tik_instance.if_scope(nc_tail != 0):
                _do_single_nc(nc_tail, nc_loop)

        def _run_h_loop_default(h_loop_idx, h_do_len):
            with self.tik_instance.for_range(0, w_loop_num) as w_loop_idx:
                _run_w_loop_default(w_loop_idx, self.weight_idx_sigment_num, h_loop_idx, h_do_len)
            with self.tik_instance.if_scope(w_tail_num != 0):
                _run_w_loop_default(w_loop_num, w_tail_num, h_loop_idx, h_do_len)

        with self.tik_instance.for_range(0, h_loop_num) as _loop_idx:
            _run_h_loop_default(_loop_idx, self.height_idx_sigment_num)
        with self.tik_instance.if_scope(h_tail_num != 0):
            _run_h_loop_default(h_loop_num, h_tail_num)

    def _do_resize_base_tiling_key(self):

        with self.tik_instance.if_scope(self.tiling_key == 1):
            with self.tik_instance.new_stmt_scope():
                self._function_default_1()
        with self.tik_instance.if_scope(self.tiling_key == 2):
            with self.tik_instance.new_stmt_scope():
                self._function_default_2()
        with self.tik_instance.if_scope(self.tiling_key == 3):
            with self.tik_instance.new_stmt_scope():
                self._function_default_3()
        with self.tik_instance.if_scope(self.tiling_key == 4):
            with self.tik_instance.new_stmt_scope():
                self._function_default_4()

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

    def resize_bilinear_v2_operator(self):
        """
        resize_bilinear_v2_operator
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


@register_operator("ResizeBilinearV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def resize_bilinear_v2(images, size, y, align_corners=False, half_pixel_centers=False,
                       kernel_name="resize_bilinear_v2"):
    """Resize `images` to `size` using bilinear interpolation.

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
        cce kernel name, default value is `resize_bilinear_v2`

    Returns
    -------
    tik_instance
    """
    check_vgatherb_supported = tbe_platform.api_check_support("tik.vgatherb")
    check_vbi_supported = tbe_platform.api_check_support("tik.vbi", "float32")
    if check_vgatherb_supported and check_vbi_supported:
        return resize_bilinear_v2_with_gatherb(images, size, y, align_corners, half_pixel_centers,
                                               kernel_name)

    obj = ResizeBilinearV2(images, size, y, align_corners, half_pixel_centers,
                           kernel_name)

    return obj.resize_bilinear_v2_operator()
