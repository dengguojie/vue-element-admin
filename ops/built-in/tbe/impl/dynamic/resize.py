from impl.dynamic.resize_bilinear_v2 import ResizeBilinearV2
from impl.dynamic.resize_bilinear_v2 import tbe_context
from impl.dynamic.resize_bilinear_v2 import Constant as Constant_linear
from impl.dynamic.resize_nearest_neighbor_v2 import ResizeNearestNeighbor
from impl.dynamic.resize_nearest_neighbor_v2 import Constant
from impl.dynamic.resize_nearest_neighbor_v2 import partial
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl import common_util
from impl.util.util_tik_comm_func import OpBase
from impl.util.platform_adapter import tik


class ResizeModeNearestNeighbor(ResizeNearestNeighbor):

    def __init__(self, images, roi, scales, size, y, align_corners, half_pixel_centers, kernel_name):
        OpBase.__init__(self)
        self.images_dtype = images.get("dtype").lower()
        self.size_dtype = "int32"
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

        self.kernel_name = kernel_name
        self.ub_size_bytes = self.ub_size_bytes - Constant.RESERVED_UB_SIZE

        self.shape_c0 = 16
        self.block_num = 16 if self.images_dtype in ("float16",) else 8
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.block_num
        self.c0_blocks = self.shape_c0 // self.block_num

        self.l1_size_bytes = 0 \
            if not tbe_platform.intrinsic_check_support("Intrinsic_data_move_l12ub") \
            else tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
        self.l1_exists = True if self.l1_size_bytes > 0 else False
        self.l1_max_num = self.l1_size_bytes // 2 // (2 if self.images_dtype in ("float16",) else 4)

        # init gm addr
        tiling_dict = {"dtype": "int64", "shape": (Constant.TILING_ARG_NUM,)}
        input_list = []
        for input_dict in (images, roi, scales, size):
            if input_dict is not None:
                input_list.append(input_dict)
        self.op_init_gm(input_list, [y], tiling_info=tiling_dict, is_fused_1d=True)
        self.images_gm = self.input_gm_list[0]
        self.out_gm = self.output_gm_list[0]

        self.compute_dtype = "float32" if tbe_platform.api_check_support("tik.vadds", "float32") else "float16"
        self.coordinate_dtype = "int32"
        if self.compute_dtype == "float16" and tbe_platform.api_check_support("tik.vconv", "s162f16"):
            self.coordinate_dtype = "int16"
        self.tiling_dtype = self.coordinate_dtype if self.compute_dtype == "float16" else "int64"
        self.compute_block_num = 32 // common_util.get_data_size(self.compute_dtype)
        self.block_strides = 64 // self.compute_block_num

        self.is_support_vdiv = tbe_platform.api_check_support("tik.vdiv", self.compute_dtype)
        self.is_support_vgatherb = tbe_platform.api_check_support("tik.vgatherb")

        # will be set as `(IN_X / OUT_X)` or `((IN_X - 1)/(OUT_X - 1))` when needed
        self.h_rec_scale_fpx = self.tik_instance.Scalar(self.compute_dtype, name="h_rec_scale_fpx")
        self.w_rec_scale_fpx = self.tik_instance.Scalar(self.compute_dtype, name="w_rec_scale_fpx")
        # will be set as `(OUT_X // IN_X)` w/o considering align_corners
        self.h_scale = self.tik_instance.Scalar("int64", name="h_scale")
        self.w_scale = self.tik_instance.Scalar("int64", name="w_scale")

        # init tiling data
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_key")
        self.image_nc1 = self.tik_instance.Scalar("int64", name="image_nc1")
        self.input_height = self.tik_instance.Scalar("int64", name="input_height")
        self.input_width = self.tik_instance.Scalar("int64", name="input_width")
        self.output_height = self.tik_instance.Scalar("int64", name="output_height")
        self.output_width = self.tik_instance.Scalar("int64", name="output_width")
        self.tiling_nc1_cut_num = self.tik_instance.Scalar("int64", name="tiling_nc1_cut_num")
        self.tiling_height_cut_num = self.tik_instance.Scalar("int64", name="tiling_height_cut_num")
        self.tiling_width_cut_num = self.tik_instance.Scalar("int64", name="tiling_width_cut_num")

        # init scalars for each core
        self._resize_mode_init_scale()

    def resize_nearest_neighbor_v2_operator(self):
        """
        resize_nearest_neighbor_v2_operator
        """
        # register compute base on tiling_key
        register_func = partial(self.regist_compute, tiling_func=self._functions)
        for k in self._processors:
            register_func(k, key=k)

        # run all registered compute base tiling key
        self.op_run_compute()

        # Build CCE
        max_w_len = (self.ub_max_num // self.shape_c0) if self.l1_exists else ((self.ub_max_num // self.shape_c0) - 1)
        tbe_context.get_context().add_compile_info("vars", {"ub_size": self.ub_size_bytes,
                                                            "core_num": self.core_nums,
                                                            "max_w_len": max_w_len,  # max_w_len actually is w_scale
                                                            "align_corners": int(self.align_corners),
                                                            "half_pixel_centers": int(self.half_pixel_centers),
                                                            "mode_name": 20})

        self.op_build_cce()
        return self.tik_instance

    def _resize_mode_init_scale(self):
        # nc1 start addr offset per core
        self.core_nc_start = self.tik_instance.Scalar("int64", name="core_nc_start")
        # h start addr offset per core
        self.core_height_start = self.tik_instance.Scalar("int64", name="core_height_start")
        # w start addr offset per core
        self.core_width_start = self.tik_instance.Scalar("int64", name="core_width_start")
        # nc1 process len per core
        self.nc_per_core = self.tik_instance.Scalar("int64", name="nc_per_core")
        # h process len per core
        self.h_per_core = self.tik_instance.Scalar("int64", name="h_per_core")
        # w process len per core
        self.w_per_core = self.tik_instance.Scalar("int64", name="w_per_core")
        # cut-ed height, input or output height
        self.cut_height_num = self.tik_instance.Scalar("int64", name="cut_height_num")
        # cut-ed width, input or output width
        self.cut_width_num = self.tik_instance.Scalar("int64", name="cut_width_num")


class ResizeModeBilinear(ResizeBilinearV2):

    def __init__(self, images, roi, scales, size, y, align_corners, half_pixel_centers, kernel_name):
        OpBase.__init__(self)
        self.is_bilinear = True
        self.images_dtype = images.get("dtype").lower()
        self.output_dtype = y.get("dtype").lower()
        self.inner_dtype = "float32"
        self.size_dtype = "int32"
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers

        self.kernel_name = kernel_name
        self.ub_size_bytes = self.ub_size_bytes - Constant_linear.RESERVED_UB_SIZE

        self.block_num = 16 if self.inner_dtype in ("float16",) else 8
        self.vector_num = self.block_num * 8
        self.input_block_num = 16 if self.images_dtype in ("float16",) else 8
        self.output_block_num = 16 if self.output_dtype in ("float16",) else 8
        self.block_num = 16 if self.inner_dtype in ("float16",) else 8
        self.ub_max_num = self.ub_size_bytes // 32 // 2 * self.block_num
        self.inner_bytes_size = tbe_platform.get_bit_len(self.inner_dtype) // 8
        self.input_bytes_size = tbe_platform.get_bit_len(self.images_dtype) // 8

        self.images_shape_c0 = 16
        self.height_idx_sigment_num = 512
        self.width_idx_sigment_num = 512
        input_list = []
        for input_dict in (images, roi, scales, size):
            if input_dict is not None:
                input_list.append(input_dict)
        # init gm addr
        tiling_dict = {"dtype": "int64", "shape": (Constant_linear.TILING_ARG_NUM,)}
        self.op_init_gm(input_list, [y], tiling_info=tiling_dict, is_fused_1d=True)
        self.images_gm = self.input_gm_list[0]
        self.out_gm = self.output_gm_list[0]

        # gen assist ub for [0, 1, 2, ...., 255]
        assist_value = list(range(Constant_linear.ASSIST_NUM))
        self.assist_gm = self.tik_instance.Tensor("float32", (Constant_linear.ASSIST_NUM,),
                                                  name="assist_gm",
                                                  scope=tik.scope_gm,
                                                  init_value=assist_value)

        self.stride_threshold = Constant_linear.MAX_UINT16 if self.images_dtype in ("float16",) \
            else Constant_linear.MAX_UINT16 // 2
        if self.output_dtype in ("float16",):
            self.dst_stride_threshold = Constant_linear.MAX_UINT16
        else:
            self.dst_stride_threshold = Constant_linear.MAX_UINT16 // 2
        self.is_suport_vdiv = tbe_platform.api_check_support("tik.vdiv", "float32")
        # init tiling data
        self._resize_mode_init_scale()

        # init ub
        self._resize_mode_init_ub()

    def resize_bilinear_v2_operator(self):
        # regist compute base on tiling_key
        self.regist_compute(100110, self._function_reisze_with_nc_process)
        self.regist_compute(999999, self._tiling_compute_with_no_bilinear)
        self.regist_compute(100000, self._tiling_compute_default)
        # run all regist compute base tiling key
        self.op_run_compute()
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": self.ub_size_bytes,
                "core_num": self.core_nums,
                "max_w_len": self.ub_max_num // self.images_shape_c0,
                "align_corners": int(self.align_corners),
                "half_pixel_centers": int(self.half_pixel_centers),
                "mode_name": 21
            })
        # Build CCE
        self.op_build_cce()
        return self.tik_instance

    def _resize_mode_init_scale(self):
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

        # init stride scalar flag
        self.scalar_is_src_stride = self.tik_instance.Scalar("int32", name="scalar_is_src_stride", init_value=1)
        self.scalar_is_dst_stride = self.tik_instance.Scalar("int32", name="scalar_is_dst_stride", init_value=1)

    def _resize_mode_init_ub(self):
        self.height_idx_ub = None
        self.width_idx_ub = None
        self.idx_ub_fp32 = None
        self.idx_cb_fp32 = None
        self.image_out_ub = None
        self.image_in_cb_ping = None
        self.image_out_ub = None
        self.image_in_cb_ping = None


@register_operator("Resize")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
# 'pylint: disable=unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
def resize(x, roi, scales, sizes, y,
           coordinate_transformation_mode="half_pixel", cubic_coeff_a=-0.75,
           exclude_outside=0, extrapolation_value=0.0,
           mode="nearest", nearest_mode="round_prefer_floor",
           kernel_name="Resize"):
    input_format = x.get("format").upper()
    if input_format == "NC1HWC0":
        attr_name = "coordinate_transformation_mode"
        if coordinate_transformation_mode == "pytorch_half_pixel":
            half_pixel_centers = True
            align_corners = False
        elif coordinate_transformation_mode == "align_corners":
            half_pixel_centers = False
            align_corners = True
        elif coordinate_transformation_mode == "asymmetric":
            half_pixel_centers = False
            align_corners = False
        else:
            raise RuntimeError("Resize not support attr {} {}".format(attr_name, coordinate_transformation_mode))
        if mode == "nearest":
            obj = ResizeModeNearestNeighbor(x, roi, scales, sizes, y, align_corners, half_pixel_centers, kernel_name)
            return obj.resize_nearest_neighbor_v2_operator()
        elif mode == "linear":
            obj = ResizeModeBilinear(x, roi, scales, sizes, y, align_corners, half_pixel_centers, kernel_name)

            return obj.resize_bilinear_v2_operator()
        else:
            raise RuntimeError("Resize not support mode {}".format(mode))
    else:
        raise RuntimeError("Resize not support format {}".format(input_format))
