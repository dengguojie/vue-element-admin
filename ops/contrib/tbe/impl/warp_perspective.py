from te import tik
import math


class WarpPerspective:

    def __init__(self,
                 src_size,
                 dst_size,
                 interpolation,
                 constant_value,
                 kernel_name_value):
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))

        self.src_height = src_size[0]
        self.src_width = src_size[1]
        self.src_height_1 = src_size[0] - 1
        self.src_width_1 = src_size[1] - 1

        if self.src_height > 2048 or self.src_width > 2048:
            raise RuntimeError("Only support src image heigh and width both <= 2048")

        self.dst_height = dst_size[0]
        self.dst_width = dst_size[1]

        if self.dst_height > 2048 or self.dst_width > 2048:
            raise RuntimeError("Only support dst image height and width both <= 2048")

        if self.src_height < 0 or self.src_width < 0 or self.dst_height < 0 or self.dst_width < 0:
            raise RuntimeError("src_size or dst_size is illegal")

        self.interpolation = interpolation
        self.constant_value = constant_value
        self.kernel_name = kernel_name_value

        self.align_dst_height = self._ceil_div_offline(self.dst_height, 64) * 64
        self.align_dst_width = self._ceil_div_offline(self.dst_width, 64) * 64

        self.aicore_use = 2
        self.y_loop_each_aicore = self.dst_height // self.aicore_use
        self.y_loop_last_aicore = self.dst_height - self.y_loop_each_aicore * (self.aicore_use - 1)

        self.src_image = self.tik_instance.Tensor("float16",
                                                  (1, 1, self.src_height, self.src_width, 16), name="src_image",
                                                  scope=tik.scope_gm)
        self.dst_image = self.tik_instance.Tensor("float16", (1, 1, self.dst_height, self.dst_width, 16),
                                                  name="dst_image", scope=tik.scope_gm)
        self.transform_matrix = self.tik_instance.Tensor("float16", (1, 1, 1, 1, 16), name="transform_matrix",
                                                         scope=tik.scope_gm)
        self.loop_width = self.dst_width

    def compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            transform_matrix_ub_fp16 = self.tik_instance.Tensor("float16", (16,), name="transform_matrix_ub_fp16",
                                                                scope=tik.scope_ubuf)
            transform_matrix_ub_fp32 = self.tik_instance.Tensor("float32", (16,), name="transform_matrix_ub_fp32",
                                                                scope=tik.scope_ubuf)
            self.tik_instance.data_move(transform_matrix_ub_fp16, self.transform_matrix, 0, 1, 1, 0, 0)
            self.tik_instance.vconv(16, "", transform_matrix_ub_fp32, transform_matrix_ub_fp16, 1, 1, 1, 8, 4)

            # M0*col  M3*col  M6*col
            tmp_m_mul_x_fp32 = self.tik_instance.Tensor("float32", (3, self.align_dst_width), name="tmp_m_mul_x_fp32",
                                                        scope=tik.scope_ubuf)
            # M1*raw  M4*raw  M7*raw
            tmp_m_mul_y_fp32 = self.tik_instance.Tensor("float32", (3, self.align_dst_height), name="tmp_m_mul_y_fp32",
                                                        scope=tik.scope_ubuf)

            self._get_mul_value_fp32(tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, transform_matrix_ub_fp32)

            matrix2_fp32 = self.tik_instance.Scalar("float32")
            matrix2_fp32.set_as(transform_matrix_ub_fp32[2])
            matrix5_fp32 = self.tik_instance.Scalar("float32")
            matrix5_fp32.set_as(transform_matrix_ub_fp32[5])
            matrix8_fp32 = self.tik_instance.Scalar("float32")
            matrix8_fp32.set_as(transform_matrix_ub_fp32[8])

            offset = index * self.y_loop_each_aicore
            last_index = self.aicore_use - 1
            with self.tik_instance.if_scope(index != last_index):
                if self.interpolation == "INTER_LINEAR":
                    self._compute_each_core_linear(offset, self.y_loop_each_aicore, tmp_m_mul_x_fp32,
                                                   tmp_m_mul_y_fp32, matrix2_fp32, matrix5_fp32, matrix8_fp32)
                else:
                    self._compute_each_core_near(offset, self.y_loop_each_aicore, tmp_m_mul_x_fp32,
                                                 tmp_m_mul_y_fp32, matrix2_fp32, matrix5_fp32, matrix8_fp32)
            with self.tik_instance.else_scope():
                if self.interpolation == "INTER_LINEAR":
                    self._compute_each_core_linear(offset, self.y_loop_last_aicore, tmp_m_mul_x_fp32,
                                                   tmp_m_mul_y_fp32, matrix2_fp32, matrix5_fp32, matrix8_fp32)
                else:
                    self._compute_each_core_near(offset, self.y_loop_last_aicore, tmp_m_mul_x_fp32,
                                                 tmp_m_mul_y_fp32, matrix2_fp32, matrix5_fp32, matrix8_fp32)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.src_image, self.transform_matrix],
                                   outputs=[self.dst_image])

    def _compute_each_core_near(self, offset, loop_y, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, matrix2_fp32,
                                matrix5_fp32, matrix8_fp32):
        position_x_scalar = self.tik_instance.Scalar("int32")
        position_y_scalar = self.tik_instance.Scalar("int32")
        process_width = self.align_dst_width
        repeat_time = process_width // 64

        common_tensor = {"x0_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                             name="x0_fp32", scope=tik.scope_ubuf),
                         "y0_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                             name="y0_fp32", scope=tik.scope_ubuf),
                         "w0_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                             name="w0_fp32", scope=tik.scope_ubuf),
                         "fx_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                             name="fx_fp32", scope=tik.scope_ubuf),
                         "fy_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                             name="fy_fp32", scope=tik.scope_ubuf),
                         "x_int32": self.tik_instance.Tensor("int32", (process_width,),
                                                             name="x_int32", scope=tik.scope_ubuf),
                         "y_int32": self.tik_instance.Tensor("int32", (process_width,),
                                                             name="y_int32", scope=tik.scope_ubuf),
                         "fx_round_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fx_round_fp32",
                                                                   scope=tik.scope_ubuf),
                         "fy_round_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fy_round_fp32",
                                                                   scope=tik.scope_ubuf),
                         "dst_line_value": self.tik_instance.Tensor("float16", (process_width, 16),
                                                                    name="dst_line_value", scope=tik.scope_ubuf)}

        with self.tik_instance.for_range(0, loop_y) as raw_temp:
            raw = raw_temp + offset
            self._get_coordinate(raw, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, repeat_time, matrix2_fp32, matrix5_fp32,
                                 matrix8_fp32, process_width, 0, common_tensor)
            # X equal to round(fx)
            # Y equal to round(fy)
            self.tik_instance.vadds(64, common_tensor["fx_round_fp32"], common_tensor["fx_fp32"], 0.5,
                                    repeat_time, 1, 1, 8, 8)
            self.tik_instance.vadds(64, common_tensor["fy_round_fp32"], common_tensor["fy_fp32"], 0.5,
                                    repeat_time, 1, 1, 8, 8)
            self._floor_f32toi32(common_tensor["x_int32"], common_tensor["fx_round_fp32"], process_width)
            self._floor_f32toi32(common_tensor["y_int32"], common_tensor["fy_round_fp32"], process_width)

            self._set_constant(common_tensor["dst_line_value"], process_width)
            with self.tik_instance.for_range(0, self.loop_width) as col:
                position_x_scalar.set_as(common_tensor["x_int32"][col])
                position_y_scalar.set_as(common_tensor["y_int32"][col])
                with self.tik_instance.if_scope(tik.all(position_y_scalar >= 0,
                                                        position_y_scalar < self.src_height, position_x_scalar >= 0,
                                                        position_x_scalar < self.src_width)):
                    self.tik_instance.data_move(common_tensor["dst_line_value"][col, 0],
                                                self.src_image[0, 0, position_y_scalar, position_x_scalar, 0], 0, 1, 1,
                                                0, 0)
            self.tik_instance.data_move(self.dst_image[0, 0, raw, 0, 0], common_tensor["dst_line_value"],
                                        0, 1, self.loop_width, 0, 0)

    def _set_constant(self, tensor, length):
        repeat_time_first_step = int(math.floor((length * 16 // 128) / 255))
        repeat_time_second_step = int(math.floor((length * 16 - 128 * 255 * repeat_time_first_step) / 128))
        repeat_time_third_step = length * 16 - 128 * 255 * repeat_time_first_step - repeat_time_second_step * 128

        if repeat_time_first_step != 0:
            self.tik_instance.vector_dup(128, tensor, self.constant_value, 255, 1, 8)

        if repeat_time_second_step != 0:
            self.tik_instance.vector_dup(128, tensor[repeat_time_first_step * 128 * 255],
                                         self.constant_value, repeat_time_second_step, 1, 8)

        if repeat_time_third_step != 0:
            self.tik_instance.vector_dup(128, tensor[repeat_time_first_step * 128 * 255 +
                                         repeat_time_second_step * 128], self.constant_value, 1, 1, 8)

    def _gen_linear_local_tensor(self, process_width):
        local_tensor = {"x0_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                            name="x0_fp32", scope=tik.scope_ubuf),
                        "y0_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                            name="y0_fp32", scope=tik.scope_ubuf),
                        "w0_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                            name="w0_fp32", scope=tik.scope_ubuf),
                        "fx_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                            name="fx_fp32", scope=tik.scope_ubuf),
                        "fy_fp32": self.tik_instance.Tensor("float32", (process_width,),
                                                            name="fy_fp32", scope=tik.scope_ubuf),
                        "x_int32": self.tik_instance.Tensor("int32", (process_width,),
                                                            name="x_int32", scope=tik.scope_ubuf),
                        "y_int32": self.tik_instance.Tensor("int32", (process_width,),
                                                            name="y_int32", scope=tik.scope_ubuf),
                        "fx_floor_fp16": self.tik_instance.Tensor("float16", (process_width,),
                                                                  name="fx_floor_fp16", scope=tik.scope_ubuf),
                        "fy_floor_fp16": self.tik_instance.Tensor("float16", (process_width,), name="fy_floor_fp16",
                                                                  scope=tik.scope_ubuf),
                        "fx_floor_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fx_floor_fp32",
                                                                  scope=tik.scope_ubuf),
                        "fy_floor_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fy_floor_fp32",
                                                                  scope=tik.scope_ubuf),
                        "fx_ceil_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fx_ceil_fp32",
                                                                 scope=tik.scope_ubuf),
                        "fy_ceil_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fy_ceil_fp32",
                                                                 scope=tik.scope_ubuf),
                        "ceilx_fx_fp32": self.tik_instance.Tensor("float32", (process_width,), name="ceilx_fx_fp32",
                                                                  scope=tik.scope_ubuf),
                        "ceily_fy_fp32": self.tik_instance.Tensor("float32", (process_width,), name="ceily_fy_fp32",
                                                                  scope=tik.scope_ubuf),
                        "fx_floorx_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fx_floorx_fp32",
                                                                   scope=tik.scope_ubuf),
                        "fy_floory_fp32": self.tik_instance.Tensor("float32", (process_width,), name="fy_floory_fp32",
                                                                   scope=tik.scope_ubuf),
                        "weight_fp32": self.tik_instance.Tensor("float32", (4, process_width,), name="weight_fp32",
                                                                scope=tik.scope_ubuf),
                        "dst_line_value": self.tik_instance.Tensor("float16", (process_width, 16),
                                                                   name="dst_line_value", scope=tik.scope_ubuf)}
        return local_tensor

    def _update_linear_local_tensor(self, local_tensor):
        local_tensor["src_pixel_00"] = self.tik_instance.Tensor("float16", (16,),
                                                                name="src_pixel_00", scope=tik.scope_ubuf)
        local_tensor["src_pixel_01"] = self.tik_instance.Tensor("float16", (16,),
                                                                name="src_pixel_01", scope=tik.scope_ubuf)
        local_tensor["src_pixel_10"] = self.tik_instance.Tensor("float16", (16,),
                                                                name="src_pixel_10", scope=tik.scope_ubuf)
        local_tensor["src_pixel_11"] = self.tik_instance.Tensor("float16", (16,),
                                                                name="src_pixel_11", scope=tik.scope_ubuf)
        local_tensor["src_pixel_00_fp32"] = self.tik_instance.Tensor("float32", (16,),
                                                                     name="src_pixel_00_fp32", scope=tik.scope_ubuf)
        local_tensor["src_pixel_01_fp32"] = self.tik_instance.Tensor("float32", (16,),
                                                                     name="src_pixel_01_fp32", scope=tik.scope_ubuf)
        local_tensor["src_pixel_10_fp32"] = self.tik_instance.Tensor("float32", (16,),
                                                                     name="src_pixel_10_fp32", scope=tik.scope_ubuf)
        local_tensor["src_pixel_11_fp32"] = self.tik_instance.Tensor("float32", (16,),
                                                                     name="src_pixel_11_fp32", scope=tik.scope_ubuf)

    def _compute_each_core_linear(self, offset, loop_y, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, matrix2_fp32,
                                  matrix5_fp32, matrix8_fp32):
        if self.align_dst_width > 1024:
            process_width = 1024
            left_process_width = self.align_dst_width - 1024
        else:
            process_width = self.align_dst_width
            left_process_width = 0

        if left_process_width > 0:
            self.loop_width = process_width

        repeat_time = process_width // 64
        local_tensor = self._gen_linear_local_tensor(process_width)
        self._update_linear_local_tensor(local_tensor)

        with self.tik_instance.for_range(0, loop_y) as raw_temp:
            raw = raw_temp + offset
            self._get_coordinate(raw, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, repeat_time, matrix2_fp32, matrix5_fp32,
                                 matrix8_fp32, process_width, 0, local_tensor)
            self._get_linear_round_coordinate(process_width, repeat_time, local_tensor)
            self.tik_instance.vector_dup(128, local_tensor["dst_line_value"],
                                         self.constant_value, process_width // 8, 1, 8)
            self._compute_for_linear_cols(local_tensor, self.loop_width)
            self.tik_instance.data_move(self.dst_image[0, 0, raw, 0, 0],
                                        local_tensor["dst_line_value"], 0, 1, self.loop_width, 0, 0)

            # second step
            if left_process_width > 0:
                real_width = self.dst_width - 1024
                repeat_time_left = left_process_width // 64
                self._get_coordinate(raw, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, repeat_time_left, matrix2_fp32,
                                     matrix5_fp32, matrix8_fp32, process_width, 1024, local_tensor)
                self._get_linear_round_coordinate(process_width, repeat_time_left, local_tensor)
                self.tik_instance.vector_dup(128, local_tensor["dst_line_value"],
                                             self.constant_value, process_width // 8, 1, 8)
                self._compute_for_linear_cols(local_tensor, real_width)
                self.tik_instance.data_move(self.dst_image[0, 0, raw, 1024, 0], local_tensor["dst_line_value"],
                                            0, 1, real_width, 0, 0)

    def _compute_for_linear_cols(self, local_tensor, real_width):
        position_x_scalar = self.tik_instance.Scalar("int32")
        position_y_scalar = self.tik_instance.Scalar("int32")
        tmp_tensor = self.tik_instance.Tensor("float32", (16,), name="tmp_tensor", scope=tik.scope_ubuf)
        dst_value_fp32 = self.tik_instance.Tensor("float32", (16,), name="dst_value", scope=tik.scope_ubuf)
        one_scalar = self.tik_instance.Scalar("int32", init_value=1)
        tmp_fp32_scalar = self.tik_instance.Scalar("float32")
        with self.tik_instance.for_range(0, real_width) as col:
            position_x_scalar.set_as(local_tensor["x_int32"][col])
            position_y_scalar.set_as(local_tensor["y_int32"][col])
            with self.tik_instance.if_scope(tik.all(position_y_scalar >= 0, position_y_scalar < self.src_height,
                                                    position_x_scalar >= 0, position_x_scalar < self.src_width)):
                with self.tik_instance.if_scope(tik.all(position_y_scalar < self.src_height_1,
                                                        position_x_scalar < self.src_width_1)):
                    self.tik_instance.data_move(local_tensor["src_pixel_00"], self.src_image[
                        0, 0, position_y_scalar, position_x_scalar, 0], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(local_tensor["src_pixel_01"], self.src_image[
                        0, 0, position_y_scalar, position_x_scalar + one_scalar, 0], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(local_tensor["src_pixel_10"], self.src_image[
                        0, 0, position_y_scalar + one_scalar, position_x_scalar, 0], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(local_tensor["src_pixel_11"], self.src_image[
                        0, 0, position_y_scalar + one_scalar, position_x_scalar + one_scalar, 0], 0, 1, 1,
                                                0, 0)
                    # fp16 -> fp32
                    self.tik_instance.vconv(16, "", local_tensor["src_pixel_00_fp32"],
                                            local_tensor["src_pixel_00"], 1, 1, 1, 2, 1)
                    self.tik_instance.vconv(16, "", local_tensor["src_pixel_01_fp32"],
                                            local_tensor["src_pixel_01"], 1, 1, 1, 2, 1)
                    self.tik_instance.vconv(16, "", local_tensor["src_pixel_10_fp32"],
                                            local_tensor["src_pixel_10"], 1, 1, 1, 2, 1)
                    self.tik_instance.vconv(16, "", local_tensor["src_pixel_11_fp32"],
                                            local_tensor["src_pixel_11"], 1, 1, 1, 2, 1)

                    tmp_fp32_scalar.set_as(local_tensor["weight_fp32"][0, col])
                    self.tik_instance.vmuls(16, dst_value_fp32, local_tensor["src_pixel_00_fp32"],
                                            tmp_fp32_scalar, 1, 1, 1, 1, 1)
                    tmp_fp32_scalar.set_as(local_tensor["weight_fp32"][1, col])
                    self.tik_instance.vmuls(16, tmp_tensor, local_tensor["src_pixel_01_fp32"],
                                            tmp_fp32_scalar, 1, 1, 1, 1, 1)
                    self.tik_instance.vadd(16, dst_value_fp32, dst_value_fp32, tmp_tensor, 1, 1, 1, 1, 1, 1, 1)
                    tmp_fp32_scalar.set_as(local_tensor["weight_fp32"][2, col])
                    self.tik_instance.vmuls(16, tmp_tensor, local_tensor["src_pixel_10_fp32"],
                                            tmp_fp32_scalar, 1, 1, 1, 1, 1)
                    self.tik_instance.vadd(16, dst_value_fp32, dst_value_fp32, tmp_tensor, 1, 1, 1, 1, 1, 1, 1)
                    tmp_fp32_scalar.set_as(local_tensor["weight_fp32"][3, col])
                    self.tik_instance.vmuls(16, tmp_tensor, local_tensor["src_pixel_11_fp32"],
                                            tmp_fp32_scalar, 1, 1, 1, 1, 1)
                    self.tik_instance.vadd(16, dst_value_fp32, dst_value_fp32, tmp_tensor, 1, 1, 1, 1, 1, 1, 1)
                    self.tik_instance.vconv(16, "", local_tensor["dst_line_value"][col, 0],
                                            dst_value_fp32, 1, 1, 1, 1, 2)

    def _get_coordinate(self, raw, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, repeat_time, matrix2_fp32, matrix5_fp32,
                        matrix8_fp32, process_width, offset, local_tensor):
        m1_y_scalar = self.tik_instance.Scalar("float32")
        m4_y_scalar = self.tik_instance.Scalar("float32")
        m7_y_scalar = self.tik_instance.Scalar("float32")

        m1_y_scalar.set_as(tmp_m_mul_y_fp32[0, raw])
        m4_y_scalar.set_as(tmp_m_mul_y_fp32[1, raw])
        m7_y_scalar.set_as(tmp_m_mul_y_fp32[2, raw])

        # first_step
        # x0 equal to M0*col + M1*raw + M2
        self.tik_instance.vadds(64, local_tensor["x0_fp32"], tmp_m_mul_x_fp32[0, offset],
                                m1_y_scalar, repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(64, local_tensor["x0_fp32"], local_tensor["x0_fp32"], matrix2_fp32,
                                repeat_time, 1, 1, 8, 8)
        # y0 equal to M3*col + M4*raw + M5
        self.tik_instance.vadds(64, local_tensor["y0_fp32"], tmp_m_mul_x_fp32[1, offset], m4_y_scalar,
                                repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(64, local_tensor["y0_fp32"], local_tensor["y0_fp32"], matrix5_fp32,
                                repeat_time, 1, 1, 8, 8)
        # w0 equal to M6*col + M7*raw + M8
        self.tik_instance.vadds(64, local_tensor["w0_fp32"], tmp_m_mul_x_fp32[2, offset], m7_y_scalar,
                                repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(64, local_tensor["w0_fp32"], local_tensor["w0_fp32"], matrix8_fp32,
                                repeat_time, 1, 1, 8, 8)
        self._vrec_fp32(local_tensor["w0_fp32"], local_tensor["w0_fp32"], 3, process_width)

        # fx equal to (x0 * w0)
        self.tik_instance.vmul(64, local_tensor["fx_fp32"], local_tensor["x0_fp32"], local_tensor["w0_fp32"],
                               repeat_time, 1, 1, 1, 8, 8, 8)
        # fy equal to (y0 * w0)
        self.tik_instance.vmul(64, local_tensor["fy_fp32"], local_tensor["y0_fp32"], local_tensor["w0_fp32"],
                               repeat_time, 1, 1, 1, 8, 8, 8)

    def _get_linear_round_coordinate(self, process_width, repeat_time, local_tensor):
        # X equal to floor(fx)
        # Y equal to floor(fy)
        self._floor_f32toi32(local_tensor["x_int32"], local_tensor["fx_fp32"], process_width)
        self._floor_f32toi32(local_tensor["y_int32"], local_tensor["fy_fp32"], process_width)

        self.tik_instance.vconv(64, "", local_tensor["fx_floor_fp16"], local_tensor["x_int32"],
                                repeat_time, 1, 1, 4, 8, 1.0)
        self.tik_instance.vconv(64, "", local_tensor["fx_floor_fp32"], local_tensor["fx_floor_fp16"],
                                repeat_time, 1, 1, 8, 4)
        self.tik_instance.vconv(64, "", local_tensor["fy_floor_fp16"], local_tensor["y_int32"],
                                repeat_time, 1, 1, 4, 8, 1.0)
        self.tik_instance.vconv(64, "", local_tensor["fy_floor_fp32"], local_tensor["fy_floor_fp16"],
                                repeat_time, 1, 1, 8, 4)

        self.tik_instance.vadds(64, local_tensor["fx_ceil_fp32"], local_tensor["fx_floor_fp32"], 1,
                                repeat_time, 1, 1, 8, 8)
        self.tik_instance.vadds(64, local_tensor["fy_ceil_fp32"], local_tensor["fy_floor_fp32"], 1,
                                repeat_time, 1, 1, 8, 8)
        # x2 - col y2 - raw
        self.tik_instance.vsub(64, local_tensor["ceilx_fx_fp32"], local_tensor["fx_ceil_fp32"],
                               local_tensor["fx_fp32"],
                               repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, local_tensor["ceily_fy_fp32"], local_tensor["fy_ceil_fp32"],
                               local_tensor["fy_fp32"],
                               repeat_time, 1, 1, 1, 8, 8, 8)

        # col - x1 raw - y1
        self.tik_instance.vsub(64, local_tensor["fx_floorx_fp32"], local_tensor["fx_fp32"],
                               local_tensor["fx_floor_fp32"], repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vsub(64, local_tensor["fy_floory_fp32"], local_tensor["fy_fp32"],
                               local_tensor["fy_floor_fp32"], repeat_time, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmul(64, local_tensor["weight_fp32"][0, 0], local_tensor["ceilx_fx_fp32"],
                               local_tensor["ceily_fy_fp32"], repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(64, local_tensor["weight_fp32"][1, 0], local_tensor["fx_floorx_fp32"],
                               local_tensor["ceily_fy_fp32"], repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(64, local_tensor["weight_fp32"][2, 0], local_tensor["ceilx_fx_fp32"],
                               local_tensor["fy_floory_fp32"], repeat_time, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(64, local_tensor["weight_fp32"][3, 0], local_tensor["fx_floorx_fp32"],
                               local_tensor["fy_floory_fp32"], repeat_time, 1, 1, 1, 8, 8, 8)

    def _get_mul_value_fp32(self, tmp_m_mul_x_fp32, tmp_m_mul_y_fp32, transform_matrix_ub_fp32):
        with self.tik_instance.new_stmt_scope():
            dst_x_set_int32 = self.tik_instance.Tensor("int32", (self.align_dst_width,), name="dst_x_set_int32",
                                                       scope=tik.scope_ubuf)
            dst_y_set_int32 = self.tik_instance.Tensor("int32", (self.align_dst_height,), name="dst_y_set_int32",
                                                       scope=tik.scope_ubuf)
            dst_x_set_fp32 = self.tik_instance.Tensor("float32", (self.align_dst_width,), name="dst_x_set_fp32",
                                                      scope=tik.scope_ubuf)
            dst_y_set_fp32 = self.tik_instance.Tensor("float32", (self.align_dst_height,), name="dst_y_set_fp32",
                                                      scope=tik.scope_ubuf)
            dst_x_set_fp16 = self.tik_instance.Tensor("float16", (self.align_dst_width,), name="dst_x_set_fp16",
                                                      scope=tik.scope_ubuf)
            dst_y_set_fp16 = self.tik_instance.Tensor("float16", (self.align_dst_height,), name="dst_y_set_fp16",
                                                      scope=tik.scope_ubuf)
            matirx_value = self.tik_instance.Scalar("float32")

            with self.tik_instance.for_range(0, self.align_dst_height) as height:
                dst_y_set_int32[height] = height
            with self.tik_instance.for_range(0, self.align_dst_width) as width:
                dst_x_set_int32[width] = width

            repeat_time_w = self.align_dst_width // 64
            # width int32 -> fp16 -> fp32
            self.tik_instance.vconv(64, "", dst_x_set_fp16, dst_x_set_int32, repeat_time_w, 1, 1, 4, 8, 1.0)
            self.tik_instance.vconv(64, "", dst_x_set_fp32, dst_x_set_fp16, repeat_time_w, 1, 1, 8, 4)
            # M0*col  M3*col  M6*col
            matirx_value.set_as(transform_matrix_ub_fp32[0])
            self.tik_instance.vmuls(64, tmp_m_mul_x_fp32[0, 0], dst_x_set_fp32, matirx_value, repeat_time_w, 1, 1, 8,
                                    8)
            matirx_value.set_as(transform_matrix_ub_fp32[3])
            self.tik_instance.vmuls(64, tmp_m_mul_x_fp32[1, 0], dst_x_set_fp32, matirx_value, repeat_time_w, 1, 1, 8,
                                    8)
            matirx_value.set_as(transform_matrix_ub_fp32[6])
            self.tik_instance.vmuls(64, tmp_m_mul_x_fp32[2, 0], dst_x_set_fp32, matirx_value, repeat_time_w, 1, 1, 8,
                                    8)

            repeat_time_h = self.align_dst_height // 64
            # width int32 -> fp16 -> fp32
            self.tik_instance.vconv(64, "", dst_y_set_fp16, dst_y_set_int32, repeat_time_h, 1, 1, 4, 8, 1.0)
            self.tik_instance.vconv(64, "", dst_y_set_fp32, dst_y_set_fp16, repeat_time_h, 1, 1, 8, 4)
            # M1*raw M4*raw M7*raw
            matirx_value.set_as(transform_matrix_ub_fp32[1])
            self.tik_instance.vmuls(64, tmp_m_mul_y_fp32[0, 0], dst_y_set_fp32, matirx_value, repeat_time_h, 1, 1, 8,
                                    8)
            matirx_value.set_as(transform_matrix_ub_fp32[4])
            self.tik_instance.vmuls(64, tmp_m_mul_y_fp32[1, 0], dst_y_set_fp32, matirx_value, repeat_time_h, 1, 1, 8,
                                    8)
            matirx_value.set_as(transform_matrix_ub_fp32[7])
            self.tik_instance.vmuls(64, tmp_m_mul_y_fp32[2, 0], dst_y_set_fp32, matirx_value, repeat_time_h, 1, 1, 8,
                                    8)

    def _vrec_fp32(self, input_fp32, output_fp32, iterations, data_len):
        with self.tik_instance.new_stmt_scope():
            tmp_scalar_i32 = self.tik_instance.Scalar("int32")
            zeros_scalar_fp32 = self.tik_instance.Scalar("float32", init_value=0.0)
            ones_scalar_fp32 = self.tik_instance.Scalar("float32", init_value=1.0)
            safe_data = self.tik_instance.Tensor("float32", (data_len,), name="safe_data", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, data_len) as i:
                tmp_scalar_i32.set_as(input_fp32[i])
                with self.tik_instance.if_scope(tmp_scalar_i32 == 0):
                    safe_data[i].set_as(zeros_scalar_fp32)
                with self.tik_instance.else_scope():
                    safe_data[i].set_as(ones_scalar_fp32)
            repeat_time = data_len // 64
            two_fp32_scalar = self.tik_instance.Scalar("float32", init_value=2.0)
            save_a_fp32 = self.tik_instance.Tensor("float32", (data_len,), name="save_a_fp32", scope=tik.scope_ubuf)
            self.tik_instance.vadds(64, save_a_fp32, input_fp32, 0, repeat_time, 1, 1, 8, 8)

            xn_fp32 = self.tik_instance.Tensor("float32", (data_len,), name="xn_fp32", scope=tik.scope_ubuf)
            a_mul_xn = self.tik_instance.Tensor("float32", (data_len,), name="a_mul_xn", scope=tik.scope_ubuf)
            two_fp32 = self.tik_instance.Tensor("float32", (data_len,), name="two_fp32", scope=tik.scope_ubuf)
            two_sub_xnxa_fp32 = self.tik_instance.Tensor("float32", (data_len,), name="two_sub_xnxa_fp32",
                                                         scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(64, two_fp32, two_fp32_scalar, repeat_time, 1, 8)
            self.tik_instance.vrec(64, xn_fp32, save_a_fp32, repeat_time, 1, 1, 8, 8)
            # xn*(2 - a*xn)
            with self.tik_instance.for_range(0, iterations):
                self.tik_instance.vmul(64, a_mul_xn, save_a_fp32, xn_fp32, repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vsub(64, two_sub_xnxa_fp32, two_fp32, a_mul_xn, repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vmul(64, output_fp32, xn_fp32, two_sub_xnxa_fp32, repeat_time, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vadds(64, xn_fp32, output_fp32, 0, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vmul(64, output_fp32, safe_data, output_fp32, repeat_time, 1, 1, 1, 8, 8, 8)

    def _floor_f32toi32(self, ub_ret, ub_in, data_len):
        with self.tik_instance.new_stmt_scope():
            repeat_time = data_len // 64
            ub_functmp_f16 = self.tik_instance.Tensor("float16", (data_len,), name="ub_functmp_f16",
                                                      scope=tik.scope_ubuf)
            ub_functmp_f32 = self.tik_instance.Tensor("float32", (data_len,), name="ub_functmp_f32",
                                                      scope=tik.scope_ubuf)
            ub_functmp2_f32 = self.tik_instance.Tensor("float32", (data_len,), name="ub_functmp2_f32",
                                                       scope=tik.scope_ubuf)
            ub_functmp_i32 = self.tik_instance.Tensor("int32", (data_len,), name="ub_functmp_i32",
                                                      scope=tik.scope_ubuf)
            ub_16_f32_val0 = self.tik_instance.Tensor("float32", (data_len,), name="ub_16_f32_val0",
                                                      scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(64, ub_16_f32_val0, 0.0, repeat_time, 1, 8)

            self.tik_instance.vadds(64, ub_functmp2_f32, ub_in, 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vconv(64, "none", ub_functmp_f16, ub_functmp2_f32, repeat_time, 1, 1, 4, 8)
            self.tik_instance.vconv(64, "floor", ub_functmp_i32, ub_functmp_f16, repeat_time, 1, 1, 8, 4)
            self.tik_instance.vconv(64, "none", ub_functmp_f16, ub_functmp_i32, repeat_time, 1, 1, 4, 8, 1.0)
            self.tik_instance.vconv(64, "none", ub_functmp2_f32, ub_functmp_f16, repeat_time, 1, 1, 8, 4)
            self.tik_instance.vsub(64, ub_functmp2_f32, ub_in, ub_functmp2_f32, repeat_time, 1, 1, 1, 8, 8, 8)
            # out equal to -1 when in < 0, out equal to 0 when in bt 0.
            self.tik_instance.vmin(64, ub_functmp2_f32, ub_functmp2_f32, ub_16_f32_val0, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vrec(64, ub_functmp_f32, ub_functmp2_f32, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vabs(64, ub_functmp_f32, ub_functmp_f32, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vmul(64, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, repeat_time, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vrec(64, ub_functmp_f32, ub_functmp2_f32, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vabs(64, ub_functmp_f32, ub_functmp_f32, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vmul(64, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, repeat_time, 1, 1, 1, 8, 8, 8)
            # add 0.5 to make sure result of vconv precisely
            self.tik_instance.vadds(64, ub_functmp2_f32, ub_functmp2_f32, 0.5, repeat_time, 1, 1, 8, 8)
            self.tik_instance.vconv(64, "none", ub_functmp_f16, ub_functmp2_f32, repeat_time, 1, 1, 4, 8)
            self.tik_instance.vconv(64, "floor", ub_ret, ub_functmp_f16, repeat_time, 1, 1, 8, 4)
            self.tik_instance.vadd(64, ub_ret, ub_ret, ub_functmp_i32, repeat_time, 1, 1, 1, 8, 8, 8)

    def _ceil_div_offline(self, value, factor):
        if value % factor == 0:
            return value // factor
        else:
            return value // factor + 1


def warp_perspective(src_dict, transform_matrix_dict, dst_dict, constant_value, interpolation, dst_height, dst_width,
                     kernel_name="WarpPerspective"):
    """
    warpPerspective Op Generator.
    :param src_dict: {"shape":[], "dtype": ""}
    :param transform_matrix_dict:  {"shape":[], "dtype": ""}
    :param dst_dict: {"shape":[], "dtype": ""}
    :param interpolation: "INTER_LINEAR" or "INTER_NEAREST"
    :param constant_value: the value of boundary handles
    :param kernel_name: op kernel name
    :return:NA
    """
    src_shape = src_dict.get("shape")
    src_size = [src_shape[2], src_shape[3]]
    dst_size = [dst_height, dst_width]
    obj = WarpPerspective(src_size, dst_size, interpolation, constant_value, kernel_name)
    obj.compute()
