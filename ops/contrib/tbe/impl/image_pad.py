# -*- coding:utf-8 -*-
from te import tik

MAX_SIZE = 1024


class ImagePad:
    """
    Parameters
    ----------
    input_shape : input shape of ImagePad, [H, W, C]
    paddings : all direction pad len, [[top, bottom], [left, right], [front, back]]
    constant_value : pad constant value
    kernel_name : pad op name
    Returns
    -------
    None
    """

    def __init__(self, input_shape, paddings, constant_value=0, kernel_name="ImagePad"):

        if len(input_shape) != 3:
            raise RuntimeError("only support 3D")

        if not isinstance(constant_value, (int)):
            raise TypeError("constant_value should be int")

        if len(input_shape) != len(paddings):
            raise RuntimeError("input_shape len should be equal to paddings")

        for pad_unit in paddings:
            if len(pad_unit) != 2:
                raise RuntimeError("Paddings's shape is not in the form of (n,2)")

        max_buf_mem = MAX_SIZE * MAX_SIZE * MAX_SIZE
        if input_shape[0] * input_shape[1] * input_shape[2] * 2 > max_buf_mem or input_shape[0] * \
                input_shape[1] * input_shape[2] <= 0:
            raise RuntimeError("input mem size should less than 1GB")

        self.tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
        self.kernel_name = kernel_name

        self.paddings = paddings
        self.pad_value = constant_value

        # paddings_param
        self.pad_top_len = self.paddings[0][0]
        self.pad_bottom_len = self.paddings[0][1]
        self.pad_left_len = self.paddings[1][0]
        self.pad_right_len = self.paddings[1][1]
        self.pad_front_len = self.paddings[2][0]
        self.pad_back_len = self.paddings[2][1]

        # input_shape
        self.input_shape = input_shape
        self.input_h = input_shape[0]
        self.input_w = input_shape[1]
        self.input_c = input_shape[2]

        # output_shape
        self.output_image_height = self.pad_top_len + self.pad_bottom_len + self.input_h
        self.output_image_width = self.pad_left_len + self.pad_right_len + self.input_w
        self.output_image_channel = self.input_c + self.pad_front_len + self.pad_back_len

        if self.output_image_height * self.output_image_width * self.output_image_channel * 2 > max_buf_mem:
            raise RuntimeError("only support output less than 536870911B")

        self.output_shape = (self.output_image_height, self.output_image_width, self.output_image_channel)

        self.input_image = self.tik_inst.Tensor("float16", self.input_shape, scope=tik.scope_gm,
                                                name="input_image")
        self.output_image = self.tik_inst.Tensor("float16", self.output_shape, scope=tik.scope_gm,
                                                 name="output_image")

        self.core_1_mid_process_line = self.pad_top_len
        self.core_1_mid_process_len = self.input_h // 2 + 1
        self.core_2_mid_process_line = self.core_1_mid_process_line + self.core_1_mid_process_len
        self.core_2_mid_process_len = self.input_h - self.core_1_mid_process_len

    def compute(self):
        if self.input_shape == self.output_shape:
            self._no_pad_mode()
        else:
            self._mod1_compute()

        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_image], outputs=[self.output_image])

    def _get_over_write_ub_for_lt_16(self, h_len, tmp_line_ub, tmp_value_scalar, over_write_ub):
        with self.tik_inst.for_range(0, h_len) as iter_h:
            with self.tik_inst.for_range(0, self.input_w) as iter_w:
                with self.tik_inst.for_range(0, self.input_c) as iter_c:
                    tmp_value_scalar.set_as(tmp_line_ub[iter_h * self.input_c * self.input_w +
                                                        iter_w * self.input_c + iter_c])
                    over_write_ub[iter_h * self.pad_left_len * self.output_image_channel + iter_h *
                                  self.output_image_width * self.output_image_channel + iter_w *
                                  self.output_image_channel + self.pad_front_len + iter_c].set_as(tmp_value_scalar)

    def _overlap_lt_16(self, tmp_line_len, post_process_line_index, aligned_over_write_len):
        tmp_line_ub = self.tik_inst.Tensor("float16", (tmp_line_len,), scope=tik.scope_ubuf, name="tmp_line_ub")
        self.tik_inst.data_move(tmp_line_ub, self.input_image[post_process_line_index - self.pad_top_len, 0, 0],
                                0, 1, tmp_line_len // 16, 0, 0)
        over_write_ub = self.tik_inst.Tensor("float16", (aligned_over_write_len,), scope=tik.scope_ubuf,
                                             name="over_write_ub")
        self._generate_pad_vector_fp16(over_write_ub, aligned_over_write_len, self.pad_value)

        tmp_index_scalar = self.tik_inst.Scalar("uint16", init_value=0)
        tmp_value_scalar = self.tik_inst.Scalar("float16")
        h_len = aligned_over_write_len // (self.output_image_width * self.output_image_channel)

        self._get_over_write_ub_for_lt_16(h_len, tmp_line_ub, tmp_value_scalar, over_write_ub)
        tmp_index_scalar.set_as((h_len - 1) * self.pad_left_len * self.output_image_channel + h_len *
                                self.output_image_width * self.output_image_channel)
        with self.tik_inst.if_scope(tmp_index_scalar < aligned_over_write_len):
            tmp_index_scalar.set_as(tmp_index_scalar + self.pad_left_len * self.output_image_channel)
        with self.tik_inst.else_scope():
            pass
        with self.tik_inst.if_scope(tmp_index_scalar < aligned_over_write_len):
            w_len_scalar = self.tik_inst.Scalar("uint16")
            w_len_scalar.set_as(aligned_over_write_len - tmp_index_scalar)
            w_len_scalar.set_as(w_len_scalar // self.output_image_channel)
            with self.tik_inst.for_range(0, w_len_scalar) as w_len:
                with self.tik_inst.for_range(0, self.input_c) as in_c:
                    tmp_value_scalar.set_as(
                        tmp_line_ub[h_len * self.input_w * self.input_c + w_len * self.input_c + in_c])
                    over_write_ub[
                        h_len * self.output_image_width * self.output_image_channel + w_len *
                        self.output_image_channel + self.pad_front_len + in_c].set_as(tmp_value_scalar)

            tmp_index_scalar.set_as(
                tmp_index_scalar + w_len_scalar * self.output_image_channel + self.pad_front_len)

            with self.tik_inst.if_scope(tmp_index_scalar < aligned_over_write_len):
                c_len_scalar = self.tik_inst.Scalar("int64")
                c_len_scalar.set_as(aligned_over_write_len - tmp_index_scalar)
                cmp_scalar = self.tik_inst.Scalar("int64", init_value=self.input_c)
                self.tik_inst.scalar_min(c_len_scalar, c_len_scalar, cmp_scalar)
                with self.tik_inst.for_range(0, c_len_scalar) as iter_c:
                    tmp_value_scalar.set_as(
                        tmp_line_ub[h_len * self.input_w * self.input_c + w_len_scalar * self.input_c + iter_c])
                    over_write_ub[tmp_index_scalar + iter_c].set_as(tmp_value_scalar)
            with self.tik_inst.else_scope():
                pass
        with self.tik_inst.else_scope():
            pass
        self.tik_inst.data_move(self.output_image[post_process_line_index, self.pad_left_len, 0], over_write_ub, 0,
                                1, aligned_over_write_len // 16, 0, 0)

    def _overlap_bt_16_no_pad_c(self, tmp_line_len, post_process_line_index, aligned_over_write_len):
        with self.tik_inst.new_stmt_scope():
            if self.input_w * self.input_c >= tmp_line_len:
                tmp_line_ub = self.tik_inst.Tensor("float16", (tmp_line_len,), scope=tik.scope_ubuf,
                                                   name="tmp_line_ub")
                self.tik_inst.data_move(tmp_line_ub,
                                        self.input_image[post_process_line_index - self.pad_top_len, 0, 0], 0,
                                        1, tmp_line_len // 16, 0, 0)
                self.tik_inst.data_move(self.output_image[post_process_line_index, self.pad_left_len, 0],
                                        tmp_line_ub, 0, 1, aligned_over_write_len // 16, 0, 0)
            else:
                tmp_help_scalar = self.tik_inst.Scalar("float16")
                tmp_line_ub = self.tik_inst.Tensor("float16", (tmp_line_len,), scope=tik.scope_ubuf,
                                                   name="tmp_line_ub")
                tmp_line_back_ub = self.tik_inst.Tensor("float16", (tmp_line_len,), scope=tik.scope_ubuf,
                                                        name="tmp_line_back_ub")
                self.tik_inst.data_move(tmp_line_back_ub,
                                        self.input_image[post_process_line_index - self.pad_top_len, 0, 0], 0,
                                        1, tmp_line_len // 16, 0, 0)
                self._generate_pad_vector_fp16(tmp_line_ub, tmp_line_len, self.pad_value)
                with self.tik_inst.for_range(0, self.input_w) as i:
                    with self.tik_inst.for_range(0, self.input_c) as j:
                        tmp_help_scalar.set_as(tmp_line_back_ub[i * self.input_c + j])
                        tmp_line_ub[i * self.input_c + j].set_as(tmp_help_scalar)
                self.tik_inst.data_move(self.output_image[post_process_line_index, self.pad_left_len, 0],
                                        tmp_line_ub, 0, 1, aligned_over_write_len // 16, 0, 0)

    def _overlap_bt_16_with_pad_c(self, tmp_line_len, post_process_line_index, aligned_over_write_len):
        with self.tik_inst.new_stmt_scope():
            tmp_line_ub = self.tik_inst.Tensor("float16", (tmp_line_len,), scope=tik.scope_ubuf,
                                               name="tmp_line_ub")
            self.tik_inst.data_move(tmp_line_ub,
                                    self.input_image[post_process_line_index - self.pad_top_len, 0, 0], 0, 1,
                                    tmp_line_len // 16, 0, 0)
            invalid_data_len = tmp_line_len - self.input_w * self.input_c
            if invalid_data_len > 0:
                invalid_process_scalar = self.tik_inst.Scalar("float16", init_value=self.pad_value)
                with self.tik_inst.for_range(0, invalid_data_len) as i:
                    tmp_line_ub[self.input_w * self.input_c + i].set_as(invalid_process_scalar)

            over_write_ub = self.tik_inst.Tensor("float16", (aligned_over_write_len,), scope=tik.scope_ubuf,
                                                 name="over_write_ub")
            self._generate_pad_vector_fp16(over_write_ub, aligned_over_write_len, self.pad_value)

            tmp_index_scalar = self.tik_inst.Scalar("uint16")
            tmp_index_scalar.set_as(0)
            tmp_value_scalar = self.tik_inst.Scalar("float16")

            unit_len = self.input_c + self.pad_front_len + self.pad_back_len
            rp_t = self.tik_inst.Scalar("uint16")
            rp_t.set_as(aligned_over_write_len)
            rp_t.set_as(rp_t // unit_len)

            with self.tik_inst.for_range(0, rp_t) as i:
                with self.tik_inst.for_range(0, self.input_c) as j:
                    tmp_value_scalar.set_as(tmp_line_ub[i * self.input_c + j])
                    over_write_ub[i * unit_len + self.pad_front_len + j].set_as(tmp_value_scalar)

            tmp_index_scalar.set_as(tmp_index_scalar + rp_t * unit_len + self.pad_front_len)
            with self.tik_inst.if_scope(tmp_index_scalar < aligned_over_write_len):
                last_len_scalar = self.tik_inst.Scalar("int64")
                last_len_scalar.set_as(aligned_over_write_len - tmp_index_scalar)
                cmp_scalar = self.tik_inst.Scalar("int64", init_value=self.input_c)
                self.tik_inst.scalar_min(last_len_scalar, last_len_scalar, cmp_scalar)

                with self.tik_inst.for_range(0, last_len_scalar) as i:
                    tmp_value_scalar.set_as(tmp_line_ub[rp_t * self.input_c + i])
                    over_write_ub[tmp_index_scalar + i].set_as(tmp_value_scalar)

            self.tik_inst.data_move(self.output_image[post_process_line_index, self.pad_left_len, 0],
                                    over_write_ub, 0, 1, aligned_over_write_len // 16, 0, 0)

    # only process for double core
    def _mod1_compute(self):
        with self.tik_inst.for_range(0, 2, block_num=2) as index:
            with self.tik_inst.if_scope(index == 0):
                self._pad_process_core_1()
            with self.tik_inst.else_scope():
                self._pad_process_core_2()

        # post process over write
        post_process_line_index = self.core_2_mid_process_line
        aligned_over_write_len = 16
        over_write_point_width = aligned_over_write_len // self.output_image_channel + 1
        tmp_line_len = self._align_offline(over_write_point_width * self.input_c, 16)

        if self.core_2_mid_process_len == 0:
            pass
        elif self.output_image_channel * self.output_image_width < aligned_over_write_len:
            self._overlap_lt_16(tmp_line_len, post_process_line_index, aligned_over_write_len)
        else:
            if self.pad_front_len == 0 and self.pad_back_len == 0:
                self._overlap_bt_16_no_pad_c(tmp_line_len, post_process_line_index, aligned_over_write_len)
            else:
                self._overlap_bt_16_with_pad_c(tmp_line_len, post_process_line_index, aligned_over_write_len)

    # process top and mid
    def _pad_process_core_1(self):
        if self.pad_top_len != 0:
            top_pad_vector_len = self._align_offline(
                (self.output_image_width * self.pad_top_len + self.pad_left_len) * self.output_image_channel, 16)
            self._top_process(top_pad_vector_len)
        else:
            top_pad_vector_len = self._align_offline(self.pad_left_len * self.output_image_channel, 16)
            self._top_process(top_pad_vector_len)
        self._mid_process(self.core_1_mid_process_line, self.core_1_mid_process_len, 1)

    # process mid and bottom
    def _pad_process_core_2(self):
        if self.core_2_mid_process_len != 0:
            self._mid_process(self.core_2_mid_process_line, self.core_2_mid_process_len, 2)

        if self.pad_bottom_len != 0:
            bottom_pad_vector_len = self._align_offline(
                (self.output_image_width * self.pad_bottom_len) * self.output_image_channel, 16)
            self._bottom_process(bottom_pad_vector_len)

    def _top_process(self, top_pad_vector_len):
        if top_pad_vector_len == 0:
            return
        if top_pad_vector_len * 2 < 200 * 1024:
            with self.tik_inst.new_stmt_scope():
                top_pad_vector = self.tik_inst.Tensor("float16", (top_pad_vector_len,), scope=tik.scope_ubuf,
                                                      name="top_pad_vector")
                self._generate_pad_vector_fp16(top_pad_vector, top_pad_vector_len, self.pad_value)
                self.tik_inst.data_move(self.output_image, top_pad_vector, 0, 1, top_pad_vector_len // 16, 0, 0)
        else:
            each_loop_pad_vector_len = 200 * 1024 // 2
            loop_times = top_pad_vector_len // each_loop_pad_vector_len
            last_pad_data_len = top_pad_vector_len - loop_times * each_loop_pad_vector_len
            with self.tik_inst.new_stmt_scope():
                top_pad_vector = self.tik_inst.Tensor("float16", (each_loop_pad_vector_len,), scope=tik.scope_ubuf,
                                                      name="top_pad_vector")
                self._generate_pad_vector_fp16(top_pad_vector, each_loop_pad_vector_len, self.pad_value)
                with self.tik_inst.for_range(0, loop_times) as i:
                    self.tik_inst.data_move(self.output_image[i * each_loop_pad_vector_len], top_pad_vector, 0, 1,
                                            each_loop_pad_vector_len // 16, 0, 0)
            if last_pad_data_len != 0:
                with self.tik_inst.new_stmt_scope():
                    top_pad_vector_last = self.tik_inst.Tensor("float16", (last_pad_data_len,), scope=tik.scope_ubuf,
                                                               name="top_pad_value_last")
                    self._generate_pad_vector_fp16(top_pad_vector_last, last_pad_data_len, self.pad_value)
                    self.tik_inst.data_move(self.output_image[loop_times * each_loop_pad_vector_len],
                                            top_pad_vector_last, 0, 1, last_pad_data_len // 16, 0, 0)

    def _bottom_process(self, bottom_pad_vector_len):
        if bottom_pad_vector_len * 2 < 200 * 1024:
            with self.tik_inst.new_stmt_scope():
                bottom_pad_vector = self.tik_inst.Tensor("float16", (bottom_pad_vector_len,), scope=tik.scope_ubuf,
                                                         name="bottom_pad_vector")
                self._generate_pad_vector_fp16(bottom_pad_vector, bottom_pad_vector_len, self.pad_value)
                self.tik_inst.data_move(self.output_image[self.pad_top_len + self.input_h, 0, 0], bottom_pad_vector, 0,
                                        1, bottom_pad_vector_len // 16, 0, 0)
        else:
            each_loop_pad_vector_len = 200 * 1024 // 2
            loop_times = bottom_pad_vector_len // each_loop_pad_vector_len
            last_pad_data_len = bottom_pad_vector_len - loop_times * each_loop_pad_vector_len

            with self.tik_inst.new_stmt_scope():
                bottom_pad_vector = self.tik_inst.Tensor("float16", (each_loop_pad_vector_len,), scope=tik.scope_ubuf,
                                                         name="bottom_pad_vector")
                self._generate_pad_vector_fp16(bottom_pad_vector, each_loop_pad_vector_len, self.pad_value)
                start_index = self.output_image_width * self.output_image_channel * (self.pad_top_len + self.input_h)
                with self.tik_inst.for_range(0, loop_times) as i:
                    self.tik_inst.data_move(self.output_image[start_index + i * each_loop_pad_vector_len],
                                            bottom_pad_vector, 0, 1, each_loop_pad_vector_len // 16, 0, 0)
            if last_pad_data_len != 0:
                with self.tik_inst.new_stmt_scope():
                    bottom_pad_vector_last = self.tik_inst.Tensor("float16", (last_pad_data_len,),
                                                                  scope=tik.scope_ubuf, name="bottom_pad_vector_last")
                    self._generate_pad_vector_fp16(bottom_pad_vector_last, last_pad_data_len, self.pad_value)
                    self.tik_inst.data_move(self.output_image[start_index + loop_times * each_loop_pad_vector_len],
                                            bottom_pad_vector_last, 0, 1, last_pad_data_len // 16, 0, 0)

    def _mid_no_pad_c_lt_exe(self, process_len, start_line, image_width_align, last_line_index, mid_modify_len,
                             mid_modify_len_new, bridge_ub, mid_modify_ub):
        with self.tik_inst.for_range(0, process_len) as i:
            self.tik_inst.data_move(bridge_ub, self.input_image[start_line - self.pad_top_len + i, 0, 0],
                                    0, 1, image_width_align // 16, 0, 0)
            self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len, 0], bridge_ub, 0,
                                    1, image_width_align // 16, 0, 0)
            if self.pad_left_len != 0 or self.pad_right_len != 0:
                with self.tik_inst.if_scope(i != last_line_index):
                    self.tik_inst.data_move(
                        self.output_image[start_line + i, self.pad_left_len + self.input_w, 0],
                        mid_modify_ub, 0, 1, mid_modify_len // 16, 0, 0)
                with self.tik_inst.else_scope():
                    self.tik_inst.data_move(
                        self.output_image[start_line + i, self.pad_left_len + self.input_w, 0],
                        mid_modify_ub, 0, 1, mid_modify_len_new // 16, 0, 0)

    def _mid_process_no_pad_c_lt_200k(self, core_index, process_len, start_line, image_width_align, mid_modify_len):
        if (core_index == 2) or (core_index == 1 and self.core_2_mid_process_len == 0):
            last_line_index = process_len - 1
            mid_modify_len_new = self._align_offline(self.pad_right_len * self.output_image_channel, 16)
        else:
            last_line_index = process_len
            mid_modify_len_new = mid_modify_len
        with self.tik_inst.new_stmt_scope():
            if self.pad_left_len != 0 or self.pad_right_len != 0:
                mid_modify_ub = self.tik_inst.Tensor("float16", (mid_modify_len,), scope=tik.scope_ubuf,
                                                     name="mid_modify_ub")
                self._generate_pad_vector_fp16(mid_modify_ub, mid_modify_len, self.pad_value)
            bridge_ub = self.tik_inst.Tensor("float16", (image_width_align,), scope=tik.scope_ubuf,
                                             name="bridge_ub")
            self._mid_no_pad_c_lt_exe(process_len, start_line, image_width_align, last_line_index, mid_modify_len,
                                      mid_modify_len_new, bridge_ub, mid_modify_ub)

    def _mid_no_pad_c_bt_exe_for_input(self, rp_t, common_vector, start_line, max_width, repeat_len, last_width,
                                       last_len, i):
        if rp_t > 0:
            with self.tik_inst.for_range(0, rp_t) as j:
                self.tik_inst.data_move(common_vector, self.input_image[
                    start_line - self.pad_top_len + i, j * max_width, 0],
                                        0, 1, repeat_len // 16, 0, 0)
                self.tik_inst.data_move(
                    self.output_image[start_line + i, self.pad_left_len + j * max_width, 0],
                    common_vector, 0, 1, repeat_len // 16, 0, 0)

        if last_width != 0:
            self.tik_inst.data_move(common_vector, self.input_image[
                start_line - self.pad_top_len + i, rp_t * max_width, 0],
                                    0, 1, last_len // 16, 0, 0)
            self.tik_inst.data_move(
                self.output_image[start_line + i, self.pad_left_len + rp_t * max_width, 0],
                common_vector, 0, 1, last_len // 16, 0, 0)

    def _mid_no_pad_c_bt_exe_norm(self, common_vector, repeat_len, last_line_index, rp_p, start_line, max_width,
                                  last_line_rp, i):
        self._generate_pad_vector_fp16(common_vector, repeat_len, self.pad_value)
        with self.tik_inst.if_scope(i != last_line_index):
            if rp_p > 0:
                with self.tik_inst.for_range(0, rp_p) as j:
                    self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len +
                                                              self.input_w + j * max_width, 0],
                                            common_vector, 0, 1, repeat_len // 16, 0, 0)
            else:
                pass

        with self.tik_inst.else_scope():
            if last_line_rp > 0:
                with self.tik_inst.for_range(0, last_line_rp) as j:
                    self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len +
                                                              self.input_w + j * max_width, 0],
                                            common_vector, 0, 1, repeat_len // 16, 0, 0)
            else:
                pass

    def _mid_no_pad_c_bt_exe_last(self, last_line_index, last_p_len, start_line, rp_p, max_width, common_vector,
                                  re_line_width, last_line_rp, re_line_len, i):
        with self.tik_inst.if_scope(i != last_line_index):
            if last_p_len != 0:
                self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len +
                                                          self.input_w + rp_p * max_width, 0],
                                        common_vector, 0, 1, last_p_len // 16, 0, 0)
            else:
                pass

        with self.tik_inst.else_scope():
            if re_line_width > 0:
                self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len +
                                                          self.input_w + last_line_rp * max_width, 0],
                                        common_vector, 0, 1, re_line_len // 16, 0, 0)
            else:
                pass

    def _mid_no_pad_c_bt_exe(self, start_line, process_len, max_width, repeat_len, last_len, rp_t,
                             rp_p, last_line_rp, last_p_len, re_line_len, re_line_width, common_vector,
                             last_line_index, last_width):
        with self.tik_inst.for_range(0, process_len) as i:
            # process input
            self._mid_no_pad_c_bt_exe_for_input(rp_t, common_vector, start_line, max_width, repeat_len, last_width,
                                                last_len, i)

            # process pad
            self._mid_no_pad_c_bt_exe_norm(common_vector, repeat_len, last_line_index, rp_p, start_line, max_width,
                                           last_line_rp, i)
            self._mid_no_pad_c_bt_exe_last(last_line_index, last_p_len, start_line, rp_p, max_width, common_vector,
                                           re_line_width, last_line_rp, re_line_len, i)

    def _mid_process_no_pad_c_bt_200k(self, core_index, process_len, start_line):
        max_data_len = 100 * 1024
        max_width = max_data_len // self.input_c
        repeat_len = self._align_offline((max_width * self.input_c), 16)
        rp_t = self.input_w // max_width
        last_width = self.input_w - max_width * rp_t
        last_len = self._align_offline((last_width * self.input_c), 16)

        pad_width = self.pad_left_len + self.pad_right_len
        pad_len = self._align_offline((self.pad_left_len + self.pad_right_len), 16)
        rp_p = pad_len // max_width
        last_p_with = pad_width - rp_p * max_width
        last_p_len = self._align_offline((last_p_with * self.input_c), 16)

        if (core_index == 2) or (core_index == 1 and self.core_2_mid_process_len == 0):
            last_line_index = process_len - 1
            last_line_rp = self.pad_right_len // max_width
            re_line_width = self.pad_right_len - last_line_rp * max_width
            re_line_len = self._align_offline(re_line_width * self.input_c, 16)
        else:
            last_line_index = process_len
            last_line_rp = rp_p
            re_line_width = last_p_with
            re_line_len = last_p_len

        with self.tik_inst.new_stmt_scope():
            common_vector = self.tik_inst.Tensor("float16", (max_data_len,), scope=tik.scope_ubuf,
                                                 name="common_vector")
            self._mid_no_pad_c_bt_exe(start_line, process_len, max_width, repeat_len, last_len, rp_t,
                                      rp_p, last_line_rp, last_p_len, re_line_len, re_line_width, common_vector,
                                      last_line_index, last_width)

    def _mid_process_no_pad_c(self, core_index, process_len, start_line):
        image_width_align = self._align_offline(self.input_w * self.output_image_channel, 16)
        mid_modify_len = self._align_offline((self.pad_left_len + self.pad_right_len) * self.output_image_channel, 16)
        if (image_width_align + mid_modify_len) * 2 <= 200 * 1024:
            self._mid_process_no_pad_c_lt_200k(core_index, process_len, start_line, image_width_align, mid_modify_len)
        else:
            self._mid_process_no_pad_c_bt_200k(core_index, process_len, start_line)

    def _mid_with_pad_c_lt_exe(self, process_len, start_line, line_tensor, line_tensor_len, tmp_scalar, tmp_tensor,
                               tmp_tensor_len, last_line_index, last_process_len):
        with self.tik_inst.for_range(0, process_len) as i:
            self.tik_inst.data_move(line_tensor, self.input_image[start_line - self.pad_top_len + i, 0, 0],
                                    0, 1, line_tensor_len // 16, 0, 0)
            with self.tik_inst.for_range(0, self.input_w) as j:
                with self.tik_inst.for_range(0, self.input_c) as k:
                    tmp_scalar.set_as(line_tensor[j * self.input_c + k])
                    tmp_tensor[self.pad_front_len + j * self.output_image_channel + k].set_as(tmp_scalar)

            with self.tik_inst.if_scope(i != last_line_index):
                self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len, 0],
                                        tmp_tensor, 0, 1, tmp_tensor_len // 16, 0, 0)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(self.output_image[start_line + i, self.pad_left_len, 0],
                                        tmp_tensor, 0, 1, last_process_len // 16, 0, 0)

    def _mid_process_with_pad_c_lt_100k(self, core_index, process_len, start_line):
        with self.tik_inst.new_stmt_scope():
            tmp_tensor_len = self._align_offline((self.output_image_width * self.output_image_channel), 16)
            tmp_tensor = self.tik_inst.Tensor("float16", (tmp_tensor_len,), scope=tik.scope_ubuf,
                                              name="tmp_tensor")
            self._generate_pad_vector_fp16(tmp_tensor, tmp_tensor_len, self.pad_value)
            line_tensor_len = self._align_offline((self.input_w * self.input_c), 16)
            line_tensor = self.tik_inst.Tensor("float16", (line_tensor_len,), scope=tik.scope_ubuf,
                                               name="line_tensor")
            tmp_scalar = self.tik_inst.Scalar("float16")

            if (core_index == 2) or (core_index == 1 and self.core_2_mid_process_len == 0):
                last_line_index = process_len - 1
                last_process_len = self._align_offline(
                    (self.output_image_width - self.pad_left_len) * self.output_image_channel, 16)
            else:
                last_line_index = process_len
                last_process_len = tmp_tensor_len
            self._mid_with_pad_c_lt_exe(process_len, start_line, line_tensor, line_tensor_len, tmp_scalar, tmp_tensor,
                                        tmp_tensor_len, last_line_index, last_process_len)

    def _mid_with_pad_c_bt_exe_norm_each_batch_width(self, batch_width, tmp_scalar, recv_vec_ub, send_vec_ub,
                                                     output_point_channels):
        with self.tik_inst.for_range(0, batch_width) as k:
            with self.tik_inst.for_range(0, self.input_c) as iter_p:
                tmp_scalar.set_as(recv_vec_ub[k * self.input_c + iter_p])
                send_vec_ub[k * output_point_channels + self.pad_front_len + iter_p].set_as(
                    tmp_scalar)

    def _mid_with_pad_c_bt_exe_norm(self, repeat_time, recv_vec_ub, send_vec_ub, single_batch_send_vec_len,
                                    single_batch_recv_vec_len, start_line, batch_width, output_point_channels,
                                    tmp_scalar, last_width, last_recv_vec_len, last_send_vec_len, i):
        if repeat_time > 0:
            with self.tik_inst.for_range(0, repeat_time) as j:
                self._generate_pad_vector_fp16(send_vec_ub, single_batch_send_vec_len, self.pad_value)
                self.tik_inst.data_move(recv_vec_ub, self.input_image[
                    start_line - self.pad_top_len + i, j * batch_width, 0],
                                        0, 1, single_batch_recv_vec_len // 16, 0, 0)
                self._mid_with_pad_c_bt_exe_norm_each_batch_width(batch_width, tmp_scalar, recv_vec_ub, send_vec_ub,
                                                                  output_point_channels)
                self.tik_inst.data_move(
                    self.output_image[start_line + i, self.pad_left_len + j * batch_width, 0],
                    send_vec_ub, 0, 1, single_batch_send_vec_len // 16, 0, 0)

        if last_width != 0:
            self._generate_pad_vector_fp16(send_vec_ub, single_batch_send_vec_len, self.pad_value)
            self.tik_inst.data_move(recv_vec_ub, self.input_image[
                start_line - self.pad_top_len + i, repeat_time * batch_width, 0],
                                    0, 1, last_recv_vec_len // 16, 0, 0)
            self._mid_with_pad_c_bt_exe_norm_each_batch_width(last_width, tmp_scalar, recv_vec_ub, send_vec_ub,
                                                              output_point_channels)

            self.tik_inst.data_move(
                self.output_image[start_line + i, self.pad_left_len + repeat_time * batch_width, 0],
                send_vec_ub, 0, 1, last_send_vec_len // 16, 0, 0)

    def _mid_with_pad_c_bt_exe_last_left(self, last_line_index, last_for_x_width, cols_index_scalar, repeat_time_for_x,
                                         batch_width, start_line, rows_index_scalar, send_vec_ub, last_for_x_len,
                                         re_line_width, last_line_rp, re_line_len, i):
        with self.tik_inst.if_scope(i != last_line_index):
            if last_for_x_width > 0:
                cols_index_scalar.set_as(
                    self.pad_left_len + self.input_w + repeat_time_for_x * batch_width)
                with self.tik_inst.if_scope(cols_index_scalar > self.output_image_width):
                    rows_index_scalar.set_as(start_line + 1)
                with self.tik_inst.else_scope():
                    rows_index_scalar.set_as(start_line)
                cols_index_scalar.set_as(cols_index_scalar % self.output_image_width)
                self.tik_inst.data_move(self.output_image[rows_index_scalar + i, cols_index_scalar, 0],
                                        send_vec_ub, 0, 1, last_for_x_len // 16, 0, 0)
        with self.tik_inst.else_scope():
            if re_line_width > 0:
                cols_index_scalar.set_as(self.pad_left_len + self.input_w + last_line_rp * batch_width)
                with self.tik_inst.if_scope(cols_index_scalar > self.output_image_width):
                    rows_index_scalar.set_as(start_line + 1)
                with self.tik_inst.else_scope():
                    rows_index_scalar.set_as(start_line)
                cols_index_scalar.set_as(cols_index_scalar % self.output_image_width)
                self.tik_inst.data_move(self.output_image[rows_index_scalar + i, cols_index_scalar, 0],
                                        send_vec_ub, 0, 1, re_line_len // 16, 0, 0)

    def _mid_with_pad_c_bt_exe_last_each_repeat(self, cols_index_scalar, batch_width, rows_index_scalar, start_line,
                                                send_vec_ub, pad_len, i, j):
        cols_index_scalar.set_as(self.pad_left_len + self.input_w + j * batch_width)
        with self.tik_inst.if_scope(cols_index_scalar > self.output_image_width):
            rows_index_scalar.set_as(start_line + 1)
        with self.tik_inst.else_scope():
            rows_index_scalar.set_as(start_line)
        cols_index_scalar.set_as(cols_index_scalar % self.output_image_width)
        self.tik_inst.data_move(
            self.output_image[rows_index_scalar + i, cols_index_scalar, 0],
            send_vec_ub, 0, 1, pad_len // 16, 0, 0)

    def _mid_with_pad_c_bt_exe_last(self, send_vec_ub, single_batch_send_vec_len, last_line_index, repeat_time_for_x,
                                    rows_index_scalar, cols_index_scalar, batch_width, start_line, last_line_rp,
                                    left_right_pad_len, re_line_len, last_for_x_width, last_for_x_len,
                                    re_line_width, i):
        self._generate_pad_vector_fp16(send_vec_ub, single_batch_send_vec_len, self.pad_value)
        with self.tik_inst.if_scope(i != last_line_index):
            if repeat_time_for_x > 0:
                with self.tik_inst.for_range(0, repeat_time_for_x) as j:
                    self._mid_with_pad_c_bt_exe_last_each_repeat(self, cols_index_scalar, batch_width,
                                                                 rows_index_scalar, start_line, send_vec_ub,
                                                                 left_right_pad_len, i, j)
            else:
                pass

        with self.tik_inst.else_scope():
            if last_line_rp > 0:
                with self.tik_inst.for_range(0, last_line_rp) as j:
                    self._mid_with_pad_c_bt_exe_last_each_repeat(self, cols_index_scalar, batch_width,
                                                                 rows_index_scalar, start_line, send_vec_ub,
                                                                 single_batch_send_vec_len, i, j)
            else:
                pass

        self._mid_with_pad_c_bt_exe_last_left(last_line_index, last_for_x_width, cols_index_scalar, repeat_time_for_x,
                                              batch_width, start_line, rows_index_scalar, send_vec_ub, last_for_x_len,
                                              re_line_width, last_line_rp, re_line_len, i)

    def _mid_with_pad_c_bt_exe(self, process_len, start_line, repeat_time, send_vec_ub, single_batch_send_vec_len,
                               single_batch_recv_vec_len, recv_vec_ub, output_point_channels, tmp_scalar,
                               last_recv_vec_len, last_width, batch_width, last_send_vec_len, repeat_time_for_x,
                               cols_index_scalar, rows_index_scalar, left_right_pad_len, last_line_index, re_line_len,
                               last_line_rp, last_for_x_width, last_for_x_len, re_line_width):
        with self.tik_inst.for_range(0, process_len) as i:
            self._mid_with_pad_c_bt_exe_norm(repeat_time, recv_vec_ub, send_vec_ub, single_batch_send_vec_len,
                                             single_batch_recv_vec_len, start_line, batch_width, output_point_channels,
                                             tmp_scalar, last_width, last_recv_vec_len, last_send_vec_len, i)
            self._mid_with_pad_c_bt_exe_last(send_vec_ub, single_batch_send_vec_len, last_line_index,
                                             repeat_time_for_x,
                                             rows_index_scalar, cols_index_scalar, batch_width, start_line,
                                             last_line_rp, left_right_pad_len, re_line_len, last_for_x_width,
                                             last_for_x_len, re_line_width, i)

    def _mid_process_with_pad_c_bt_100k(self, core_index, process_len, start_line, max_data_len):
        output_point_channels = self.input_c + self.pad_front_len + self.pad_back_len
        single_point_buffer_len = self.input_c + output_point_channels
        batch_width = max_data_len // single_point_buffer_len
        repeat_time = self.input_w // batch_width
        last_width = self.input_w - repeat_time * batch_width
        single_batch_recv_vec_len = self._align_offline((batch_width * self.input_c), 16)
        single_batch_send_vec_len = self._align_offline(batch_width * output_point_channels, 16)
        last_recv_vec_len = self._align_offline((last_width * self.input_c), 16)
        last_send_vec_len = self._align_offline(last_width * output_point_channels, 16)

        left_right_pad_width = self.pad_left_len + self.pad_right_len
        repeat_time_for_x = left_right_pad_width // batch_width
        left_right_pad_len = self._align_offline(batch_width * output_point_channels, 16)
        last_for_x_width = left_right_pad_width - repeat_time_for_x * batch_width
        last_for_x_len = self._align_offline(last_for_x_width * output_point_channels, 16)

        if (core_index == 2) or (core_index == 1 and self.core_2_mid_process_len == 0):
            last_line_index = process_len - 1
            last_line_rp = self.pad_right_len // batch_width
            re_line_width = self.pad_right_len - last_line_rp * batch_width
            re_line_len = self._align_offline(re_line_width * self.output_image_channel, 16)
        else:
            last_line_index = process_len
            last_line_rp = repeat_time_for_x
            re_line_width = last_for_x_width
            re_line_len = last_for_x_len

        with self.tik_inst.new_stmt_scope():
            tmp_scalar = self.tik_inst.Scalar("float16")
            rows_index_scalar = self.tik_inst.Scalar("uint32")
            cols_index_scalar = self.tik_inst.Scalar("uint32")
            recv_vec_ub = self.tik_inst.Tensor("float16", (single_batch_recv_vec_len,), scope=tik.scope_ubuf,
                                               name="recv_vec_ub")
            send_vec_ub = self.tik_inst.Tensor("float16", (single_batch_send_vec_len,), scope=tik.scope_ubuf,
                                               name="send_vec_ub")

            self._mid_with_pad_c_bt_exe(process_len, start_line, repeat_time, send_vec_ub, single_batch_send_vec_len,
                                        single_batch_recv_vec_len, recv_vec_ub, output_point_channels,
                                        tmp_scalar, last_recv_vec_len, last_width, batch_width, last_send_vec_len,
                                        repeat_time_for_x, cols_index_scalar, rows_index_scalar, left_right_pad_len,
                                        last_line_index, re_line_len, last_line_rp, last_for_x_width, last_for_x_len,
                                        re_line_width)

    def _mid_process_with_pad_c(self, core_index, process_len, start_line):
        max_data_len = 100 * 1024
        total_data_num = (self._align_offline(self.input_w * self.input_c, 16) +
                          self._align_offline((self.input_w * self.output_image_channel), 16) +
                          self._align_offline((self.pad_left_len + self.pad_right_len) *
                                              self.output_image_channel, 16))

        if total_data_num <= max_data_len:
            self._mid_process_with_pad_c_lt_100k(core_index, process_len, start_line)
        else:
            self._mid_process_with_pad_c_bt_100k(core_index, process_len, start_line, max_data_len)

    def _mid_process(self, start_line, process_len, core_index):
        if self.pad_front_len == 0 and self.pad_back_len == 0:
            self._mid_process_no_pad_c(core_index, process_len, start_line)
        else:
            self._mid_process_with_pad_c(core_index, process_len, start_line)

    def _no_pad_mode(self):
        with self.tik_inst.new_stmt_scope():
            max_size = 120 * 1024
            tmp_tensor = self.tik_inst.Tensor("float16", (max_size,), scope=tik.scope_ubuf, name="tmp_tensor")
            input_size = self._align_offline(self.input_h * self.input_w * self.input_c, 16)
            repeat_time = input_size // max_size
            last_size = self._align_offline((input_size - repeat_time * max_size), 16)
            if repeat_time != 0:
                with self.tik_inst.for_range(0, repeat_time) as i:
                    self.tik_inst.data_move(tmp_tensor, self.input_image[i * max_size], 0, 1, max_size // 16, 0, 0)
                    self.tik_inst.data_move(self.output_image[i * max_size], tmp_tensor, 0, 1, max_size // 16, 0, 0)
                if last_size != 0:
                    self.tik_inst.data_move(tmp_tensor, self.input_image[repeat_time * max_size], 0, 1,
                                            last_size // 16, 0, 0)
                    self.tik_inst.data_move(self.output_image[repeat_time * max_size], tmp_tensor, 0, 1,
                                            last_size // 16, 0, 0)
            else:
                self.tik_inst.data_move(tmp_tensor, self.input_image, 0, 1, last_size // 16, 0, 0)
                self.tik_inst.data_move(self.output_image, tmp_tensor, 0, 1, last_size // 16, 0, 0)

    def _generate_pad_vector_fp16(self, pad_vector_ub, pad_vector_len, pad_value):
        if len(pad_vector_ub.shape) != 1:
            raise RuntimeError("pad_vector_ub not a vector")

        if pad_vector_len > pad_vector_ub.shape[0]:
            raise RuntimeError("pad_vector_len must less than pad_vector_ub.shape[0]")

        if pad_vector_len % 16 != 0:
            raise TypeError("pad_vector_len should align to 16")

        if pad_vector_len * 2 > 200 * 1024:
            raise RuntimeError("pad_vector_len is too large")

        if not isinstance(pad_value, (int)):
            raise TypeError("pad_value should be int")

        with self.tik_inst.new_stmt_scope():
            repeat_time = self.tik_inst.Scalar("uint32")
            start_index = self.tik_inst.Scalar("uint32")
            repeat_time.set_as(pad_vector_len // 128)
            start_index.set_as(0)

            with self.tik_inst.if_scope(repeat_time > 255):
                more_repeat_time = self.tik_inst.Scalar("uint32")
                more_repeat_time.set_as(repeat_time // 255)
                with self.tik_inst.for_range(0, more_repeat_time) as i:
                    self.tik_inst.vector_dup(128, pad_vector_ub[i * 255 * 128], pad_value, 255, 1, 8)
                start_index.set_as(more_repeat_time * 255 * 128)
            repeat_time.set_as(pad_vector_len - start_index)
            repeat_time.set_as(repeat_time // 128)

            with self.tik_inst.if_scope(repeat_time != 0):
                self.tik_inst.vector_dup(128, pad_vector_ub[start_index], pad_value, repeat_time, 1, 8)
                start_index.set_as(start_index + repeat_time * 128)

            last_data_len = self.tik_inst.Scalar("uint32")
            last_data_len.set_as(pad_vector_len - start_index)
            with self.tik_inst.if_scope(last_data_len != 0):
                self.tik_inst.vector_dup(last_data_len, pad_vector_ub[start_index], pad_value, 1, 1, 0)

    def _align_offline(self, value, to_align):
        return value if value % to_align == 0 else (value // to_align + 1) * to_align


def image_pad(input_dict, output_dict, paddings, pad_value, kernel_name="image_pad"):
    input_shape = input_dict["shape"]
    obj = ImagePad(input_shape, paddings, pad_value, kernel_name)
    obj.compute()
