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
resize_d
"""

import math
from collections import namedtuple
from te import tik
from te.utils import para_check
from te import platform as cce

# get available ub size
UB_SIZE = cce.cce_conf.get_soc_spec(cce.cce_conf.UB_SIZE)


class ResizeBicubic:
    """ResizeBicubic main functions
    """
    # 'pylint: disable=too-many-arguments
    def __init__(self, x, sizes, scales, coordinate_transformation_mode, cubic_coeff_a, kernel_name="resize_d"):
        """init ResizeNearestNeighbor base parameters
        """
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
        self.x_shape = x.get("shape")
        self.x_dtype = x.get("dtype")
        self.check_param1(sizes, scales)
        self.sizes = sizes
        self.scales = scales
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.cubic_coeff_a = cubic_coeff_a
        self.kernel_name = kernel_name
        self.batch_size = self.x_shape[0]
        self.c1_size = self.x_shape[1]
        self.in_size_h = self.x_shape[2]
        self.in_size_w = self.x_shape[3]
        self.nc1 = self.batch_size * self.c1_size
        self.out_size_h = self.sizes[0]
        self.out_size_w = self.sizes[1]

        self.check_param2(self.in_size_h, self.in_size_w, self.out_size_h, self.out_size_w)
        output_shape = (self.batch_size, self.c1_size, self.out_size_h, self.out_size_w)

        self.in_num = self.batch_size * self.c1_size * self.in_size_h * self.in_size_w
        self.out_num = self.batch_size * self.c1_size * self.out_size_h * self.out_size_w

        block_bite_size = 32
        dtype_bytes_size = cce.cce_intrin.get_bit_len(self.x_dtype) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size
        self.ub_tensor_size = UB_SIZE // dtype_bytes_size // \
                              self.data_each_block * self.data_each_block

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, self.x_shape, name="x_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.x_dtype, output_shape, name="output_gm", scope=tik.scope_gm)

    # 'pylint: disable=too-many-locals
    # 'pylint: disable=too-many-statements
    def resize_bicubic_compute(self):
        """
        Bicubic main part.
        """
        # (N,M) to (N,M)
        if self.out_size_h == self.in_size_h and self.out_size_w == self.in_size_w:
            self.input_output_samesize()
        # (N,M) to (N1,M1)
        else:
            with self.tik_instance.for_range(0, self.out_size_h) as out_h_index:
                temp_scalar1 = self.tik_instance.Scalar(dtype="int32", init_value=out_h_index)
                out_h_index_scalar = self.tik_instance.Scalar(dtype="float32")
                self.tik_instance.scalar_conv('none', out_h_index_scalar, temp_scalar1)

                real_y = self.tik_instance.Scalar(dtype="float32")
                real_y.set_as(self.compute_real_y(out_h_index_scalar))

                temp_scalar2 = self.tik_instance.Scalar(dtype="float32", init_value=real_y)
                input_y = self.tik_instance.Scalar(dtype="int32")
                self.tik_instance.scalar_conv('floor', input_y, temp_scalar2)

                with self.tik_instance.for_range(0, self.out_size_w) as out_w_index:
                    temp_scalar3 = self.tik_instance.Scalar(dtype="int32", init_value=out_w_index)
                    out_w_index_scalar = self.tik_instance.Scalar(dtype="float32")
                    self.tik_instance.scalar_conv('none', out_w_index_scalar, temp_scalar3)

                    real_x = self.tik_instance.Scalar(dtype="float32")
                    real_x.set_as(self.compute_real_x(out_w_index_scalar))

                    temp_scalar4 = self.tik_instance.Scalar(dtype="float32", init_value=real_x)
                    input_x = self.tik_instance.Scalar(dtype="int32")
                    self.tik_instance.scalar_conv('floor', input_x, temp_scalar4)

                    with self.tik_instance.for_range(0, self.batch_size * self.c1_size) as c:
                        # get x 16 elements
                        coefficients1 = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                                 name="coefficients1",
                                                                 scope=tik.scope_ubuf)
                        coefficients2 = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                                 name="coefficients2",
                                                                 scope=tik.scope_ubuf)
                        coefficients3 = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                                 name="coefficients3",
                                                                 scope=tik.scope_ubuf)
                        coefficients4 = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                                 name="coefficients4",
                                                                 scope=tik.scope_ubuf)
                        temp = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                        name="temp",
                                                        scope=tik.scope_ubuf)

                        with self.tik_instance.for_range(0, 4) as i:
                            # compute move_offset
                            move_offset1 = self.upsample_get_index_bounded(input_x - 1, input_y - 1 + i)
                            move_offset2 = self.upsample_get_index_bounded(input_x, input_y - 1 + i)
                            move_offset3 = self.upsample_get_index_bounded(input_x + 1, input_y - 1 + i)
                            move_offset4 = self.upsample_get_index_bounded(input_x + 2, input_y - 1 + i)

                            x_ub = self.upsample_get_value_bounded(c, move_offset1, move_offset2, move_offset3,
                                                                   move_offset4)
                            with self.tik_instance.if_scope(i == 0):
                                self.tik_instance.data_move(
                                    coefficients1,
                                    self.cubic_interp1d(x_ub.x_ub1, x_ub.x_ub2, x_ub.x_ub3, x_ub.x_ub4,
                                                        out_w_index_scalar, input_x, self.in_size_w, self.out_size_w,
                                                        0), 0, 1, 1, 0, 0)
                            with self.tik_instance.if_scope(i == 1):
                                self.tik_instance.data_move(
                                    coefficients2,
                                    self.cubic_interp1d(x_ub.x_ub1, x_ub.x_ub2, x_ub.x_ub3, x_ub.x_ub4,
                                                        out_w_index_scalar, input_x, self.in_size_w, self.out_size_w,
                                                        0), 0, 1, 1, 0, 0)
                            with self.tik_instance.if_scope(i == 2):
                                self.tik_instance.data_move(
                                    coefficients3,
                                    self.cubic_interp1d(x_ub.x_ub1, x_ub.x_ub2, x_ub.x_ub3, x_ub.x_ub4,
                                                        out_w_index_scalar, input_x, self.in_size_w, self.out_size_w,
                                                        0), 0, 1, 1, 0, 0)
                            with self.tik_instance.if_scope(i == 3):
                                self.tik_instance.data_move(
                                    coefficients4,
                                    self.cubic_interp1d(x_ub.x_ub1, x_ub.x_ub2, x_ub.x_ub3, x_ub.x_ub4,
                                                        out_w_index_scalar, input_x, self.in_size_w, self.out_size_w,
                                                        0), 0, 1, 1, 0, 0)
                                self.tik_instance.data_move(
                                    temp,
                                    self.cubic_interp1d(coefficients1, coefficients2, coefficients3, coefficients4,
                                                        out_h_index_scalar, input_y, self.in_size_h, self.out_size_h,
                                                        1), 0, 1, 1, 0, 0)

                                chw = c * self.out_size_h * self.out_size_w
                                out_move_offset = chw + out_h_index * self.out_size_w + out_w_index
                                out_max_offset = self.out_num - self.data_each_block

                                data_out = self.compute_data_out(temp)
                                dst_data_out = self.tik_instance.Tensor("float16", (self.data_each_block, ),
                                                                        name="dst_data_out",
                                                                        scope=tik.scope_ubuf)

                                if self.x_dtype == "float16":
                                    self.tik_instance.vec_conv(self.data_each_block, "none", dst_data_out, data_out, 1,
                                                               1, 1)
                                    self.move_to_output_gm(dst_data_out, out_move_offset, out_max_offset)
                                else:
                                    self.move_to_output_gm(data_out, out_move_offset, out_max_offset)

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm], outputs=[self.output_gm])

        return self.tik_instance

    @staticmethod
    def check_param1(sizes, scales):
        """
        check size of sizes and scales

        :param sizes: Required attribute
        :param scales: Optional attribute
        """
        # check sizes
        if len(sizes) != 2:
            raise RuntimeError("It is expected output_size equals to 2, + \
                                but got output_size {}.".format(len(sizes)))

        # check scales
        if scales is not None and len(scales) != 2:
            raise RuntimeError("It is expected scales_size equals to 2, + \
                                but got scales_size {}.".format(len(scales)))

    @staticmethod
    def check_param2(in_size_h, in_size_w, out_size_h, out_size_w):
        """
        check in_size_h, in_size_w, out_size_h and out_size_w

        :param in_size_h: input H
        :param in_size_w: input W
        :param out_size_h: output H
        :param out_size_w: output W
        """
        if in_size_h <= 0 or in_size_w <= 0 or out_size_h <= 0 or out_size_w <= 0:
            raise RuntimeError("Input and output sizes should be greater than 0.")

    def input_output_samesize(self):
        """
        The input shape is the same as the output shape.
        """
        x_ub = self.tik_instance.Tensor(self.x_dtype, (self.ub_tensor_size, ), name="x_ub", scope=tik.scope_ubuf)
        loop_time = self.in_num // self.ub_tensor_size
        burst_len = math.ceil(self.ub_tensor_size / self.data_each_block)
        # self.in_num >= self.ub_tensor_size 63488
        if loop_time > 0:
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                offset = loop_index * self.ub_tensor_size
                self.tik_instance.data_move(x_ub, self.x_gm[offset], 0, 1, burst_len, 0, 0)
                self.tik_instance.data_move(self.output_gm[offset], x_ub, 0, 1, burst_len, 0, 0)
        offset = loop_time * self.ub_tensor_size
        last_num = self.in_num % self.ub_tensor_size
        # self.in_num < self.ub_tensor_size
        if last_num > 0:
            self.tik_instance.data_move(x_ub, self.x_gm[offset], 0, 1, math.ceil(last_num / self.data_each_block), 0, 0)
            self.tik_instance.data_move(self.output_gm[offset], x_ub, 0, 1, math.ceil(last_num / self.data_each_block),
                                        0, 0)

    def compute_real_x(self, out_w_index_scalar):
        """
        Calculate real_x (the x coordinate of a point in input)

        :param out_w_index_scalar: Cyclic variable, out_w_index.
        :return: real_x
        """
        real_x = self.tik_instance.Scalar(dtype="float32")
        out_w_index_scalar = self.tik_instance.Scalar(dtype="float32", init_value=out_w_index_scalar)
        in_size_w = self.tik_instance.Scalar(dtype="float32", init_value=self.in_size_w)
        out_size_w = self.tik_instance.Scalar(dtype="float32", init_value=self.out_size_w)
        one_scalar = self.tik_instance.Scalar(dtype="float32", init_value=1.0)
        if self.out_size_w > 1:
            if self.coordinate_transformation_mode == "align_corners":
                real_x.set_as((out_w_index_scalar * (in_size_w - one_scalar)) / (out_size_w - one_scalar))
            else:
                if self.scales[1] > 0:
                    real_x.set_as((out_w_index_scalar + 0.5) / self.scales[1] - 0.5)
                else:
                    real_x.set_as((out_w_index_scalar + 0.5) * self.in_size_w / self.out_size_w - 0.5)
        else:
            real_x.set_as(0.0)
        return real_x

    def compute_real_y(self, out_h_index_scalar):
        """
        Calculate real_y (the y coordinate of a point in input)

        :param out_h_index_scalar: Cyclic variable, out_h_index.
        :return: real_y
        """
        real_y = self.tik_instance.Scalar(dtype="float32")
        out_h_index_scalar = self.tik_instance.Scalar(dtype="float32", init_value=out_h_index_scalar)
        in_size_h = self.tik_instance.Scalar(dtype="float32", init_value=self.in_size_h)
        out_size_h = self.tik_instance.Scalar(dtype="float32", init_value=self.out_size_h)
        one_scalar = self.tik_instance.Scalar(dtype="float32", init_value=1.0)
        if self.out_size_h > 1:
            if self.coordinate_transformation_mode == "align_corners":
                real_y.set_as((out_h_index_scalar * (in_size_h - one_scalar)) / (out_size_h - one_scalar))
            else:
                if self.scales[0] > 0:
                    real_y.set_as((out_h_index_scalar + 0.5) / self.scales[0] - 0.5)
                else:
                    real_y.set_as((out_h_index_scalar + 0.5) * self.in_size_h / self.out_size_h - 0.5)
        else:
            real_y.set_as(0.0)
        return real_y

    def upsample_get_index_bounded(self, input_x, input_y):
        """
        Get the index of the point around (input_x, input_y)

        :param input_x: Integer part of real_x.
        :param input_y: Integer part of real_y.
        :return: index
        """
        access_x_scalar = self.tik_instance.Scalar(dtype="int64")
        access_y_scalar = self.tik_instance.Scalar(dtype="int64")

        x_max_left = self.tik_instance.Scalar(dtype="int64")
        y_max_left = self.tik_instance.Scalar(dtype="int64")
        zero = self.tik_instance.Scalar(dtype="int64", init_value=0)

        x_min_left = self.tik_instance.Scalar(dtype="int64", init_value=input_x)
        x_min_right = self.tik_instance.Scalar(dtype="int64", init_value=self.in_size_w - 1)
        self.tik_instance.scalar_min(x_max_left, x_min_left, x_min_right)
        self.tik_instance.scalar_max(access_x_scalar, x_max_left, zero)

        y_min_left = self.tik_instance.Scalar(dtype="int64", init_value=input_y)
        y_min_right = self.tik_instance.Scalar(dtype="int64", init_value=self.in_size_h - 1)
        self.tik_instance.scalar_min(y_max_left, y_min_left, y_min_right)
        self.tik_instance.scalar_max(access_y_scalar, y_max_left, zero)

        move_offset = self.tik_instance.Scalar(dtype="int64")
        move_offset.set_as(access_x_scalar + access_y_scalar * self.in_size_w)

        return move_offset

    # 'pylint: disable=too-many-locals
    # 'pylint: disable=too-many-statements
    def upsample_get_value_bounded(self, c, move_offset1, move_offset2, move_offset3, move_offset4):
        """
        Get the value of the point by move_offset.

        :param c: Cyclic variable c.
        :param move_offset1: The move_offset of input.
        :param move_offset2: The move_offset of input.
        :param move_offset3: The move_offset of input.
        :param move_offset4: The move_offset of input.
        :return: x_ub1, x_ub2, x_ub3 and x_ub4.
        """
        x_ub_tensor = self.tik_instance.Tensor(self.x_dtype, (self.data_each_block, ),
                                               name="x_ub_tensor",
                                               scope=tik.scope_ubuf)
        x_ub1 = self.tik_instance.Tensor(self.x_dtype, (self.data_each_block, ), name="x_ub1", scope=tik.scope_ubuf)
        x_ub2 = self.tik_instance.Tensor(self.x_dtype, (self.data_each_block, ), name="x_ub2", scope=tik.scope_ubuf)
        x_ub3 = self.tik_instance.Tensor(self.x_dtype, (self.data_each_block, ), name="x_ub3", scope=tik.scope_ubuf)
        x_ub4 = self.tik_instance.Tensor(self.x_dtype, (self.data_each_block, ), name="x_ub4", scope=tik.scope_ubuf)
        x_gm = self.x_gm.reshape((self.in_num, ))
        nc_move_offset = c * self.in_size_h * self.in_size_w

        # get x_ub1, x_ub2, x_ub3, x_ub4
        if self.in_num <= self.data_each_block:
            self.tik_instance.data_move(x_ub_tensor, x_gm, 0, 1, 1, 0, 0)
            x_ub1[0].set_as(x_ub_tensor[nc_move_offset + move_offset1])
            x_ub2[0].set_as(x_ub_tensor[nc_move_offset + move_offset2])
            x_ub3[0].set_as(x_ub_tensor[nc_move_offset + move_offset3])
            x_ub4[0].set_as(x_ub_tensor[nc_move_offset + move_offset4])
        else:
            max_offset = self.in_num - self.data_each_block
            # Consider the size of the max_offset
            with self.tik_instance.if_scope(nc_move_offset + move_offset1 < max_offset):
                self.tik_instance.data_move(x_ub1, x_gm[nc_move_offset + move_offset1], 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                relative = nc_move_offset + move_offset1 - max_offset
                self.tik_instance.data_move(x_ub_tensor, x_gm[max_offset], 0, 1, 1, 0, 0)
                x_ub1[0].set_as(x_ub_tensor[relative])

            # same as x_ub1
            with self.tik_instance.if_scope(nc_move_offset + move_offset2 < max_offset):
                self.tik_instance.data_move(x_ub2, x_gm[nc_move_offset + move_offset2], 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                relative = nc_move_offset + move_offset2 - max_offset
                self.tik_instance.data_move(x_ub_tensor, x_gm[max_offset], 0, 1, 1, 0, 0)
                x_ub2[0].set_as(x_ub_tensor[relative])

            # same as x_ub1
            with self.tik_instance.if_scope(nc_move_offset + move_offset3 < max_offset):
                self.tik_instance.data_move(x_ub3, x_gm[nc_move_offset + move_offset3], 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                relative = nc_move_offset + move_offset3 - max_offset
                self.tik_instance.data_move(x_ub_tensor, x_gm[max_offset], 0, 1, 1, 0, 0)
                x_ub3[0].set_as(x_ub_tensor[relative])

            # same as x_ub1
            with self.tik_instance.if_scope(nc_move_offset + move_offset4 < max_offset):
                self.tik_instance.data_move(x_ub4, x_gm[nc_move_offset + move_offset4], 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                relative = nc_move_offset + move_offset4 - max_offset
                self.tik_instance.data_move(x_ub_tensor, x_gm[max_offset], 0, 1, 1, 0, 0)
                x_ub4[0].set_as(x_ub_tensor[relative])

        X_ub = namedtuple('x_ub', 'x_ub1 x_ub2 x_ub3 x_ub4')

        return X_ub(x_ub1, x_ub2, x_ub3, x_ub4)

    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    # 'pylint: disable=too-many-statements
    def cubic_interp1d(self, x_a, x_b, x_c, x_d, index_scalar, input_index, in_length, out_length, flag):
        """
        Interpolation algorithm main part.

        :param x_a: x_a[0] is a piece of data in the input, or coefficients1.
        :param x_b: x_b[0] is a piece of data in the input, or coefficients2.
        :param x_c: x_c[0] is a piece of data in the input, or coefficients3.
        :param x_d: x_d[0] is a piece of data in the input, or coefficients4.
        :param index_scalar: Cyclic variable, out_w_index or out_h_index.
        :param input_index: The integer part of the real_x(real_y), real_x(real_y)
                            is obtained by compute_real_x()(compute_real_y()).
        :param in_length: input size of W or H.
        :param out_length: output size of W or H.
        :return: coefficients
        """
        coeffs1 = self.tik_instance.Scalar(dtype="float32")
        coeffs2 = self.tik_instance.Scalar(dtype="float32")
        coeffs3 = self.tik_instance.Scalar(dtype="float32")
        coeffs4 = self.tik_instance.Scalar(dtype="float32")

        a = self.cubic_coeff_a
        temp_scalar = self.tik_instance.Scalar(dtype="int32", init_value=input_index)
        cast_input_index = self.tik_instance.Scalar(dtype="float32")
        self.tik_instance.scalar_conv('none', cast_input_index, temp_scalar)

        input_index_scalar = self.tik_instance.Scalar(dtype="float32", init_value=cast_input_index)
        one_length = self.tik_instance.Scalar(dtype="float32")
        two_length = self.tik_instance.Scalar(dtype="float32")
        three_length = self.tik_instance.Scalar(dtype="float32")
        x1 = self.tik_instance.Scalar(dtype="float32")

        # Different values of scales in input and output.
        if flag == 0:
            scales = self.scales[1]
        else:
            scales = self.scales[0]

        # Define x1 in different situations, corresponds to compute_data_out function.
        if out_length > 1:
            if self.coordinate_transformation_mode == "align_corners":
                one_length.set_as(out_length - 1)
                two_length.set_as((out_length - 1) * (out_length - 1))
                three_length.set_as((out_length - 1) * (out_length - 1) * (out_length - 1))
                x1.set_as(index_scalar * (in_length - 1.0) - input_index_scalar * one_length)
            else:
                if scales > 0.0:
                    one_length.set_as(scales)
                    two_length.set_as(scales * scales)
                    three_length.set_as(scales * scales * scales)
                    x1.set_as((index_scalar + 0.5) - (0.5 + input_index_scalar) * scales)
                else:
                    one_length.set_as(out_length)
                    two_length.set_as(out_length * out_length)
                    three_length.set_as(out_length * out_length * out_length)
                    x1.set_as((index_scalar + 0.5) * in_length - (0.5 + input_index_scalar) * out_length)
            coeffs1.set_as(((a * (x1 + one_length) - 5.0 * a * one_length) * (x1 + one_length) + 8.0 * a * two_length) *
                           (x1 + one_length) - 4.0 * a * three_length)
            coeffs2.set_as(((a + 2.0) * x1 - (a + 3.0) * one_length) * x1 * x1 + 1.0 * three_length)
            x2 = one_length - x1
            coeffs3.set_as(((a + 2.0) * x2 - (a + 3.0) * one_length) * x2 * x2 + 1.0 * three_length)
            coeffs4.set_as(((a * (x2 + one_length) - 5.0 * a * one_length) * (x2 + one_length) + 8.0 * a * two_length) *
                           (x2 + one_length) - 4.0 * a * three_length)
        else:
            coeffs1.set_as(0.0)
            coeffs2.set_as(1.0)
            coeffs3.set_as(0.0)
            coeffs4.set_as(0.0)

        temp1 = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="temp1", scope=tik.scope_ubuf)
        temp2 = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="temp2", scope=tik.scope_ubuf)
        temp3 = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="temp3", scope=tik.scope_ubuf)
        temp4 = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="temp4", scope=tik.scope_ubuf)
        temp5 = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="temp5", scope=tik.scope_ubuf)
        temp6 = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="temp6", scope=tik.scope_ubuf)
        coefficients = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                                name="coefficients",
                                                scope=tik.scope_ubuf)

        dst_x_a = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="dst_x_a", scope=tik.scope_ubuf)
        dst_x_b = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="dst_x_b", scope=tik.scope_ubuf)
        dst_x_c = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="dst_x_c", scope=tik.scope_ubuf)
        dst_x_d = self.tik_instance.Tensor("float32", (self.data_each_block, ), name="dst_x_d", scope=tik.scope_ubuf)

        if x_a.dtype == "float16":
            self.tik_instance.vec_conv(self.data_each_block, "none", dst_x_a, x_a, 1, 1, 1)
            self.tik_instance.vec_conv(self.data_each_block, "none", dst_x_b, x_b, 1, 1, 1)
            self.tik_instance.vec_conv(self.data_each_block, "none", dst_x_c, x_c, 1, 1, 1)
            self.tik_instance.vec_conv(self.data_each_block, "none", dst_x_d, x_d, 1, 1, 1)
        else:
            self.tik_instance.data_move(dst_x_a, x_a, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(dst_x_b, x_b, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(dst_x_c, x_c, 0, 1, 1, 0, 0)
            self.tik_instance.data_move(dst_x_d, x_d, 0, 1, 1, 0, 0)

        # coeffs1 * x_a + coeffs2 * x_b + coeffs3 * x_c + coeffs4 * x_d
        self.tik_instance.vec_muls(1, temp1, dst_x_a, coeffs1, 1, 1, 1)
        self.tik_instance.vec_muls(1, temp2, dst_x_b, coeffs2, 1, 1, 1)
        self.tik_instance.vec_muls(1, temp3, dst_x_c, coeffs3, 1, 1, 1)
        self.tik_instance.vec_muls(1, temp4, dst_x_d, coeffs4, 1, 1, 1)
        self.tik_instance.vec_add(1, temp5, temp1, temp2, 1, 1, 1, 1)
        self.tik_instance.vec_add(1, temp6, temp3, temp4, 1, 1, 1, 1)
        self.tik_instance.vec_add(1, coefficients, temp5, temp6, 1, 1, 1, 1)

        return coefficients

    def compute_data_out(self, temp):
        """
        Calculate data_out: divide the temp by (a*a*a and b*b*b)

        :param temp: A tenor to be processed
        :return: data_out, final result
        """
        in_three_length = self.tik_instance.Scalar(dtype="float32")
        out_three_length = self.tik_instance.Scalar(dtype="float32")

        # Define a*a*a and b*b*b in different situations,
        # same as cubic_interp1d function
        if self.out_size_w > 1:
            if self.coordinate_transformation_mode == "align_corners":
                in_three_length.set_as((self.out_size_w - 1) * (self.out_size_w - 1) * (self.out_size_w - 1))
            else:
                if self.scales[1] > 0.0:
                    in_three_length.set_as(self.scales[1] * self.scales[1] * self.scales[1])
                else:
                    in_three_length.set_as(self.out_size_w * self.out_size_w * self.out_size_w)
        else:
            in_three_length.set_as(1)

        if self.out_size_h > 1:
            if self.coordinate_transformation_mode == "align_corners":
                out_three_length.set_as((self.out_size_h - 1) * (self.out_size_h - 1) * (self.out_size_h - 1))
            else:
                if self.scales[0] > 0.0:
                    out_three_length.set_as(self.scales[0] * self.scales[0] * self.scales[0])
                else:
                    out_three_length.set_as(self.out_size_h * self.out_size_h * self.out_size_h)
        else:
            out_three_length.set_as(1)

        data_out1 = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                             name="data_out1",
                                             scope=tik.scope_ubuf)
        data_out2 = self.tik_instance.Tensor("float32", (self.data_each_block, ),
                                             name="data_out2",
                                             scope=tik.scope_ubuf)
        out_length_scalar1 = self.tik_instance.Scalar(dtype="float32")
        out_length_scalar1.set_as(1.0 / in_three_length)
        self.tik_instance.vec_muls(1, data_out1, temp, out_length_scalar1, 1, 1, 1)

        out_length_scalar2 = self.tik_instance.Scalar(dtype="float32")
        out_length_scalar2.set_as(1.0 / out_three_length)
        self.tik_instance.vec_muls(1, data_out2, data_out1, out_length_scalar2, 1, 1, 1)

        return data_out2

    def move_to_output_gm(self, data_out, out_move_offset, out_max_offset):
        """
        Data_out moves to output_gm.

        :param data_out: Results to be moved.
        :param out_move_offset: move offset
        :param out_max_offset: equals to (self.out_num - self.data_each_block)
        :return:
        """
        temp_ub = self.tik_instance.Tensor(self.x_dtype, (self.data_each_block, ), name="temp_ub", scope=tik.scope_ubuf)
        if self.out_num <= self.data_each_block:
            self.tik_instance.data_move(temp_ub, self.output_gm, 0, 1, 1, 0, 0)
            temp_ub[out_move_offset].set_as(data_out[0])
            self.tik_instance.data_move(self.output_gm, temp_ub, 0, 1, 1, 0, 0)
        else:
            with self.tik_instance.if_scope(out_move_offset < out_max_offset):
                self.tik_instance.data_move(temp_ub, self.output_gm[out_move_offset], 0, 1, 1, 0, 0)
                temp_ub[0].set_as(data_out[0])
                self.tik_instance.data_move(self.output_gm[out_move_offset], temp_ub, 0, 1, 1, 0, 0)
            with self.tik_instance.else_scope():
                relative = out_move_offset - out_max_offset
                self.tik_instance.data_move(temp_ub, self.output_gm[out_max_offset], 0, 1, 1, 0, 0)
                temp_ub[relative].set_as(data_out[0])
                self.tik_instance.data_move(self.output_gm[out_max_offset], temp_ub, 0, 1, 1, 0, 0)


class ResizeLinear:
    """
    ResizeLinear main functions
    """
    # 'pylint: disable=too-many-arguments
    # 'pylint: disable=too-many-locals
    def __init__(self, x, sizes, scales, coordinate_transformation_mode="align_corners", kernel_name="resize_d"):

        self.tik_instance = tik.Tik()

        self.x_dtype = x.get("dtype")
        self.x_shape = x.get("shape")
        self.size = sizes[0]
        self.scale = scales[0]
        self.dim0 = self.x_shape[0]
        self.dim1 = self.x_shape[1]
        self.dim_redundancy = self.x_shape[2]
        self.dim2 = self.x_shape[3]
        self.input_num = self.dim0 * self.dim1 * self.dim2
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.kernel_name = kernel_name

        self.check_param1(self.dim_redundancy, self.dim2, self.size)
        self.check_param2(sizes, scales)

        if self.coordinate_transformation_mode == "align_corners":
            self.scale_w = 0. if self.size == 1 else (self.dim2 - 1) / (self.size - 1)
        else:
            self.scale_w = 1.0 / self.scale if self.scale > 0. else (self.dim2 / self.size)

        self.output_num = self.dim0 * self.dim1 * self.size

        block_bite_size = 32
        dtype_bytes_size = cce.cce_intrin.get_bit_len(self.x_dtype) // 8
        self.data_each_block = block_bite_size // dtype_bytes_size

        self.x_gm = self.tik_instance.Tensor(self.x_dtype, (self.dim0, self.dim1, 1, self.dim2),
                                             name="x_gm",
                                             scope=tik.scope_gm)
        self.x_gm.reshape(self.x_shape)

        self.output_gm = self.tik_instance.Tensor(self.x_dtype, (self.dim0, self.dim1, 1, self.size),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)

    # 'pylint: disable=too-many-locals, too-many-branches
    def resize_linear_compute(self):
        """
        ResizeLinear main logic
        """
        self.x_gm.reshape([
            self.input_num,
        ])

        if self.output_num <= self.data_each_block:
            res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                      name="res_lastdim_ub",
                                                      scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.dim0) as i:
                with self.tik_instance.for_range(0, self.dim1) as j:
                    current_index_output = self.tik_instance.Scalar("int32",
                                                                    init_value=i * (self.dim1 * self.size) +
                                                                    j * self.size)
                    with self.tik_instance.for_range(0, self.size) as k:
                        with self.tik_instance.if_scope(self.size == 1):
                            res_lastdim_ub[current_index_output + k].set_as(
                                self.get_number_in_global_memory(i * (self.dim1 * self.dim2) + j * self.dim2))
                        with self.tik_instance.else_scope():
                            res_lastdim_ub[current_index_output + k].set_as(
                                self.compute_helper(self.scale_w, k, i * (self.dim1 * self.dim2) + j * self.dim2))

            self.tik_instance.data_move(self.output_gm, res_lastdim_ub, 0, 1, 1, 0, 0)

        elif self.size < self.data_each_block:
            loop_time = self.output_num // self.data_each_block
            with self.tik_instance.for_range(0, loop_time) as i:
                res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                          name="res_lastdim_ub",
                                                          scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.data_each_block) as j:
                    current_index = i * self.data_each_block + j
                    current_dim1 = current_index // self.size
                    with self.tik_instance.if_scope(self.size == 1):
                        res_lastdim_ub[j].set_as(
                            self.get_number_in_global_memory(current_dim1 * self.dim2))
                    with self.tik_instance.else_scope():
                        res_lastdim_ub[j].set_as(
                            self.compute_helper(self.scale_w, current_index % self.size, current_dim1 * self.dim2))
                self.tik_instance.data_move(self.output_gm[i * self.data_each_block], res_lastdim_ub, 0, 1, 1, 0, 0)

            remainder = self.output_num % self.data_each_block
            with self.tik_instance.if_scope(remainder != 0):
                remainder_begin_index = self.output_num - self.data_each_block
                res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                          name="res_lastdim_ub",
                                                          scope=tik.scope_ubuf)
                with self.tik_instance.for_range(0, self.data_each_block) as k:
                    current_index = remainder_begin_index + k
                    current_dim1 = current_index // self.size
                    with self.tik_instance.if_scope(self.size == 1):
                        res_lastdim_ub[k].set_as(
                            self.get_number_in_global_memory(current_dim1 * self.dim2))
                    with self.tik_instance.else_scope():
                        res_lastdim_ub[k].set_as(
                            self.compute_helper(self.scale_w, current_index % self.size, current_dim1 * self.dim2))
                self.tik_instance.data_move(self.output_gm[remainder_begin_index], res_lastdim_ub, 0, 1, 1, 0, 0)

        else:
            with self.tik_instance.for_range(0, self.dim0) as i:
                with self.tik_instance.for_range(0, self.dim1) as j:
                    loop_time = self.tik_instance.Scalar("int32", init_value=self.size // self.data_each_block)
                    current_index_output = self.tik_instance.Scalar("int32",
                                                                    init_value=i * (self.dim1 * self.size) +
                                                                    j * self.size)
                    with self.tik_instance.for_range(0, loop_time) as m:
                        res_lastdim_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                                  name="res_lastdim_ub",
                                                                  scope=tik.scope_ubuf)

                        with self.tik_instance.for_range(0, self.data_each_block) as n:
                            with self.tik_instance.if_scope(self.size == 1):
                                res_lastdim_ub[n].set_as(
                                    self.get_number_in_global_memory(i * (self.dim1 * self.dim2) + j * self.dim2))
                            with self.tik_instance.else_scope():
                                res_lastdim_ub[n].set_as(
                                    self.compute_helper(self.scale_w, m * self.data_each_block + n,
                                                        i * (self.dim1 * self.dim2) + j * self.dim2))
                        self.tik_instance.data_move(self.output_gm[current_index_output], res_lastdim_ub, 0, 1, 1, 0, 0)
                        current_index_output.set_as(current_index_output + self.data_each_block)
                    res_lastdim_remainder_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block],
                                                                        name="res_lastdim_remainder_ub",
                                                                        scope=tik.scope_ubuf)

                    remainder = self.size % self.data_each_block
                    with self.tik_instance.if_scope(remainder != 0):
                        remainder_begin_index = self.size - self.data_each_block
                        with self.tik_instance.for_range(0, self.data_each_block) as k:
                            with self.tik_instance.if_scope(self.size == 1):
                                res_lastdim_remainder_ub[k].set_as(
                                    self.get_number_in_global_memory(i * self.dim1 * self.dim2 + j * self.dim2))
                            with self.tik_instance.else_scope():
                                res_lastdim_remainder_ub[k].set_as(
                                    self.compute_helper(self.scale_w, remainder_begin_index + k,
                                                        i * (self.dim1 * self.dim2) + j * self.dim2))
                        self.tik_instance.data_move(
                            self.output_gm[i * (self.dim1 * self.size) + (j + 1) * self.size - self.data_each_block],
                            res_lastdim_remainder_ub, 0, 1, 1, 0, 0)

        self.output_gm.reshape([self.dim0, self.dim1, 1, self.size])

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.x_gm], outputs=[self.output_gm])
        return self.tik_instance

    def get_number_in_global_memory(self, index):
        """
        get the value with given index from input tensor (in global memory)

        Parameters
        ----------
        index : int
            the index of required value in the input tensor

        Returns
        -------
        res : input.dtype
            the value under the given index
        """
        max_offset = max(0, self.input_num - self.data_each_block)

        x_ub = self.tik_instance.Tensor(self.x_dtype, [self.data_each_block], name="x_ub", scope=tik.scope_ubuf)

        res = self.tik_instance.Scalar(self.x_dtype, name="res")

        index = self.tik_instance.Scalar("int32", init_value=index)

        with self.tik_instance.if_scope(index < max_offset):
            self.tik_instance.data_move(x_ub, self.x_gm[index], 0, 1, 1, 0, 0)
            res.set_as(x_ub[0])

        with self.tik_instance.else_scope():
            self.tik_instance.data_move(x_ub, self.x_gm[max_offset], 0, 1, 1, 0, 0)
            res.set_as(x_ub[index - max_offset])

        return res

    def compute_helper(self, scale_w, output_block_offset, input_dim_offset):
        """
        ResizeLinear main calculation logic

        Parameters
        ----------
        scale_w : float

        output_block_offset : int

        input_dim_offset : int

        Returns
        -------
        res : input.dtype
            the output value with the given parameters

        """
        # Cal real
        real_w = self.tik_instance.Scalar("float32", name="real_w")
        k = self.tik_instance.Scalar("float32", init_value=output_block_offset)
        temp_w = self.tik_instance.Scalar("float32")
        with self.tik_instance.if_scope(self.coordinate_transformation_mode == "align_corners"):
            temp_w.set_as(scale_w * k)
        with self.tik_instance.else_scope():
            temp = self.tik_instance.Scalar(dtype="float32", init_value=scale_w * (k + 0.5) - 0.5)
            with self.tik_instance.if_scope(temp < 0):
                temp_w.set_as(0.)
            with self.tik_instance.else_scope():
                temp_w.set_as(temp)
        real_w.set_as(temp_w)

        # Cal Integer of real_w
        coefficient_w = self.tik_instance.Scalar("int32", name="coefficient_w")
        self.tik_instance.scalar_conv('floor', coefficient_w, real_w)

        # Cal Decimal of real_w
        coefficient_lambda = self.tik_instance.Scalar("float32", name="coefficient_lambda")
        coefficient_lambda.set_as(real_w - coefficient_w)

        # Cal 1.0 - Decimal of real_w
        coefficient_lambda0 = self.tik_instance.Scalar("float32", name="coefficient_lambda0")
        coefficient_lambda0.set_as(1.0 - coefficient_lambda)

        index = self.tik_instance.Scalar("int32", init_value=input_dim_offset + coefficient_w)
        temp2 = self.tik_instance.Scalar(self.x_dtype, init_value=self.get_number_in_global_memory(index))

        offset = self.tik_instance.Scalar(dtype="int32", init_value=1)
        with self.tik_instance.if_scope(coefficient_w == (self.dim2 - 1)):
            offset.set_as(0)

        temp4 = self.tik_instance.Scalar(self.x_dtype, init_value=self.get_number_in_global_memory(offset + index))

        res = self.tik_instance.Scalar(dtype=self.x_dtype,
                                       init_value=(coefficient_lambda0 * temp2 + coefficient_lambda * temp4))

        return res

    @staticmethod
    def check_param1(dim_redundancy, in_size_w, out_size_w):
        """
        check  in_size_w, out_size_w:
        in_size_w and out_size_w should be greater than 0

        Parameters
        ----------
        in_size_w : int
            the last dim of input
        out_size_w : int
            the output size

        Returns
        -------
        None
        """
        # Since only NCHW format input is currently supported, the input of npu
        # is converted from 3dim to 4dim, so the relevant judgment has also been changed(if dim_redundancy != 1)
        if dim_redundancy != 1:
            raise RuntimeError("The 3rd Dim of Input Tensor should always be 1.")

        if in_size_w <= 0 or out_size_w <= 0:
            raise RuntimeError("Input and output sizes should be greater than 0.")

    def check_param2(self, sizes, scales):
        """
        check sizes, scales:
        the length of sizes and scales should both be 1,
        the value of the scales should equal to x.shape[2] / sizes[0].

        Parameters
        ----------
        sizes : list
            list with values of sizes
        scales : list
            list with values of scales

        Returns
        -------
        None
        """
        # check sizes
        if len(sizes) != 1:
            raise RuntimeError("It is expected len(sizes) equals to 1.")

        # check scales
        if len(scales) != 1 and scales is not None:
            raise RuntimeError("It is expected len(scales) equals to 1.")

        #check scales value
        if scales is not None and (sizes[0] / self.dim2 - scales[0]) > 0.0001:
            raise RuntimeError("It is expected scales[0] equals to x.shape[2] / sizes[0].")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_FLOAT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=W0613
def resize_d(x,
             y,
             sizes,
             scales=None,
             roi=None,
             coordinate_transformation_mode="half_pixel",
             cubic_coeff_a=-0.75,
             exclude_outside=0,
             extrapolation_value=0.0,
             mode="nearest",
             nearest_mode="round_prefer_floor",
             kernel_name="resize_d"):
    """
    algorithm: resize_d
    Operation for resize_d

    Parameters
    ----------
    x : dict
        dict with keys(shape and dtype) of x
    y : dict
        dict with keys(shape and dtype) of y
    sizes : list
        the shape of output about 'new_height, new_width'
    scales : list
        the value about 'scale_h, scale_w'
    roi: list
        The RoIs' coordinates are normalized in the coordinate system of the input image.
        It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    coordinate_transformation_mode : str
        This attribute describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
    cubic_coeff_a : float
        The coefficient 'a' used in cubic interpolation.
    exclude_outside : int
        If set to 1, the weight of sampling locations outside the tensor will be set to 0
        and the weight will be renormalized so that their sum is 1.0.
    extrapolation_value : float
        When coordinate_transformation_mode is "tf_crop_and_resize" and
        x_original is outside the range [0, length_original - 1],
        this value is used as the corresponding output value. Default is 0.0f.
    mode : str
        Three interpolation modes: nearest (default), linear and cubic.
    nearest_mode : str
        Four modes: round_prefer_floor (default, as known as round half down),
        round_prefer_ceil (as known as round half up), floor, ceil.
        Only used by nearest interpolation.
    kernel_name : str
        kernel name, default value is "resize_d"

    Returns
    -------
    None
    """
    res = None
    x_dim = len(x.get("shape"))
    if mode == "cubic" and x_dim == 4:
        resize_bicubic_instance = ResizeBicubic(x, sizes, scales, coordinate_transformation_mode, cubic_coeff_a,
                                                kernel_name)
        res = resize_bicubic_instance.resize_bicubic_compute()
    elif mode == "linear" and x_dim == 4:
        resize_linear = ResizeLinear(x, sizes, scales, coordinate_transformation_mode, kernel_name)
        res = resize_linear.resize_linear_compute()
    else:
        raise RuntimeError("Not supported at the moment.")
    return res
