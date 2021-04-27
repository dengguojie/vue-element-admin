# coding: utf-8

from te import tik


class BatchMatmul:

    def __init__(self, input_data_left, input_data_right, output_data, kernel_name="BatchMatmul"):
        """
        Batch matrix multiplication
        @param input_data_left: left matrix
        @param input_data_right: right matrix
        @param output_data: output data
        @param kernel_name: name of kernel
        :return:NA
        """
        self.input_data_left = input_data_left
        self.input_data_right = input_data_right
        self.output_data = output_data
        self.kernel_name = kernel_name

        self.input_data_left_shape = input_data_left.get("shape")
        self.input_data_right_shape = input_data_right.get("shape")
        self.output_data_shape = output_data.get("shape")

        self._input_checkout()
        self._output_checkout()

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.input_data_left = self.tik_instance.Tensor(
            "float16", self.input_data_left_shape, name="input_data_left", scope=tik.scope_gm
        )
        self.input_data_right = self.tik_instance.Tensor(
            "float16", self.input_data_right_shape, name="input_data_right", scope=tik.scope_gm
        )
        self.output_data = self.tik_instance.Tensor(
            "float16", self.output_data_shape, name="output_data", scope=tik.scope_gm
        )

    def _input_checkout(self):
        if len(self.input_data_left_shape) != 3 or \
                len(self.input_data_right_shape) != 3:
            raise RuntimeError("input shape does not match")
        if self.input_data_left.get("dtype") != "float16" or \
                self.input_data_right.get("dtype") != "float16":
            raise RuntimeError("input type does not match")
        if self.input_data_left_shape[2] != self.input_data_right_shape[1] or \
                self.input_data_left_shape[0] != self.input_data_right_shape[0]:
            raise RuntimeError("The left and right matrices cannot be multiplied")

    def _output_checkout(self):
        output_data_shape = (self.input_data_left_shape[0],
                             self.input_data_left_shape[1],
                             self.input_data_right_shape[2])
        if self.output_data_shape != output_data_shape:
            raise RuntimeError("output shape does not match")
        if self.output_data.get("dtype") != "float16":
            raise RuntimeError("output type does not match")

    def _get_times(self):
        input_c1, input_h1, input_w1 = self.input_data_left_shape
        input_c2, input_h2, input_w2 = self.input_data_right_shape
        times = ((input_c1 * input_w1 * 2 + input_c2 * input_w2 * 4) + 240 * 512 - 1) // (240 * 512)
        while input_c1 % times != 0:
            times += 1
        return times

    def _transposition(self, dst, src, times, dime_1, dime_2):
        repeat_times = dime_2 // (16 * times)
        dst_rep = 0 if repeat_times == 1 else 1
        src_rep = 0 if repeat_times == 1 else dime_1
        with self.tik_instance.for_range(0, dime_1 // 16) as tmp_w1:
            src_list = [src[k, tmp_w1 * 16] for k in range(16)]
            dst_list = [dst[k + tmp_w1 * 16, 0] for k in range(16)]
            self.tik_instance.vnchwconv(True, True, dst_list, src_list, repeat_times, dst_rep, src_rep)
        return dst

    def batchmatmul_compute(self):
        times = self._get_times()
        with self.tik_instance.for_range(0, times) as tmp_times:
            with self.tik_instance.for_range(0, self.input_data_left_shape[1]) as tmp_h1:
                self._compute_each_loop(times, tmp_times, tmp_h1)
        self.tik_instance.BuildCCE(
            inputs=[self.input_data_left, self.input_data_right],
            outputs=[self.output_data], kernel_name=self.kernel_name
        )

    def _compute_each_loop(self, times, tmp_times, tmp_h1):
        input_c1, input_h1, input_w1 = self.input_data_left_shape
        input_c2, input_h2, input_w2 = self.input_data_right_shape
        c_loop = input_c1 // times
        left_ub = self.tik_instance.Tensor("float16", (c_loop, input_w1), name="left_ub", scope=tik.scope_ubuf)
        right_ub = self.tik_instance.Tensor("float16", (c_loop, input_w2), name="right_ub", scope=tik.scope_ubuf)
        left_ub_trans = self.tik_instance.Tensor("float16", (input_w1, c_loop),
                                                 name="left_ub_trans", scope=tik.scope_ubuf)
        right_ub_trans = self.tik_instance.Tensor("float16", (input_w2, c_loop),
                                                  name="right_ub_trans", scope=tik.scope_ubuf)
        data_ub_ans = self.tik_instance.Tensor("float16", (input_w2, c_loop), name="data_ub_ans", scope=tik.scope_ubuf)
        data_ub_ans_ftrans = self.tik_instance.Tensor("float16", (c_loop, input_w2),
                                                      name="data_ub_ans_ftrans", scope=tik.scope_ubuf)

        self.tik_instance.data_move(
            left_ub, self.input_data_left[input_c1 * tmp_times // times, tmp_h1, 0], 0,
            input_c1 // times, input_w1 // 16, (input_h1 - 1) * input_w1 // 16, 0)
        left_ub_trans = self._transposition(left_ub_trans, left_ub, times, input_w1, input_c1)
        # init ans as 0
        self.tik_instance.vector_dup(128, data_ub_ans, 0.0, input_w2 * input_c1 // (128 * times), 1, 8, 0)
        with self.tik_instance.for_range(0, input_w1) as loop_w1:
            self.tik_instance.data_move(
                right_ub, self.input_data_right[input_c2 * tmp_times // times, loop_w1, 0], 0,
                input_c2 // times, input_w2 // 16, (input_h2 - 1) * input_w2 // 16, 0)
            right_ub_trans = self._transposition(right_ub_trans, right_ub, times, input_w2, input_c2)
            with self.tik_instance.for_range(0, input_w2) as tmp:
                if input_c1 // times <= 128:
                    self.tik_instance.vmla(
                        input_c1 // times, data_ub_ans[tmp, 0], left_ub_trans[loop_w1, 0], right_ub_trans[tmp, 0],
                        1, 1, 1, 1, input_c1 // (16 * times), input_c1 // (16 * times), input_c1 // (16 * times))
                else:
                    self.tik_instance.vmla(
                        128, data_ub_ans[tmp, 0], left_ub_trans[loop_w1, 0], right_ub_trans[tmp, 0],
                        input_c1 // (times * 128), 1, 1, 1, 8, 8, 8)
                    if (input_c1 // times) % 128 != 0:
                        block_num = (input_c1 // times) % 128
                        self.tik_instance.vmla(
                            block_num, data_ub_ans[tmp, input_c1 // times - block_num],
                            left_ub_trans[loop_w1, input_c1 // times - block_num],
                            right_ub_trans[tmp, input_c1 // times - block_num],
                            1, 1, 1, 1, block_num // 16, block_num // 16, block_num // 16)
        # trans answer to CHW
        with self.tik_instance.for_range(0, input_w2 // 16) as j:
            src_list = [data_ub_ans[j * 16 + k, 0] for k in range(16)]
            dst_list = [data_ub_ans_ftrans[k, j * 16] for k in range(16)]
            self.tik_instance.vnchwconv(True, True, dst_list, src_list, input_c2 // (16 * times), input_w2, 1)
        self.tik_instance.data_move(
            self.output_data[input_c1 * tmp_times // times, tmp_h1, 0],
            data_ub_ans_ftrans, 0, input_c2 // times, input_w2 // 16, 0, (input_h1 - 1) * input_w2 // 16
        )


def batch_matmul(input_data_left, input_data_right,
                 output_data, kernel_name="BatchMatmul"):
    obj = BatchMatmul(
        input_data_left, input_data_right, output_data, kernel_name
    )
    obj.batchmatmul_compute()
