# coding: utf-8

from te import tik

MIN_NUM = 1e-10


class L2Normalize:

    def __init__(self, input_data_gm, output_data_gm, eps, kernel_name="L2Normalize"):
        """
        Data normalization
        :param input_data_gm: input data
        :param output_data_gm: normalized data
        :param eps: A minimum value replaces 0
        :param kernel_name: name of kernel
        :return: NA
        """
        self.input_data_gm = input_data_gm
        self.output_data_gm = output_data_gm
        self.eps = eps
        self.kernel_name = kernel_name
        self.input_data_shape = input_data_gm.get("shape")
        self.output_data_shape = output_data_gm.get("shape")

        self._input_checkout()
        self._output_checkout()

        self.shape_input_ub = (1, 32, 1, 1, 16)
        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.input_data_gm = self.tik_instance.Tensor(
            "float16", self.input_data_shape,
            name="input_data_gm", scope=tik.scope_gm
        )
        self.output_data_gm = self.tik_instance.Tensor(
            "float16", self.output_data_shape,
            name="output_data_gm", scope=tik.scope_gm
        )

    def _input_checkout(self):
        if self.input_data_shape[1:] != (32, 1, 1, 16):
            raise RuntimeError("input shape does not match")
        if self.input_data_gm.get("dtype") != "float16":
            raise RuntimeError("input type does not match")

    def _output_checkout(self):
        if self.output_data_shape != self.input_data_shape:
            raise RuntimeError("output shape does not match input shape")
        if self.output_data_gm.get("dtype") != "float16":
            raise RuntimeError("output type does not match")

    def mode_compute(self):
        with self.tik_instance.for_range(0, self.input_data_shape[0], block_num=self.input_data_shape[0]) as cnt:
            # define ub tensor
            data_ub_fp16 = self.tik_instance.Tensor(
                "float16", self.shape_input_ub, name="data_ub_fp16", scope=tik.scope_ubuf
            )
            data_ub_fp32 = self.tik_instance.Tensor(
                "float32", self.shape_input_ub, name="data_ub_fp32", scope=tik.scope_ubuf
            )
            tensor_temp_1 = self.tik_instance.Tensor(
                "float32", self.shape_input_ub, name="buffer_ub", scope=tik.scope_ubuf
            )
            scalar_temp_1 = self.tik_instance.Scalar(dtype="float32")
            block_fp16 = self.input_data_shape[1] * self.input_data_shape[2] * self.input_data_shape[3]
            repeat_fp32 = self.input_data_shape[1] * self.input_data_shape[2] * self.input_data_shape[3] // 4
            self.tik_instance.data_move(data_ub_fp16, self.input_data_gm[cnt, 0, 0, 0, 0], 0, 1, block_fp16, 0, 0, 0)
            self.tik_instance.vconv(64, '', data_ub_fp32, data_ub_fp16, repeat_fp32, 1, 1, 8, 4)
            buffer_ub = self._count_l2normalize(data_ub_fp32, data_ub_fp32, repeat_fp32, tensor_temp_1, scalar_temp_1)
            self.tik_instance.vconv(64, '', data_ub_fp16, buffer_ub, repeat_fp32, 1, 1, 4, 8)
            self.tik_instance.data_move(self.output_data_gm[cnt, 0, 0, 0, 0], data_ub_fp16, 0, 1, block_fp16, 0, 0)

        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name, inputs=[self.input_data_gm], outputs=[self.output_data_gm], enable_l2=True
        )

    def _count_l2normalize(self, result, input_ub_32, repeat, tensor_temp_1, scalar_temp_1):
        self.tik_instance.vmul(64, tensor_temp_1, input_ub_32, input_ub_32, repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vcadd(64, tensor_temp_1, tensor_temp_1, repeat, 1, 1, 8)
        self.tik_instance.vcadd(8, tensor_temp_1, tensor_temp_1, 1, 1, 1, 8)
        self.tik_instance.vadds(1, tensor_temp_1, tensor_temp_1, MIN_NUM, 1, 1, 1, 8, 8)
        self.tik_instance.vrsqrt(1, tensor_temp_1, tensor_temp_1, 1, 1, 1, 8, 8)

        scalar_temp_1.set_as(tensor_temp_1[0])
        self.tik_instance.vmuls(64, result, input_ub_32, scalar_temp_1, repeat, 1, 1, 8, 8)
        return result


def l2_normalize(input_data_gm, output_data_gm, eps=MIN_NUM, kernel_name="L2Normalize"):
    obj = L2Normalize(input_data_gm, output_data_gm, eps, kernel_name)
    obj.mode_compute()
