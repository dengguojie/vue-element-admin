from te import tik


class PadChannel():
    def __init__(self, input0, output0, num_channels_to_pad, kernel_name='PadChannel'):
        self.input0 = input0
        self.output0 = output0
        self.input_shape = self.input0['shape']
        self.output_shape = self.output0['shape']
        self.input_n = self.input_shape[0]
        self.input_c1 = self.input_shape[1]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.output_c1 = self.output_shape[1]
        self.input_c = 16 * self.input_c1
        self.num_channels_to_pad = num_channels_to_pad
        self.kernel_name = kernel_name

    def compute(self):
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))

        # define input&output tensor
        input_data = tik_instance.Tensor("float16", (self.input_n, self.input_c1, self.input_h, self.input_w, 16),
                                         name="input_data",
                                         scope=tik.scope_gm)
        output_data = tik_instance.Tensor("float16", (self.input_n, self.output_c1, self.input_h, self.input_w, 16),
                                          name="output_data",
                                          scope=tik.scope_gm)

        # copy input data to ub
        data_ub = tik_instance.Tensor("float16", (1, self.input_c1, self.input_h, self.input_w, 16), name="data_ub",
                                      scope=tik.scope_ubuf)
        data_zero = tik_instance.Tensor("float16", (1, self.input_c1, self.input_h, self.input_w, 16), name="data_zero",
                                        scope=tik.scope_ubuf)
        with tik_instance.for_range(0, self.input_n) as pad_n:
            tik_instance.data_move(data_ub, input_data[pad_n, 0, 0, 0, 0], 0, 1,
                                   self.input_c1 * self.input_h * self.input_w, 0,
                                   0, 0)
            num = (self.input_c * self.input_h * self.input_w + 32639) // 32640
            with tik_instance.for_range(0, num) as cnt:
                tik_instance.vector_dup(128, data_zero[0, self.input_c1 * cnt // num, 0, 0, 0], 0.0,
                                        self.input_c1 * self.input_h * self.input_w // (8 * num), 1, 8, 0)
            tik_instance.data_move(output_data[pad_n, 0, 0, 0, 0], data_ub, 0, 1,
                                   self.input_c1 * self.input_h * self.input_w, 0,
                                   0, 0)
            tik_instance.data_move(output_data[pad_n, self.input_c1, 0, 0, 0], data_zero, 0, 1,
                                   self.input_c1 * self.input_h * self.input_w, 0,
                                   0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[input_data], outputs=[output_data], enable_l2=True)
        return tik_instance


def pad_channel(input0, output0, num_channels_to_pad, kernel_name='PadChannel'):
    padder = PadChannel(input0, output0, num_channels_to_pad, kernel_name)
    padder.compute()
