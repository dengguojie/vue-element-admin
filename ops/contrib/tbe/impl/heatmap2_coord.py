from te import tik
import math
import numpy as np

UB_BUFF_MAX = 240 * 1024
FP16_WORDLEN = 2
INT_MAX = 2 * 1024 * 1024 * 1024
MAX_STEP_HW = 2048
MAX_STEP_NC1 = 255


class HeatMapToCoord:
    def global_init(self):
        pass

    def aicore_in_use_select(self, length):
        self.xlen_each_core = (length + self.aicore_use - 1) // self.aicore_use
        self.xlen_last_core = length - self.xlen_each_core * (self.aicore_use - 1)
        self.aicore_use = (length + self.xlen_each_core - 1) // self.xlen_each_core
        if (self.aicore_use == 1):
            self.xlen_last_core = self.xlen_each_core
        print("self.xlen_each_core:", self.xlen_each_core, "self.xlen_last_core:", self.xlen_last_core)

    def __init__(self, shape, kern_name):
        self.kern_name = kern_name
        self.n = shape[0]
        self.c = shape[1] * shape[4]
        self.h = shape[2]
        self.w = shape[3]
        self.c1 = shape[1]
        self.nc1 = self.n * self.c1
        self.hw = self.h * self.w

        # if coord bigger than 2048, precision loss will be caused for fp16
        if (self.h > 2048 or self.w > 2048):
            raise RuntimeError("h and w can not bigger than 2048")

        if (self.nc1 * self.h * self.w * 16 * FP16_WORDLEN > INT_MAX):
            raise RuntimeError("data mem should not bigger than 2GB")

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.aicore_use = 2

        # input/output gm buffer
        self.gm_input = self.tik_instance.Tensor("float16", (self.n, self.c1, self.h, self.w, 16),
                                                 name="gm_input", scope=tik.scope_gm)
        self.gm_output = self.tik_instance.Tensor("float16", (self.n, self.c1, 1, 2, 16),
                                                  name="gm_output", scope=tik.scope_gm)
        print("(n, c, h, w):", self.n, self.c, self.h, self.w)

    def tiling_mode_select(self):
        switch = {
            1: self._mode1_init
        }

        self.mode = 1
        ret = switch[self.mode]()
        if (ret != 0):
            self.mode = 0

        print("tiling moode:", self.mode)

    def _step_select(self):
        # select step_hw and step_nc1
        step_cmp_cycle = math.ceil(math.log(min(self.hw, MAX_STEP_HW), 2.0))
        step_hw = int(math.pow(2.0, step_cmp_cycle))
        step_nc1 = 1

        while (step_hw > 0):
            ub_size = self._cal_ub_buff_size(step_nc1, step_hw)
            if (ub_size <= UB_BUFF_MAX):
                break

            step_hw = step_hw // 2

        if (step_nc1 <= 0):
            raise RuntimeError("ub buffer not enough")

        while (step_nc1 < min(self.nc1, MAX_STEP_NC1)):
            step_nc1 += 1
            ub_size = self._cal_ub_buff_size(step_nc1, step_hw)
            if (ub_size > UB_BUFF_MAX):
                step_nc1 -= 1
                break

        # step_hw must be 2^x for computing acceleration
        self.step_hw = step_hw
        self.step_nc1 = step_nc1

        if (self.step_hw != int(math.pow(2.0, math.ceil(math.log(self.step_hw, 2.0))))):
            raise RuntimeError("step_hw not be 2 ^ x")

        self.step_cmp_cycle = int(math.log(self.step_hw, 2.0))

    def _mode1_init(self):
        self._step_select()
        self.tmp1_cycle = (self.step_nc1 * self.step_hw) // (8 * 255)
        self.tmp2_cycle = (self.step_nc1 * self.step_hw - self.tmp1_cycle * 8 * 255) // 8
        self.tmp3_cycle = self.step_nc1 * self.step_hw - self.tmp1_cycle * 8 * 255 - self.tmp2_cycle * 8

        self.hw_cycle = (self.hw + self.step_hw - 1) // self.step_hw
        self.aicore_in_use_select(self.nc1)
        self.step_shape = (self.step_nc1, 1, self.step_hw, 16)
        print("step_nc1:", self.step_nc1, "step_hw:", self.step_hw, "step_cmp_cycle:", self.step_cmp_cycle)
        print("hw:", self.hw, "hw_cycle:", self.hw_cycle)
        print("ub_size:", self._cal_ub_buff_size(self.step_nc1, self.step_hw))
        return 0

    def mode1_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            with self.tik_instance.if_scope(index != self.aicore_use - 1):
                self.mode1_compute_each_core(self.xlen_each_core, (index * self.xlen_each_core))
            with self.tik_instance.else_scope():
                self.mode1_compute_each_core(self.xlen_last_core, (index * self.xlen_each_core))

        self.tik_instance.BuildCCE(kernel_name=self.kern_name, inputs=[self.gm_input],
                                   outputs=[self.gm_output], enable_l2=True)

    def _coord_tensor_prepare(self, coord_tensor, step_cycle):
        self.tik_instance.vector_dup(16, coord_tensor, -1.0, 1, 1, 1)
        start = 1
        offset = float(-1 * start)
        for i in range(0, step_cycle):
            if (start < 8):
                self.tik_instance.vadds(16 * start, coord_tensor[0, 0, start, 0],
                                        coord_tensor, offset, 1, 1, 1, start, start)
            else:
                repeat = start // 8
                self.tik_instance.vadds(128, coord_tensor[0, 0, start, 0], coord_tensor, offset, repeat, 1, 1, 8, 8)

            start = 2 * start
            offset = float(-1 * start)

        if (self.step_nc1 > 1):
            if (self.step_hw <= 8):
                self.tik_instance.vadds(16 * self.step_hw, coord_tensor[1, 0, 0, 0],
                                        coord_tensor[0, 0, 0, 0], 0.0, self.step_nc1 - 1,
                                        1, 1, self.step_hw, 0)
            else:
                if (self.step_hw >= self.nc1 * 8 or self.step_hw > 255):
                    for i in range(1, self.step_nc1):
                        self.tik_instance.data_move(coord_tensor[i, 0, 0, 0], coord_tensor[0, 0, 0, 0],
                                                    0, 1, self.step_hw, 0, 0, 0)
                else:
                    for i in range(0, int(self.step_hw // 8)):
                        self.tik_instance.vadds(128, coord_tensor[1, 0, i * 8, 0],
                                                coord_tensor[0, 0, i * 8, 0],
                                                0.0, self.step_nc1 - 1, 1, 1, self.step_hw, 0)

        for i in range(0, self.tmp1_cycle):
            self.tik_instance.vadds(128, coord_tensor[i * 8 * 255 * 16], coord_tensor[i * 8 * 255 * 16],
                                    float(self.step_hw + 1), 255, 1, 1, 8, 8)

        if (self.tmp2_cycle != 0):
            self.tik_instance.vadds(128, coord_tensor[self.tmp1_cycle * 8 * 255 * 16],
                                    coord_tensor[self.tmp1_cycle * 8 * 255 * 16],
                                    float(self.step_hw + 1), self.tmp2_cycle, 1, 1, 8, 8)

        if (self.tmp3_cycle != 0):
            self.tik_instance.vadds(16, coord_tensor[self.tmp1_cycle * 8 * 255 * 16 + self.tmp2_cycle * 8 * 16],
                                    coord_tensor[self.tmp1_cycle * 8 * 255 * 16 + self.tmp2_cycle * 8 * 16],
                                    float(self.step_hw + 1), self.tmp3_cycle, 1, 1, 1, 1)

        # list like [self.hw (self.hw-1) ... 3 2 1]

    def _pre_tensor_init(self):
        if (self.hw > self.step_hw):
            self.tik_instance.vector_dup(16, self.ub_max_val1, -30000.0, self.step_nc1, 1, 1)
            self.tik_instance.vector_dup(16, self.ub_max_index1_i32, 0, self.step_nc1, 1, 1)

    def _cal_ub_buff_size(self, step_nc1, step_hw):
        ub_size = step_nc1 * step_hw * 16 * 3
        if (step_hw > 255):
            ub_size += step_nc1 * 16 * 16

        ub_size += step_nc1 * 16 * 7
        if (self.hw > 2048):
            ub_size += step_nc1 * 16 * 2

        ub_size += step_nc1 * 16 * 8
        ub_size += 16 * 10
        if (self.hw > 65504):
            # alloc in i32_to_f32
            ub_size += (step_nc1 * 16 * 2 + 16 * 4)

        return (ub_size * 2)

    def _model1_init_bufs(self):
        self.ub_in = self.tik_instance.Tensor("float16", (self.step_nc1, 1, self.step_hw, 16),
                                              name="ub_in", scope=tik.scope_ubuf)
        self.ub_max = self.tik_instance.Tensor("float16", (self.step_nc1, 1, self.step_hw, 16),
                                               name="ub_max", scope=tik.scope_ubuf)
        self.ub_coord_hw = self.tik_instance.Tensor("float16", (self.step_nc1, 1, self.step_hw, 16),
                                                    name="ub_coord_hw", scope=tik.scope_ubuf)

        if (self.step_hw > 255):
            self.ub_tmp_hw16 = self.tik_instance.Tensor("float16", (self.step_nc1, 1, 16, 16),
                                                        name="ub_tmp_hw16", scope=tik.scope_ubuf)

        self.ub_tmp_i32 = self.tik_instance.Tensor("int32", (self.step_nc1, 1, 1, 16),
                                                   name="ub_tmp_i32", scope=tik.scope_ubuf)
        self.ub_tmp_f16 = self.tik_instance.Tensor("float16", (self.step_nc1, 1, 1, 16),
                                                   name="ub_tmp_f16", scope=tik.scope_ubuf)
        self.ub_tmp_f32 = self.tik_instance.Tensor("float32", (self.step_nc1, 1, 1, 16),
                                                   name="ub_tmp_f32", scope=tik.scope_ubuf)
        self.ub_tmp2_f32 = self.tik_instance.Tensor("float32", (self.step_nc1, 1, 1, 16),
                                                    name="ub_tmp2_f32", scope=tik.scope_ubuf)
        if (self.hw > 2048):
            self.ub_tmp3_f32 = self.tik_instance.Tensor("float32", (self.step_nc1, 1, 1, 16),
                                                        name="ub_tmp3_f32", scope=tik.scope_ubuf)

        self.ub_max_val1 = self.tik_instance.Tensor("float16", (self.step_nc1, 1, 1, 16),
                                                    name="ub_max_val1", scope=tik.scope_ubuf)
        self.ub_max_step_index1 = self.tik_instance.Tensor("float16", (self.step_nc1, 1, 1, 16),
                                                           name="ub_max_step_index1", scope=tik.scope_ubuf)
        self.ub_max_index1_i32 = self.tik_instance.Tensor("int32", (self.step_nc1, 1, 1, 16),
                                                          name="ub_max_index1_i32", scope=tik.scope_ubuf)
        self.ub_max_val2 = self.tik_instance.Tensor("float16", (self.step_nc1, 1, 1, 16),
                                                    name="ub_max_val2", scope=tik.scope_ubuf)
        self.ub_max_step_index2 = self.tik_instance.Tensor("float16", (self.step_nc1, 1, 1, 16),
                                                           name="ub_max_step_index2", scope=tik.scope_ubuf)
        self.ub_max_index2_i32 = self.tik_instance.Tensor("int32", (self.step_nc1, 1, 1, 16),
                                                          name="ub_max_index2_i32", scope=tik.scope_ubuf)

        self.ub_val_rec_w_16_f32 = self.tik_instance.Tensor("float32", (1, 1, 1, 16),
                                                            name="ub_val_rec_w_16_f32", scope=tik.scope_ubuf)
        self.ub_val_w_16_i32 = self.tik_instance.Tensor("int32", (1, 1, 1, 16),
                                                        name="ub_val_w_16_i32", scope=tik.scope_ubuf)
        self.ub_val0_16_i32 = self.tik_instance.Tensor("int32", (1, 1, 1, 16),
                                                       name="ub_val0_16_i32", scope=tik.scope_ubuf)
        self.ub_val1_16_i32 = self.tik_instance.Tensor("int32", (1, 1, 1, 16),
                                                       name="ub_val1_16_i32", scope=tik.scope_ubuf)
        self.ub_val_negtive1_16_i32 = self.tik_instance.Tensor("int32", (1, 1, 1, 16),
                                                               name="ub_val_negtive1_16_i32",
                                                               scope=tik.scope_ubuf)

    def _model1_init_sca(self):
        self.s_cur_nc1 = self.tik_instance.Scalar("int32")
        self.s_cur_hw = self.tik_instance.Scalar("int32")
        self.s_cur_step_nc1 = self.tik_instance.Scalar("int32")
        self.s_cur_step_hw = self.tik_instance.Scalar("int32")

        self.s_tmp_i32 = self.tik_instance.Scalar("int32")
        self.s_tmp2_i32 = self.tik_instance.Scalar("int32")
        self.s_tmp3_i32 = self.tik_instance.Scalar("int32")
        self.s_tmp4_i32 = self.tik_instance.Scalar("int32")

    def mode1_compute_each_core(self, xlen, xoffset):
        self._model1_init_bufs()
        self._model1_init_sca()
        nc1_cycle = (xlen + self.step_nc1 - 1) // self.step_nc1
        if (self.step_hw == 1):
            self.tik_instance.vector_dup(16, self.ub_max_val1, 0.0, self.step_nc1, 1, 1)
            with self.tik_instance.for_range(0, nc1_cycle) as i_nc1_cycle:
                self.s_cur_nc1.set_as(xoffset + i_nc1_cycle * self.step_nc1)
                with self.tik_instance.if_scope(self.nc1 - self.s_cur_nc1 >= self.step_nc1):
                    self.s_cur_step_nc1.set_as(self.step_nc1)
                with self.tik_instance.else_scope():
                    self.s_cur_step_nc1.set_as(self.nc1 - self.s_cur_nc1)

                self.tik_instance.data_move(self.gm_output[self.s_cur_nc1 * 2 * 16],
                                            self.ub_max_val1, 0, self.s_cur_step_nc1, 1, 0, 1, 0)
                self.tik_instance.data_move(self.gm_output[self.s_cur_nc1 * 2 * 16 + 16],
                                            self.ub_max_val1, 0, self.s_cur_step_nc1, 1, 0, 1, 0)
            return 0

        self.tik_instance.vector_dup(16, self.ub_val_rec_w_16_f32, 1.0 / self.w, 1, 1, 2)
        self.tik_instance.vector_dup(16, self.ub_val_w_16_i32, self.w, 1, 1, 2)
        self.tik_instance.vector_dup(16, self.ub_val0_16_i32, 0, 1, 1, 2)
        self.tik_instance.vector_dup(16, self.ub_val1_16_i32, 1, 1, 1, 2)
        self.tik_instance.vector_dup(16, self.ub_val_negtive1_16_i32, -1, 1, 1, 2)

        self._coord_tensor_prepare(self.ub_coord_hw, self.step_cmp_cycle)
        with self.tik_instance.for_range(0, nc1_cycle) as i_nc1_cycle:
            self.s_cur_nc1.set_as(xoffset + i_nc1_cycle * self.step_nc1)
            with self.tik_instance.if_scope(self.nc1 - self.s_cur_nc1 >= self.step_nc1):
                self.s_cur_step_nc1.set_as(self.step_nc1)
            with self.tik_instance.else_scope():
                self.s_cur_step_nc1.set_as(self.nc1 - self.s_cur_nc1)

            self._pre_tensor_init()
            with self.tik_instance.for_range(0, self.hw_cycle) as i_hw_cycle:
                self.s_cur_hw.set_as(i_hw_cycle * self.step_hw)
                with self.tik_instance.if_scope(self.hw - self.s_cur_hw >= self.step_hw):
                    self.s_cur_step_hw.set_as(self.step_hw)
                with self.tik_instance.else_scope():
                    self.s_cur_step_hw.set_as(self.hw - self.s_cur_hw)

                self._mode1_compute_each_loop()

            # cal and mv coord to gm_output
            self._index2coord_and_mvout(self.s_cur_step_nc1)

    def _i32_to_f32_le65504(self, ub_res, ub_in, s_cur_step_nc1, max_value):
        ub_steplen_f16 = self.ub_tmp_f16
        self.tik_instance.vconv(16, "none", ub_steplen_f16, ub_in, s_cur_step_nc1, 1, 1, 1, 2, 1.0)
        self.tik_instance.vconv(16, "none", ub_res, ub_steplen_f16, s_cur_step_nc1, 1, 1, 2, 1)
        if (max_value > 2048):
            ub_steplen_f32 = self.ub_tmp3_f32
            ub_steplen_i32 = self.ub_tmp_i32
            self.tik_instance.vconv(16, "none", ub_steplen_f16, ub_res, s_cur_step_nc1, 1, 1, 1, 2)
            self.tik_instance.vadds(16, ub_steplen_f16, ub_steplen_f16, 0.1, s_cur_step_nc1, 1, 1, 1, 1)
            self.tik_instance.vconv(16, "floor", ub_steplen_i32, ub_steplen_f16, s_cur_step_nc1, 1, 1, 2, 1)
            self.tik_instance.vsub(16, ub_steplen_i32, ub_in, ub_steplen_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vconv(16, "none", ub_steplen_f16, ub_steplen_i32, s_cur_step_nc1, 1, 1, 1, 2, 1.0)
            self.tik_instance.vconv(16, "none", ub_steplen_f32, ub_steplen_f16, s_cur_step_nc1, 1, 1, 2, 1)

            self.tik_instance.vadd(16, ub_res, ub_res, ub_steplen_f32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)

    def _i32_to_f32(self, ub_res, ub_in, s_cur_step_nc1, max_value):
        """
        translate int32 to float32
        :param ub_res: the result tensor. shape:(nc1, 1, 1, 16) dtype:float32
        :param ub_in: the op tensor. shape:(nc1, 1, 1, 16) dtype:int32
        :param s_cur_step_nc1: the valid nc1
        :return:

        retriction: shape[0] < 256
        """
        if (max_value <= 65504):
            self._i32_to_f32_le65504(ub_res, ub_in, s_cur_step_nc1, max_value)
            return

        if (self.hw > self.step_hw):
            ub_tmp2_i32 = self.ub_max_index2_i32
        else:
            ub_tmp2_i32 = self.ub_max_index1_i32

        ub_scale_i32 = self.tik_instance.Tensor("int32", (self.step_nc1, 1, 1, 16),
                                                name="ub_scale_i32", scope=tik.scope_ubuf)
        ub_val65504_16_i32 = self.tik_instance.Tensor("int32", (1, 1, 1, 16),
                                                      name="ub_val65504_16_i32", scope=tik.scope_ubuf)
        ub_val65504_16_f32 = self.tik_instance.Tensor("float32", (1, 1, 1, 16),
                                                      name="ub_val65504_16_f32", scope=tik.scope_ubuf)
        ub_val0_16_i32 = self.ub_val0_16_i32
        ub_val1_16_i32 = self.ub_val1_16_i32
        ub_tmp_i32 = self.ub_tmp_i32
        ub_tmp_f32 = self.ub_tmp_f32
        ub_scale_f16 = self.ub_tmp_f16

        self.tik_instance.vector_dup(16, ub_val65504_16_i32, 65504, 1, 1, 2)
        self.tik_instance.vector_dup(16, ub_val65504_16_f32, 65504.0, 1, 1, 2)

        self.tik_instance.vmul(16, ub_scale_i32, ub_scale_i32, ub_val0_16_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)

        self.tik_instance.data_move(ub_tmp_i32, ub_in, 0, s_cur_step_nc1, 2, 0, 0, 0)
        with self.tik_instance.for_range(0, max_value // 65504):
            self.tik_instance.vsub(16, ub_tmp_i32, ub_tmp_i32, ub_val65504_16_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
            self.tik_instance.vmax(16, ub_tmp_i32, ub_tmp_i32, ub_val0_16_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
            self.tik_instance.vmin(16, ub_tmp2_i32, ub_tmp_i32, ub_val1_16_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
            self.tik_instance.vadd(16, ub_scale_i32, ub_scale_i32, ub_tmp2_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)

        self.tik_instance.vmul(16, ub_tmp2_i32, ub_scale_i32, ub_val65504_16_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
        self.tik_instance.vsub(16, ub_tmp2_i32, ub_in, ub_tmp2_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)

        self._i32_to_f32_le65504(ub_tmp_f32, ub_tmp2_i32, s_cur_step_nc1, 65504)
        self.tik_instance.vconv(16, "none", ub_scale_f16, ub_scale_i32, s_cur_step_nc1, 1, 1, 1, 2, 1.0)
        self.tik_instance.vconv(16, "none", ub_res, ub_scale_f16, s_cur_step_nc1, 1, 1, 2, 1)
        self.tik_instance.vmul(16, ub_res, ub_res, ub_val65504_16_f32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)

        self.tik_instance.vadd(16, ub_res, ub_res, ub_tmp_f32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)

    def _round_f32_to_i32_le2048(self, ub_res, ub_src, s_cur_step_nc1):
        ub_steplen_f16 = self.ub_tmp_f16
        self.tik_instance.vadds(16, ub_src, ub_src, 0.5, s_cur_step_nc1, 1, 1, 2, 2)
        self.tik_instance.vconv(16, "none", ub_steplen_f16, ub_src, s_cur_step_nc1, 1, 1, 1, 2)
        self.tik_instance.vconv(16, "floor", ub_res, ub_steplen_f16, s_cur_step_nc1, 1, 1, 2, 1)

    def _index2coord_and_mvout(self, s_cur_step_nc1):
        if (self.hw > self.step_hw):
            ub_max_index_i32 = self.ub_max_index1_i32
            ub_h_coord_i32 = self.ub_max_index2_i32
        else:
            ub_max_index_i32 = self.ub_max_index2_i32
            ub_h_coord_i32 = self.ub_max_index1_i32

        # translate ub_max_index_i32 from int32 to float32
        self._i32_to_f32(self.ub_tmp2_f32, ub_max_index_i32, self.s_cur_step_nc1, self.hw)

        # calculate h_coord
        # index / w
        self.tik_instance.vmul(16, self.ub_tmp_f32, self.ub_tmp2_f32, self.ub_val_rec_w_16_f32,
                               s_cur_step_nc1, 1, 1, 1, 2, 2, 0)

        # calculating h_coord by int(index / w) may have precision loss, and use the way listed below.
        # round(index divide to w)
        self._round_f32_to_i32_le2048(ub_h_coord_i32, self.ub_tmp_f32, s_cur_step_nc1)

        # round(index / w) mutipy to w
        self.tik_instance.vmul(16, self.ub_tmp_i32, ub_h_coord_i32, self.ub_val_w_16_i32,
                               s_cur_step_nc1, 1, 1, 1, 2, 2, 0)

        # index sub round(index / w) * w
        self.tik_instance.vsub(16, self.ub_tmp_i32, ub_max_index_i32, self.ub_tmp_i32, s_cur_step_nc1,
                               1, 1, 1, 2, 2, 2)

        # calibrating h_coord
        self.tik_instance.vmin(16, self.ub_tmp_i32, self.ub_tmp_i32, self.ub_val0_16_i32,
                               s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
        self.tik_instance.vmax(16, self.ub_tmp_i32, self.ub_tmp_i32, self.ub_val_negtive1_16_i32,
                               s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
        self.tik_instance.vadd(16, ub_h_coord_i32, ub_h_coord_i32, self.ub_tmp_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        # translate h_coord from int32 to float16
        self.tik_instance.vconv(16, "none", self.ub_tmp_f16, ub_h_coord_i32, s_cur_step_nc1, 1, 1, 1, 2, 1.0)

        # calculate w_coord
        self.tik_instance.vmul(16, self.ub_tmp_i32, ub_h_coord_i32,
                               self.ub_val_w_16_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 0)
        self.tik_instance.vsub(16, self.ub_tmp_i32, ub_max_index_i32, self.ub_tmp_i32, s_cur_step_nc1,
                               1, 1, 1, 2, 2, 2)
        # translate w_coord from int32 to float16
        self.tik_instance.vconv(16, "none", self.ub_max_step_index2, self.ub_tmp_i32, s_cur_step_nc1, 1, 1, 1, 2, 1.0)

        # mv coord to gm_output
        self.tik_instance.data_move(self.gm_output[self.s_cur_nc1 * 2 * 16],
                                    self.ub_max_step_index2, 0, s_cur_step_nc1, 1, 0, 1, 0)
        self.tik_instance.data_move(self.gm_output[self.s_cur_nc1 * 2 * 16 + 16],
                                    self.ub_tmp_f16, 0, s_cur_step_nc1, 1, 0, 1, 0)

    def _ub_in_init(self):
        # Init ub_in[x] to min(fp16):-65504.0. Only last block need.
        with self.tik_instance.if_scope(self.s_cur_step_hw != self.step_hw):
            tmp1_cycle = self.tmp1_cycle
            tmp2_cycle = self.tmp2_cycle
            tmp3_cycle = self.tmp3_cycle
            ub_in = self.ub_in
            with self.tik_instance.for_range(0, tmp1_cycle) as i:
                self.tik_instance.vmuls(128, ub_in[i * 8 * 255 * 16],
                                        ub_in[i * 8 * 255 * 16],
                                        0.0, 255, 1, 1, 8, 8)
                self.tik_instance.vadds(128, ub_in[i * 8 * 255 * 16],
                                        ub_in[i * 8 * 255 * 16],
                                        -30000.0, 255, 1, 1, 8, 8)

            if (tmp2_cycle != 0):
                self.tik_instance.vmuls(128, ub_in[tmp1_cycle * 128 * 255],
                                        ub_in[tmp1_cycle * 128 * 255],
                                        0.0, tmp2_cycle, 1, 1, 8, 8)
                self.tik_instance.vadds(128, ub_in[tmp1_cycle * 128 * 255],
                                        ub_in[tmp1_cycle * 128 * 255],
                                        -30000.0, tmp2_cycle, 1, 1, 8, 8)

            if (tmp3_cycle != 0):
                self.tik_instance.vmuls(16, ub_in[tmp1_cycle * 128 * 255 + 128 * tmp2_cycle],
                                        ub_in[tmp1_cycle * 128 * 255 + 128 * tmp2_cycle],
                                        0.0, tmp3_cycle, 1, 1, 1, 1)
                self.tik_instance.vadds(16, ub_in[tmp1_cycle * 128 * 255 + 128 * tmp2_cycle],
                                        ub_in[tmp1_cycle * 128 * 255 + 128 * tmp2_cycle],
                                        -30000.0, tmp3_cycle, 1, 1, 1, 1)

    def _mv_gm_to_ub_in(self):
        # restriction self.s_cur_step_nc1 <= 4095 self.s_cur_step_hw <= 65535
        if (self.hw <= 65535):
            self.tik_instance.data_move(self.ub_in,
                                        self.gm_input[self.s_cur_nc1 * self.hw * 16 + self.s_cur_hw * 16],
                                        0, self.s_cur_step_nc1, self.s_cur_step_hw,
                                        self.hw - self.s_cur_step_hw, self.step_hw - self.s_cur_step_hw, 0)
        else:
            with self.tik_instance.for_range(0, self.s_cur_step_nc1) as i:
                self.tik_instance.data_move(self.ub_in[i, 0, 0, 0],
                                            self.gm_input[(self.s_cur_nc1 + i) * self.hw * 16 + self.s_cur_hw * 16],
                                            0, 1, self.s_cur_step_hw,
                                            0, 0, 0)
        # cp ub_in to ub_max
        self.tik_instance.data_move(self.ub_max, self.ub_in, 0, self.s_cur_step_nc1, self.step_hw, 0, 0, 0)

    def _find_max_val_at_hw_le256(self, tensor, s_cur_step_nc1, curr_hw, valid_hw, s_cmp_cnt):
        if (curr_hw > 255):
            raise RuntimeError("invalid curr_hw of tensor which bigger than 255")

        self.s_tmp_i32.set_as(s_cur_step_nc1 // 255)
        self.s_tmp2_i32.set_as(s_cur_step_nc1 - self.s_tmp_i32 * 255)
        self.s_tmp3_i32.set_as(valid_hw)
        with self.tik_instance.for_range(0, s_cmp_cnt):
            with self.tik_instance.for_range(0, self.s_tmp_i32) as i:
                self.tik_instance.vmax(16 * self.s_tmp3_i32, tensor[i * 255, 0, 0, 0],
                                       tensor[i * 255, 0, 0, 0],
                                       tensor[i * 255, 0, self.s_tmp3_i32, 0],
                                       255, 1, 1, 1, curr_hw, curr_hw, curr_hw)

            with self.tik_instance.if_scope(self.s_tmp2_i32 != 0):
                self.tik_instance.vmax(16 * self.s_tmp3_i32, tensor[self.s_tmp_i32 * 255, 0, 0, 0],
                                       tensor[self.s_tmp_i32 * 255, 0, 0, 0],
                                       tensor[self.s_tmp_i32 * 255, 0, self.s_tmp3_i32, 0],
                                       self.s_tmp2_i32, 1, 1, 1, curr_hw, curr_hw, curr_hw)
            self.s_tmp3_i32.set_as(self.s_tmp3_i32 // 2)

    def _get_ub_max_val(self, s_cur_step_nc1, ub_max):
        with self.tik_instance.for_range(0, s_cur_step_nc1) as i_nc1:
            with self.tik_instance.for_range(0, self.s_tmp_i32) as i:
                self.tik_instance.vmax(128, ub_max[i_nc1, 0, i * 8 * 255, 0],
                                       ub_max[i_nc1, 0, i * 8 * 255, 0],
                                       ub_max[i_nc1, 0, i * 8 * 255 + self.s_tmp4_i32, 0],
                                       255, 1, 1, 1, 8, 8, 8)

            with self.tik_instance.if_scope(self.s_tmp2_i32 != 0):
                self.tik_instance.vmax(128, ub_max[i_nc1, 0, self.s_tmp_i32 * 8 * 255, 0],
                                       ub_max[i_nc1, 0, self.s_tmp_i32 * 8 * 255, 0],
                                       ub_max[i_nc1, 0, self.s_tmp_i32 * 8 * 255 + self.s_tmp4_i32, 0],
                                       self.s_tmp2_i32, 1, 1, 1, 8, 8, 8)

            with self.tik_instance.if_scope(self.s_tmp3_i32 != 0):
                tmp_num = self.s_tmp_i32 * 8 * 255 + self.s_tmp2_i32 * 8
                self.tik_instance.vmax(16, ub_max[i_nc1, 0, tmp_num, 0],
                                       ub_max[i_nc1, 0, tmp_num, 0],
                                       ub_max[i_nc1, 0, tmp_num + self.s_tmp4_i32, 0],
                                       self.s_tmp3_i32, 1, 1, 1, 1, 1, 1)

        self.s_tmp4_i32.set_as(self.s_tmp4_i32 // 2)

    def _pick_out_max_val(self, res_tensor, ub_max, shape, s_cur_step_nc1):
        """
        pick out max value in ub_max at w-direction
        :param res_tensor: result tensor. shape was (nc1, 1, 1, 16)
        :param ub_max: op tenser. shape was (nc1, 1, w, 16)
        :param shape: the shape of ub_max. format:(nc1, 1, w, 16)
        :param s_cur_step_nc1: the valid nc1 of ub_max
        :return:

        restiction: w should be 2 ^x
        """
        step_hw = shape[2]

        # pick out max value
        length = step_hw // 2
        if length > 8:
            length = 8
            self.s_tmp4_i32.set_as(step_hw // 2)
            with self.tik_instance.for_range(0, self.step_cmp_cycle - 4):
                self.s_tmp_i32.set_as(self.s_tmp4_i32 // (8 * 255))
                self.s_tmp2_i32.set_as((self.s_tmp4_i32 - self.s_tmp_i32 * 8 * 255) // 8)
                self.s_tmp3_i32.set_as(self.s_tmp4_i32 - self.s_tmp_i32 * 8 * 255 - self.s_tmp2_i32 * 8)
                self._get_ub_max_val(s_cur_step_nc1, ub_max)
            self.s_tmp4_i32.set_as(4)
        else:
            self.s_tmp4_i32.set_as(self.step_cmp_cycle)

        if step_hw > 255:
            self.tik_instance.data_move(self.ub_tmp_hw16, ub_max, 0, s_cur_step_nc1, 16, step_hw - 16, 0, 0)
            self._find_max_val_at_hw_le256(self.ub_tmp_hw16, s_cur_step_nc1, 16, length, self.s_tmp4_i32)
            self.tik_instance.data_move(res_tensor, self.ub_tmp_hw16, 0, s_cur_step_nc1, 1, 16 - 1, 0, 0)
        else:
            self._find_max_val_at_hw_le256(ub_max, s_cur_step_nc1, step_hw, length, self.s_tmp4_i32)
            self.tik_instance.data_move(res_tensor, ub_max, 0, s_cur_step_nc1, 1, step_hw - 1, 0, 0)

    def _cal_max_val_filter_proc_tmp2(self, res_tensor, src_tensor):
        self.tik_instance.vsub(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                               res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                               src_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                               self.s_tmp2_i32, 1, 1, 1, 8, 8, 8)
        for j in range(0, 2):
            self.tik_instance.vrec(128, src_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   self.s_tmp2_i32, 1, 1, 8, 8)
            self.tik_instance.vmul(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   src_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   self.s_tmp2_i32, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vadds(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16], -1.0,
                                self.s_tmp2_i32, 1, 1, 8, 8)
        self.tik_instance.vabs(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                               res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                               self.s_tmp2_i32, 1, 1, 8, 8)
        self.tik_instance.vadds(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16], -0.5,
                                self.s_tmp2_i32, 1, 1, 8, 8)
        self.tik_instance.vrelu(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                self.s_tmp2_i32, 1, 1, 8, 8)
        self.tik_instance.vmuls(128, res_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16], 2.0,
                                self.s_tmp2_i32, 1, 1, 8, 8)

    def _cal_max_val_filter_proc_tmp3(self, res_tensor, src_tensor):
        self.tik_instance.vsub(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                               res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                               src_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                               self.s_tmp3_i32, 1, 1, 1, 1, 1, 1)

        for j in range(0, 2):
            self.tik_instance.vrec(16, src_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   self.s_tmp3_i32, 1, 1, 1, 1)
            self.tik_instance.vmul(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   src_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   self.s_tmp3_i32, 1, 1, 1, 1, 1, 1)

        self.tik_instance.vadds(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16], -1.0,
                                self.s_tmp3_i32, 1, 1, 1, 1)
        self.tik_instance.vabs(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                               res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                               self.s_tmp3_i32, 1, 1, 1, 1)

        self.tik_instance.vadds(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16], -0.5,
                                self.s_tmp3_i32, 1, 1, 1, 1)
        self.tik_instance.vrelu(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                self.s_tmp3_i32, 1, 1, 1, 1)
        self.tik_instance.vmuls(16, res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                res_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16], 2.0,
                                self.s_tmp3_i32, 1, 1, 1, 1)

    def _cal_max_val_filter(self, res_tensor, src_tensor, max_tensor, shape, s_cur_step_nc1):
        """
        filter the max value in src_tensor.
        if src_tensor[i_nc1, 0, i_w, i_c0] == max_tensor[i_nc1, 0, 0, i_c0]:
            res_tensor[i_nc1, 0, i_w, i_c0] = 1
        else:
             res_tensor[i_nc1, 0, i_w, i_c0] = 0

        :param res_tensor: the filter tensor. shape was (nc1, 1, w, 16)
        :param src_tensor: the src tensor. shape was (nc1, 1, w, 16)
        :param max_tensor: the max value tensor. shape was (nc1, 1, 1, 16)
        :param shape: the shape of  res_tensor
        :param s_cur_step_nc1: the valid nc1 of src_tensor
        :return:
        """
        w = shape[2]

        # make the max tensor with shape_(nc1, 1, w, 16)
        self.tik_instance.data_move(res_tensor, max_tensor, 0, self.s_cur_step_nc1, 1, 0, w - 1, 0)
        if (w > 1):
            self.s_tmp4_i32.set_as(1)
            with self.tik_instance.for_range(0, self.step_cmp_cycle):
                self.tik_instance.data_move(res_tensor[0, 0, self.s_tmp4_i32, 0], res_tensor,
                                            0, s_cur_step_nc1, self.s_tmp4_i32,
                                            w - self.s_tmp4_i32, w - self.s_tmp4_i32, 0)

                self.s_tmp4_i32.set_as(self.s_tmp4_i32 * 2)

        self.s_tmp4_i32.set_as(s_cur_step_nc1 * w)
        self.s_tmp_i32.set_as(self.s_tmp4_i32 // (8 * 255))
        self.s_tmp2_i32.set_as((self.s_tmp4_i32 - self.s_tmp_i32 * 8 * 255) // 8)
        self.s_tmp3_i32.set_as(self.s_tmp4_i32 - self.s_tmp_i32 * 8 * 255 - self.s_tmp2_i32 * 8)
        with self.tik_instance.for_range(0, self.s_tmp_i32) as i:
            self.tik_instance.vsub(128, res_tensor[i * 8 * 255 * 16],
                                   res_tensor[i * 8 * 255 * 16],
                                   src_tensor[i * 8 * 255 * 16],
                                   255, 1, 1, 1, 8, 8, 8)

            # condition1 res_tensor[i_nc1, 0, i_w, i_c0] equal to  0: res_tensor[i_nc1, 0, i_w, i_c0] equal to 0
            # condition2 res_tensor[i_nc1, 0, i_w, i_c0] equal to  1.0_around
            for j in range(0, 2):
                self.tik_instance.vrec(128, src_tensor[i * 8 * 255 * 16],
                                       res_tensor[i * 8 * 255 * 16],
                                       255, 1, 1, 8, 8)
                self.tik_instance.vmul(128, res_tensor[i * 8 * 255 * 16],
                                       res_tensor[i * 8 * 255 * 16],
                                       src_tensor[i * 8 * 255 * 16],
                                       255, 1, 1, 1, 8, 8, 8)

            # condition1 res_tensor[i_nc1, 0, i_w, i_c0] equal to 0: res_tensor[i_nc1, 0, i_w, i_c0] equal to 1.0
            # condition2 res_tensor[i_nc1, 0, i_w, i_c0] equal to 0.0_around
            self.tik_instance.vadds(128, res_tensor[i * 8 * 255 * 16],
                                    res_tensor[i * 8 * 255 * 16], -1.0,
                                    255, 1, 1, 8, 8)
            self.tik_instance.vabs(128, res_tensor[i * 8 * 255 * 16],
                                   res_tensor[i * 8 * 255 * 16],
                                   255, 1, 1, 8, 8)

            # make sure 0.0_around to 0.0
            self.tik_instance.vadds(128, res_tensor[i * 8 * 255 * 16],
                                    res_tensor[i * 8 * 255 * 16], -0.5,
                                    255, 1, 1, 8, 8)
            self.tik_instance.vrelu(128, res_tensor[i * 8 * 255 * 16],
                                    res_tensor[i * 8 * 255 * 16],
                                    255, 1, 1, 8, 8)
            self.tik_instance.vmuls(128, res_tensor[i * 8 * 255 * 16],
                                    res_tensor[i * 8 * 255 * 16], 2.0,
                                    255, 1, 1, 8, 8)

        with self.tik_instance.if_scope(self.s_tmp2_i32 != 0):
            self._cal_max_val_filter_proc_tmp2(res_tensor, src_tensor)

        with self.tik_instance.if_scope(self.s_tmp3_i32 != 0):
            self._cal_max_val_filter_proc_tmp3(res_tensor, src_tensor)

    def _get_index_of_max(self, res_tensor, coord_tensor, filter_tensor, shape, s_cur_step_nc1):
        """

        :param res_tensor: the index tensor. shape:(nc1, 1, 1, 16)
        :param coord_tensor: the coord tensor. shape:(nc1, 1, w, 16)
        :param filter_tensor: the coord tensor. shape:(nc1, 1, w, 16)
        :param shape: the shape of filter_tensor
        :param s_cur_step_nc1: the valid nc1 of src_tensor
        :return:
        """
        w = shape[2]
        self.s_tmp4_i32.set_as(s_cur_step_nc1 * w)
        self.s_tmp_i32.set_as(self.s_tmp4_i32 // (8 * 255))
        self.s_tmp2_i32.set_as((self.s_tmp4_i32 - self.s_tmp_i32 * 8 * 255) // 8)
        self.s_tmp3_i32.set_as(self.s_tmp4_i32 - self.s_tmp_i32 * 8 * 255 - self.s_tmp2_i32 * 8)

        with self.tik_instance.for_range(0, self.s_tmp_i32) as i:
            self.tik_instance.vmul(128, filter_tensor[i * 8 * 255 * 16],
                                   filter_tensor[i * 8 * 255 * 16],
                                   coord_tensor[i * 8 * 255 * 16],
                                   255, 1, 1, 1, 8, 8, 8)

        with self.tik_instance.if_scope(self.s_tmp2_i32 != 0):
            self.tik_instance.vmul(128, filter_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   filter_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   coord_tensor[self.s_tmp_i32 * 8 * 255 * 16],
                                   self.s_tmp2_i32, 1, 1, 1, 8, 8, 8)

        with self.tik_instance.if_scope(self.s_tmp3_i32 != 0):
            self.tik_instance.vmul(16, filter_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   filter_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   coord_tensor[self.s_tmp_i32 * 8 * 255 * 16 + self.s_tmp2_i32 * 8 * 16],
                                   self.s_tmp3_i32, 1, 1, 1, 1, 1, 1)

        # pick out the max value tensor of filter_tensor
        self._pick_out_max_val(res_tensor, filter_tensor, shape, s_cur_step_nc1)
        self.tik_instance.vadds(16, res_tensor, res_tensor, float(-1 * (w + 1)), s_cur_step_nc1, 1, 1, 1, 1)
        self.tik_instance.vmuls(16, res_tensor, res_tensor, -1.0, s_cur_step_nc1, 1, 1, 1, 1)

    def _cal_max_and_index(self, max_tensor, index_tensor, src_tensor, shape):
        """
        pick out the max value and its index in src_tensor at w-driction
        :param max_tensor: the max tensor. shape was (nc1, 1, 1, 16)
        :param index_tensor: the index tensor. shape:(nc1, 1, 1, 16)
        :param src_tensor: op tensor. shape was (nc1, 1, w, 16)
        :param shape: the shape of src_tensor
        :return: the index tensor of max value
        """
        # pick out max value in src_tensor at w-direction
        self._pick_out_max_val(max_tensor, src_tensor, shape, self.s_cur_step_nc1)

        self._cal_max_val_filter(src_tensor, self.ub_in, max_tensor, shape, self.s_cur_step_nc1)

        self._get_index_of_max(index_tensor, self.ub_coord_hw, src_tensor, shape, self.s_cur_step_nc1)

    def _cmp_with_pre_result(self, pre_max_tensor, pre_index_tensor, max_tensor, index_tensor, s_cur_step_nc1):
        """
        compare max_tensor and pick out the index of max value. the result save to pre tensor
        :param pre_max_tensor: pre max_tensor. shape:(nc1, 1, 1, 1)
        :param pre_index_tensor: pre index tensor. shape:(nc1, 1, 1, 1)
        :param max_tensor: current max_tensor. shape:(nc1, 1, 1, 1)
        :param index_tensor: current index tensor. shape:(nc1, 1, 1, 1)
        :param s_cur_step_nc1: the valid nc1 length
        :return:
        """
        self.tik_instance.vmax(16, max_tensor, pre_max_tensor, max_tensor, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vsub(16, pre_max_tensor, max_tensor, pre_max_tensor, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vrec(16, self.ub_tmp_f16, pre_max_tensor, s_cur_step_nc1, 1, 1, 2, 2)
        self.tik_instance.vmul(16, pre_max_tensor, pre_max_tensor, self.ub_tmp_f16, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vrec(16, self.ub_tmp_f16, pre_max_tensor, s_cur_step_nc1, 1, 1, 2, 2)
        self.tik_instance.vmul(16, pre_max_tensor, pre_max_tensor, self.ub_tmp_f16, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vadds(16, pre_max_tensor, pre_max_tensor, 0.1, s_cur_step_nc1, 1, 1, 2, 2)
        self.tik_instance.vconv(16, "floor", self.ub_tmp_i32, pre_max_tensor, s_cur_step_nc1, 1, 1, 2, 1)

        # get the index tensor of max value
        self.tik_instance.vmul(16, index_tensor, index_tensor, self.ub_tmp_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vmax(16, pre_index_tensor, pre_index_tensor, index_tensor, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
        # get the max value tensor
        self.tik_instance.data_move(pre_max_tensor, max_tensor, 0, s_cur_step_nc1, 1, 0, 0, 0)

    def _mode1_compute_each_loop(self):
        self._ub_in_init()

        self._mv_gm_to_ub_in()

        self._cal_max_and_index(self.ub_max_val2, self.ub_max_step_index2, self.ub_max, self.step_shape)

        # restriction step_nc1 < 256
        # restriction step_hw < 2049, otherwise it will make precision loss
        s_cur_step_nc1 = self.s_cur_step_nc1
        # translate the index tensor from fp16 to int32
        # now tensor start with 1, but we need to make tensor start with 0 for index2coord translation.
        self.tik_instance.vadds(16, self.ub_max_step_index2, self.ub_max_step_index2, -0.9,
                                s_cur_step_nc1, 1, 1, 1, 1)
        self.tik_instance.vconv(16, "floor", self.ub_max_index2_i32, self.ub_max_step_index2,
                                s_cur_step_nc1, 1, 1, 2, 1)
        if (self.hw > self.step_hw):
            # add index offset
            self.tik_instance.vector_dup(16, self.ub_tmp_i32, self.s_cur_hw, s_cur_step_nc1, 1, 2)
            self.tik_instance.vadd(16, self.ub_max_index2_i32, self.ub_max_index2_i32,
                                   self.ub_tmp_i32, s_cur_step_nc1, 1, 1, 1, 2, 2, 2)
            # compare with last index tensor and pick out the max one
            self._cmp_with_pre_result(self.ub_max_val1, self.ub_max_index1_i32,
                                      self.ub_max_val2, self.ub_max_index2_i32, s_cur_step_nc1)

    def tik_output_debug(self):
        data_np_a = np.random.random([self.n, self.c1, self.h, self.w, 16]).astype(np.float)
        data_a_tran = data_np_a.astype(np.float16)
        for j in range(0, self.h):
            for i in range(0, self.w):
                data_a_tran[:, :, j, i, :] = i * 0.1 + 1 * j

        print("data_in", data_a_tran)
        feed_dict = {
            "gm_input": data_a_tran,
        }

        out, = self.tik_instance.tikdb.start_debug(feed_dict, False)
        print("data_out", out)


def heatmap2_coord(x, y, kernel_name="heatmap2_coord", test=False):
    shape = x.get("shape")
    obj = HeatMapToCoord(shape, kernel_name)

    obj.tiling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {
        1: obj.mode1_compute
    }

    switch[obj.mode]()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
