from te import tik
import math
import numpy as np

ERR = 1
UB_BUFF_MAX = 240 * 1024


class tik_resample:
    def _cmd_cycle_and_offset(self, steplen_16):
        offset1 = 0
        cycle1 = steplen_16 // (64 * 255)
        offset2 = offset1 + cycle1 * (64 * 255)
        cycle2 = (steplen_16 - offset2) // 64
        offset3 = offset2 + cycle2 * 64
        cycle3 = steplen_16 - offset3
        return offset1, cycle1, offset2, cycle2, offset3, cycle3

    def _floor_f32toi32(self, ub_ret, ub_in, steplen_16):
        """
        The calculation formula of the following code ub_ret = floor(ub_in)
        constrictions:
            steplen_16 <= 16 * 255
        """

        ub_functmp_f16 = self.ub_steplen_f16
        ub_functmp_f32 = self.ub_steplen_f32
        ub_functmp2_f32 = self.ub_steplen2_f32
        ub_functmp2_i32 = self.ub_steplen2_i32
        ub_16_f32_val0 = self.ub_16_f32_val0

        self.tik_instance.vadds(16, ub_functmp2_f32, ub_in, 0.5, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2)
        self.tik_instance.vconv(16, "floor", ub_functmp2_i32, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
        self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_i32, steplen_16 // 16, 1, 1, 1, 2, 1.0)
        self.tik_instance.vconv(16, "none", ub_functmp2_f32, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
        self.tik_instance.vsub(16, ub_functmp2_f32, ub_in, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)

        if self.is_large == 1:
            self.tik_instance.vadds(16, ub_functmp_f32, ub_functmp2_f32, 0.5, steplen_16 // 16, 1, 1, 2, 2)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp_f32, steplen_16 // 16, 1, 1, 1, 2)
            self.tik_instance.vconv(16, "floor", ub_ret, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_ret, steplen_16 // 16, 1, 1, 1, 2, 1.0)
            self.tik_instance.vconv(16, "none", ub_functmp_f32, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
            self.tik_instance.vadd(16, ub_functmp2_i32, ub_ret, ub_functmp2_i32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vsub(16, ub_functmp2_f32, ub_functmp2_f32, ub_functmp_f32, steplen_16 // 16, 1, 1, 1, 2,
                                   2, 2)

        # The calculation formula of the following code out = -1 when in < 0, out = 0 when in >= 0.
        self.tik_instance.vmin(16, ub_functmp2_f32, ub_functmp2_f32, ub_16_f32_val0, steplen_16 // 16, 1, 1, 0, 2, 2, 0)
        self.tik_instance.vrec(16, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vabs(16, ub_functmp_f32, ub_functmp_f32, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vmul(16, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vrec(16, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vabs(16, ub_functmp_f32, ub_functmp_f32, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vmul(16, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)
        # add 0.5 to make sure result of vconv precisely
        self.tik_instance.vadds(16, ub_functmp2_f32, ub_functmp2_f32, 0.5, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2)
        self.tik_instance.vconv(16, "floor", ub_ret, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
        self.tik_instance.vadd(16, ub_ret, ub_ret, ub_functmp2_i32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)

    def _ceil_f32toi32(self, ub_ret, ub_in, steplen_16):
        """
        The calculation formula of the following code  ub_ret = ceil(ub_in)
        constrictions:
            steplen_16 <= 16 * 255
        """

        ub_functmp_f16 = self.ub_steplen_f16
        ub_functmp_f32 = self.ub_steplen_f32
        ub_functmp2_f32 = self.ub_steplen2_f32
        ub_functmp2_i32 = self.ub_steplen2_i32
        ub_16_f32_val0 = self.ub_16_f32_val0

        self.tik_instance.vadds(16, ub_functmp2_f32, ub_in, 0.5, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2)
        self.tik_instance.vconv(16, "floor", ub_functmp2_i32, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
        self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_i32, steplen_16 // 16, 1, 1, 1, 2, 1.0)
        self.tik_instance.vconv(16, "none", ub_functmp2_f32, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
        self.tik_instance.vsub(16, ub_functmp2_f32, ub_in, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)

        if self.is_large == 1:
            self.tik_instance.vadds(16, ub_functmp_f32, ub_functmp2_f32, 0.5, steplen_16 // 16, 1, 1, 2, 2)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp_f32, steplen_16 // 16, 1, 1, 1, 2)
            self.tik_instance.vconv(16, "floor", ub_ret, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
            self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_ret, steplen_16 // 16, 1, 1, 1, 2, 1.0)
            self.tik_instance.vconv(16, "none", ub_functmp_f32, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
            self.tik_instance.vadd(16, ub_functmp2_i32, ub_ret, ub_functmp2_i32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)
            self.tik_instance.vsub(16, ub_functmp2_f32, ub_functmp2_f32, ub_functmp_f32, steplen_16 // 16, 1, 1, 1, 2,
                                   2, 2)

        # The calculation formula of the following code out = 0 when in <= 0, out = 1 when in > 0.
        self.tik_instance.vmax(16, ub_functmp2_f32, ub_functmp2_f32, ub_16_f32_val0, steplen_16 // 16, 1, 1, 0, 2, 2, 0)
        self.tik_instance.vrec(16, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vmul(16, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)
        self.tik_instance.vrec(16, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vmul(16, ub_functmp2_f32, ub_functmp_f32, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)

        # add 0.5 to make sure result of vconv precisely
        self.tik_instance.vadds(16, ub_functmp2_f32, ub_functmp2_f32, 0.5, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vconv(16, "none", ub_functmp_f16, ub_functmp2_f32, steplen_16 // 16, 1, 1, 1, 2)
        self.tik_instance.vconv(16, "floor", ub_ret, ub_functmp_f16, steplen_16 // 16, 1, 1, 2, 1)
        self.tik_instance.vadd(16, ub_ret, ub_ret, ub_functmp2_i32, steplen_16 // 16, 1, 1, 1, 2, 2, 2)

    def _i32tof32(self, ub_ret, ub_in, steplen_16, ub_steplen_f16, ub_steplen_f32, ub_steplen_i32):
        """
        The calculation formula of the following code ub_ret = float(ub_in)
        constrictions:
            steplen_16 % 16 == 0
            ub_in[x] < 65504
        """

        offset1, cycle1, offset2, cycle2, offset3, cycle3 = self._cmd_cycle_and_offset(steplen_16)
        for i in range(0, cycle1):
            self.tik_instance.vconv(64, "none", ub_steplen_f16, ub_in, 255, 1, 1, 4, 8, 1.0)
            self.tik_instance.vconv(64, "none", ub_ret, ub_steplen_f16, 255, 1, 1, 8, 4)
            # if ub_in[x] > 2048, due to the fp16 precision, the following operations need to be performed:
            if self.is_large == 1:
                # obtain the difference between the original i32 input and the converted value for calibration
                self.tik_instance.vconv(64, "none", ub_steplen_f16, ub_ret, 255, 1, 1, 4, 8)
                self.tik_instance.vadds(64, ub_steplen_f16, ub_steplen_f16, 0.1, 255, 1, 1, 4, 4)
                self.tik_instance.vconv(64, "floor", ub_steplen_i32, ub_steplen_f16, 255, 1, 1, 8, 4)
                self.tik_instance.vsub(64, ub_steplen_i32, ub_in, ub_steplen_i32, 255, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vconv(64, "none", ub_steplen_f16, ub_steplen_i32, 255, 1, 1, 4, 8, 1.0)
                self.tik_instance.vconv(64, "none", ub_steplen_f32, ub_steplen_f16, 255, 1, 1, 8, 4)
                self.tik_instance.vadd(64, ub_ret, ub_ret, ub_steplen_f32, 255, 1, 1, 1, 8, 8, 8)

        if cycle2 != 0:
            self.tik_instance.vconv(64, "none", ub_steplen_f16[offset2], ub_in[offset2], cycle2, 1, 1, 4, 8, 1.0)
            self.tik_instance.vconv(64, "none", ub_ret[offset2], ub_steplen_f16[offset2], cycle2, 1, 1, 8, 4)
            if self.is_large == 1:
                self.tik_instance.vconv(64, "none", ub_steplen_f16[offset2], ub_ret[offset2], cycle2, 1, 1, 4, 8)
                self.tik_instance.vadds(64, ub_steplen_f16[offset2], ub_steplen_f16[offset2], 0.1, cycle2, 1, 1, 4, 4)
                self.tik_instance.vconv(64, "floor", ub_steplen_i32[offset2], ub_steplen_f16[offset2], cycle2, 1, 1, 8,
                                        4)
                self.tik_instance.vsub(64, ub_steplen_i32[offset2], ub_in[offset2], ub_steplen_i32[offset2], cycle2, 1,
                                       1, 1, 8, 8, 8)
                self.tik_instance.vconv(64, "none", ub_steplen_f16[offset2], ub_steplen_i32[offset2], cycle2, 1, 1, 4,
                                        8, 1.0)
                self.tik_instance.vconv(64, "none", ub_steplen_f32[offset2], ub_steplen_f16[offset2], cycle2, 1, 1, 8,
                                        4)
                self.tik_instance.vadd(64, ub_ret[offset2], ub_ret[offset2], ub_steplen_f32[offset2], cycle2, 1, 1, 1,
                                       8, 8, 8)

        if cycle3 != 0:
            fp16_stride = cycle3 // 16
            fp32_stride = cycle3 // 8
            self.tik_instance.vconv(cycle3, "none", ub_steplen_f16[offset3], ub_in[offset3], 1, 1, 1, fp16_stride,
                                    fp32_stride, 1.0)
            self.tik_instance.vconv(cycle3, "none", ub_ret[offset3], ub_steplen_f16[offset3], 1, 1, 1, 16, fp16_stride)
            if self.is_large == 1:
                self.tik_instance.vconv(cycle3, "none", ub_steplen_f16[offset3], ub_ret[offset3], 1, 1, 1, fp16_stride,
                                        fp32_stride)
                self.tik_instance.vadds(cycle3, ub_steplen_f16[offset3], ub_steplen_f16[offset3], 0.1, 1, 1, 1,
                                        fp16_stride, fp16_stride)
                self.tik_instance.vconv(cycle3, "floor", ub_steplen_i32[offset3], ub_steplen_f16[offset3], 1, 1, 1,
                                        fp32_stride, fp16_stride)
                self.tik_instance.vsub(cycle3, ub_steplen_i32[offset3], ub_in[offset3], ub_steplen_i32[offset3], 1, 1,
                                       1, 1, fp32_stride, fp32_stride, fp32_stride)
                self.tik_instance.vconv(cycle3, "none", ub_steplen_f16[offset3], ub_steplen_i32[offset3], 1, 1, 1,
                                        fp16_stride, fp32_stride, 1.0)
                self.tik_instance.vconv(cycle3, "none", ub_steplen_f32[offset3], ub_steplen_f16[offset3], 1, 1, 1,
                                        fp32_stride, fp16_stride)
                self.tik_instance.vadd(cycle3, ub_ret[offset3], ub_ret[offset3], ub_steplen_f32[offset3], 1, 1, 1, 1,
                                       fp32_stride, fp32_stride, fp32_stride)

    def global_init(self):
        global n, c, h, w, c1, nc1, type, out_h, out_w
        global ax, ay, fx, fy, rx, ry
        global win_factor, win_xlen_max, win_ylen_max
        global x_steplen, x_steplen_16, y_steplen, y_steplen_16, x_repeat, y_repeat

        n = self.n
        c = self.c
        h = self.h
        w = self.w
        c1 = self.c1
        nc1 = self.nc1
        type = self.type
        out_h = self.out_h
        out_w = self.out_w
        ax = self.ax
        ay = self.ay
        fx = self.fx
        fy = self.fy
        rx = self.rx
        ry = self.ry
        win_factor = self.win_factor
        win_xlen_max = self.win_xlen_max
        win_ylen_max = self.win_ylen_max

        x_steplen = self.x_steplen
        x_steplen_16 = self.x_steplen_16
        y_steplen = self.y_steplen
        y_steplen_16 = self.y_steplen_16
        x_repeat = self.x_repeat
        y_repeat = self.y_repeat

        print("n c h w", n, c, h, w)
        print("out_h out_w:", out_h, out_w)
        print("ax ay:", ax, ay)
        print("fx fy:", fx, fy)
        print("win_xlen_max win_ylen_max:", win_xlen_max, win_ylen_max)
        print("x_steplen y_steplen:", x_steplen, y_steplen)
        print("aicore_use:", self.aicore_use)

    def _aicore_in_use_select(self, len):
        self.xlen_each_core = (len + self.aicore_use - 1) // self.aicore_use
        self.xlen_last_core = len - self.xlen_each_core * (self.aicore_use - 1)
        self.aicore_use = (len + self.xlen_each_core - 1) // self.xlen_each_core
        if (self.aicore_use == 1):
            self.xlen_last_core = self.xlen_each_core
        print("self.xlen_each_core:", self.xlen_each_core, "self.xlen_last_core:", self.xlen_last_core)

    def _get_attr(self, antialias, fx, fy, type):
        if ((fx > 1) or (fy > 1)):
            pass
        else:
            antialias = False

        if (type == 1):
            kernel_width = 1.0
            win_factor = 0.5
        elif (type == 2):
            kernel_width = 2.0
            win_factor = 1.0
        else:
            raise RuntimeError("only support NEAREST and LINEAR.")

        if (antialias is False):
            ax = 1.0
            ay = 1.0
        else:
            ax = 1.0 / fx
            ay = 1.0 / fy

        if (fx < 1.0):
            rx = 2
        else:
            rx = int(math.ceil(kernel_width / ax))

        if (fy < 1.0):
            ry = 2
        else:
            ry = int(math.ceil(kernel_width / ay))
        return antialias, kernel_width, win_factor, ax, ay, rx, ry

    def __init__(self, shape, type, antialias, out_h, out_w, kern_name):
        n, c1, h, w = shape[0], shape[1], shape[2], shape[3]
        c = c1 * 16

        fx = float(w) / float(out_w)
        fy = float(h) / float(out_h)
        antialias, kernel_width, win_factor, ax, ay, rx, ry = self._get_attr(antialias, fx, fy, type)

        win_xlen_max = min((2 * rx + 1), int(math.ceil((2.0 * win_factor) / ax)))
        win_ylen_max = min((2 * ry + 1), int(math.ceil((2.0 * win_factor) / ay)))
        if (type == 1):
            win_xlen_max = 1
            win_ylen_max = 1

        if (out_w <= 2048 and out_w * fx + fy / 2.0 <= 2048 and out_h <= 2048 and out_h * fy + fx / 2.0 <= 2048):
            self.is_large = 0
        elif (out_w > 131008 or out_w * fx + fy / 2.0 > 131008 or out_h > 131008 and out_h * fy + fx / 2.0 > 131008):
            raise RuntimeError("in_w or out_w or in_h or out_h was supposed not bigger than 131008.")
        else:
            self.is_large = 1

        self.n, self.c, self.h, self.w = n, c, h, w
        self.c1, self.nc1, self.type, self.antialias = c1, n * c1, type, antialias
        self.out_h, self.out_w, self.ax, self.ay = out_h, out_w, ax, ay
        self.fx, self.fy, self.rx, self.ry = fx, fy, rx, ry
        self.kernel_width, self.win_factor = kernel_width, win_factor
        self.win_xlen_max, self.win_ylen_max, self.kern_name = win_xlen_max, win_ylen_max, kern_name

        self.tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
        self.aicore_use = 2

        self.wscale = self.out_w // self.w
        if (self.wscale * self.w != self.out_w):
            self.wscale = 0

        # input/output gm buffer
        self.gm_input = self.tik_instance.Tensor("float16", (n, c1, h, w, 16), name="gm_input", scope=tik.scope_gm)
        self.gm_output = self.tik_instance.Tensor("float16", (n, c1, out_h, out_w, 16), name="gm_output",
                                                  scope=tik.scope_gm)

    def _const_tensor_and_scalar_declare(self):
        self.ub_16_f16_val1 = self.tik_instance.Tensor("float16", (16,), name="ub_16_f16_val1", scope=tik.scope_ubuf)
        self.ub_16_i32_val0 = self.tik_instance.Tensor("int32", (16,), name="ub_16_i32_val0", scope=tik.scope_ubuf)
        self.ub_16_f32_val0 = self.tik_instance.Tensor("float32", (16,), name="ub_16_f32_val0", scope=tik.scope_ubuf)
        self.ub_16_f32_val1 = self.tik_instance.Tensor("float32", (16,), name="ub_16_f32_val1", scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(16, self.ub_16_f16_val1, 1.0, 1, 1, 1)
        self.tik_instance.vector_dup(16, self.ub_16_i32_val0, 0, 1, 1, 2)
        self.tik_instance.vector_dup(16, self.ub_16_f32_val0, 0.0, 1, 1, 2)
        self.tik_instance.vector_dup(16, self.ub_16_f32_val1, 1.0, 1, 1, 2)

        self.cur_wlen = self.tik_instance.Scalar("int32")
        self.cur_hlen = self.tik_instance.Scalar("int32")
        self.start_w = self.tik_instance.Scalar("int32")
        self.start_h = self.tik_instance.Scalar("int32")
        self.tmp_i32 = self.tik_instance.Scalar("int32")
        self.tmp2_i32 = self.tik_instance.Scalar("int32")
        self.tmp_f32 = self.tik_instance.Scalar("float32")
        self.tmp_f16 = self.tik_instance.Scalar("float16")

    def _common_declare(self):
        self.ub_x_in_round = self.tik_instance.Tensor("int32", (x_steplen_16,), name="ub_x_in_round",
                                                      scope=tik.scope_ubuf)
        self.ub_y_in_round = self.tik_instance.Tensor("int32", (y_steplen_16,), name="ub_y_in_round",
                                                      scope=tik.scope_ubuf)
        # x_start in point-select-win
        self.ub_x_start = self.tik_instance.Tensor("int32", (x_steplen_16,), name="ub_x_start", scope=tik.scope_ubuf)
        # y_start in point-select-win
        self.ub_y_start = self.tik_instance.Tensor("int32", (y_steplen_16,), name="ub_y_start", scope=tik.scope_ubuf)

        self.ub_aadx_win = self.tik_instance.Tensor("float16", (1, 1, win_xlen_max, x_steplen, 16), name="ub_aadx_win",
                                                    scope=tik.scope_ubuf)
        self.ub_aady_win = self.tik_instance.Tensor("float16", (1, 1, win_ylen_max, y_steplen, 16), name="ub_aady_win",
                                                    scope=tik.scope_ubuf)
        self.ub_aadx_sum = self.tik_instance.Tensor("float16", (1, 1, 1, x_steplen, 16), name="ub_aadx_sum",
                                                    scope=tik.scope_ubuf)
        self.ub_aady_sum = self.tik_instance.Tensor("float16", (1, 1, 1, y_steplen, 16), name="ub_aady_sum",
                                                    scope=tik.scope_ubuf)
        self.ub_aadxy_sum = self.tik_instance.Tensor("float16", (1, 1, 1, x_steplen, 16), name="ub_aadxy_sum",
                                                     scope=tik.scope_ubuf)
        self.ub_aadxy_p_f16 = self.tik_instance.Tensor("float16", (1, 1, 1, x_steplen, 16), name="ub_aadxy_p_f16",
                                                       scope=tik.scope_ubuf)

        self.ub_x_in_f32 = self.tik_instance.Tensor("float32", (x_steplen_16,), name="ub_x_in_f32",
                                                    scope=tik.scope_ubuf)
        self.ub_y_in_f32 = self.tik_instance.Tensor("float32", (y_steplen_16,), name="ub_y_in_f32",
                                                    scope=tik.scope_ubuf)
        self.ub_steplen_f16 = self.tik_instance.Tensor("float16", (max(x_steplen_16, y_steplen_16),),
                                                       name="ub_steplen_f16", scope=tik.scope_ubuf)
        self.ub_steplen_f32 = self.tik_instance.Tensor("float32", (max(x_steplen_16, y_steplen_16),),
                                                       name="ub_steplen_f32", scope=tik.scope_ubuf)
        self.ub_steplen2_f32 = self.tik_instance.Tensor("float32", (max(x_steplen_16, y_steplen_16),),
                                                        name="ub_steplen2_f32", scope=tik.scope_ubuf)
        self.ub_steplen3_f32 = self.tik_instance.Tensor("float32", (max(x_steplen_16, y_steplen_16),),
                                                        name="ub_steplen3_f32", scope=tik.scope_ubuf)
        self.ub_steplen_i32 = self.tik_instance.Tensor("int32", (max(x_steplen_16, y_steplen_16),),
                                                       name="ub_steplen_i32", scope=tik.scope_ubuf)
        self.ub_steplen2_i32 = self.tik_instance.Tensor("int32", (max(x_steplen_16, y_steplen_16),),
                                                        name="ub_steplen2_i32", scope=tik.scope_ubuf)

        self.ub_steplen_16f16 = self.tik_instance.Tensor("float16", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                         name="ub_steplen_16f16", scope=tik.scope_ubuf)
        self.ub_steplen_16f32 = self.tik_instance.Tensor("float32", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                         name="ub_steplen_16f32", scope=tik.scope_ubuf)
        self.ub_steplen2_16f32 = self.tik_instance.Tensor("float32", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                          name="ub_steplen2_16f32", scope=tik.scope_ubuf)
        self.ub_xy_16f32 = self.tik_instance.Tensor("float32", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                    name="ub_xy_16f32", scope=tik.scope_ubuf)
        self.ub_xy_in_16f32 = self.tik_instance.Tensor("float32", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                       name="ub_xy_in_16f32", scope=tik.scope_ubuf)
        self.ub_aadxy_p_i32 = self.tik_instance.Tensor("int32", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                       name="ub_aadxy_p_i32", scope=tik.scope_ubuf)
        self.ub_steplen_16i32 = self.tik_instance.Tensor("int32", (1, 1, 1, max(x_steplen, y_steplen), 16),
                                                         name="ub_steplen_16i32", scope=tik.scope_ubuf)

        self._const_tensor_and_scalar_declare()

    def tiling_mode_select(self):
        # mode:1-tiling in w, 2-tiling in nc1, 4-optimal tiling in nc1
        mode_list = [4, 2, 1]
        switch = {
            1: self._mode1_init,
            2: self._mode2_init,
            4: self._mode4_init
        }
        if (self.wscale != 0 and self.wscale < 100 and self.nc1 * 8 * 4 >= self.out_w):
            self.mode = 4
        elif (self.nc1 * 8 >= self.out_w):
            self.mode = 2
        else:
            self.mode = 1

        ret = switch[self.mode]()
        if (ret != 0):
            for i in mode_list:
                self.mode = i
                ret = switch[self.mode]()
                if (ret == 0):
                    break
                self.mode = 0

        print("tiling moode:", self.mode)

    def _common_ubsize(self, x_steplen, y_steplen):
        win_xlen_max = self.win_xlen_max
        win_ylen_max = self.win_ylen_max

        x_steplen_16 = ((x_steplen + 15) // 16) * 16
        y_steplen_16 = ((y_steplen + 15) // 16) * 16

        ubsize = 0
        ubsize += ((2 * x_steplen_16 + 2 * y_steplen_16) * 2)
        ubsize += (win_xlen_max * x_steplen * 16 + win_ylen_max * y_steplen * 16 +
                   x_steplen * 16 * 3 + y_steplen * 16 + 16)
        ubsize += (2 * x_steplen_16 + 2 * y_steplen_16 + max(x_steplen_16, y_steplen_16) * 11)
        ubsize += (max(x_steplen, y_steplen) * 16 * 12)
        ubsize += 48

        return ubsize

    def _mode4_ubsize(self, x_steplen, y_steplen):
        win_xlen_max = self.win_xlen_max
        win_ylen_max = self.win_ylen_max
        nc1 = self.nc1
        w = self.w
        wscale = self.wscale

        ubsize = self._common_ubsize(x_steplen, y_steplen)

        ubsize += (min(255, nc1) * x_steplen * 16)
        if (win_xlen_max != 1):
            ubsize += nc1 * win_ylen_max * w * 16 + nc1 * win_ylen_max * win_xlen_max * x_steplen * 16
        else:
            ubsize += nc1 * win_ylen_max * w * 16 + nc1 * win_ylen_max * (win_xlen_max * x_steplen + wscale) * 16

        ubsize = ubsize * 2
        return ubsize

    def _mode4_init_check(self):
        if (self.h * self.w > 65535 or self.out_h * self.out_w > 65535):
            return ERR
        if (self.h <= self.win_ylen_max):
            return ERR
        if (self.nc1 * self.win_ylen_max > 255):
            return ERR
        return None

    def _mode4_init(self):
        if self._mode4_init_check() == ERR:
            return ERR

        self.nc1_cycle = (self.nc1 + 254) // 255
        self._aicore_in_use_select(self.out_w)

        ubsize = 0
        # try to adjust y_steplen
        x_steplen = 8
        y_steplen = ((self.out_h + 7) // 8) * 8
        while (y_steplen > 0):
            ubsize = self._mode4_ubsize(x_steplen, y_steplen)
            if (ubsize > UB_BUFF_MAX):
                y_steplen = y_steplen - 8
            else:
                break

        if (y_steplen <= 0):
            return ERR

        x_steplen = min(((self.xlen_each_core + 7) // 8) * 8,
                        ((255 // (self.win_ylen_max * self.win_xlen_max))) // 8 * 8)
        while (x_steplen > 0):
            ubsize = self._mode4_ubsize(x_steplen, y_steplen)
            if (ubsize > UB_BUFF_MAX):
                x_steplen = x_steplen - 8
            else:
                break

        if (x_steplen <= 0):
            return ERR

        print("ubsize:", ubsize)

        if ((self.win_xlen_max * x_steplen + self.wscale) > 255 and (type == 1)):
            return ERR

        x_steplen_16 = ((x_steplen + 15) // 16) * 16
        y_steplen_16 = ((y_steplen + 15) // 16) * 16
        x_repeat = x_steplen // 8
        y_repeat = y_steplen // 8

        self.x_steplen = x_steplen
        self.x_steplen_16 = x_steplen_16
        self.y_steplen = y_steplen
        self.y_steplen_16 = y_steplen_16
        self.x_repeat = x_repeat
        self.y_repeat = y_repeat

        if (x_steplen - (x_steplen // 8) * 8 != 0 or y_steplen - (y_steplen // 8) * 8 != 0):
            print("ERR:invalid steplen")
            return ERR

        return 0

    def mode4_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            with self.tik_instance.if_scope(index != self.aicore_use - 1):
                self._mode4_compute_each_core(self.xlen_each_core, (index * self.xlen_each_core))
            with self.tik_instance.else_scope():
                self._mode4_compute_each_core(self.xlen_last_core, (index * self.xlen_each_core))

        self.tik_instance.BuildCCE(kernel_name=self.kern_name, inputs=[self.gm_input], outputs=[self.gm_output])

    def _mode4_compute_each_core(self, xlen, xoffset):
        self._common_declare()
        self.ub_out = self.tik_instance.Tensor("float16", (min(255, nc1), 1, x_steplen, 16), name="ub_out",
                                               scope=tik.scope_ubuf)
        self.cur_nc1len = self.tik_instance.Scalar("int32")
        self.pool_start_h = self.tik_instance.Scalar("int32")
        # init to invalid at beginning
        self.pool_start_h.set_as(65535)

        self.tmp3_i32 = self.tik_instance.Scalar("int32")
        self.ub_in = self.tik_instance.Tensor("float16", (n, c1, win_ylen_max, w, 16), name="ub_in",
                                              scope=tik.scope_ubuf)
        if (type == 2):
            self.ub_in2 = self.tik_instance.Tensor("float16", (n, c1, win_ylen_max, win_xlen_max * x_steplen, 16),
                                                   name="ub_in2", scope=tik.scope_ubuf)
        else:
            self.ub_in2 = self.tik_instance.Tensor("float16",
                                                   (n, c1, win_ylen_max, win_xlen_max * x_steplen + self.wscale, 16),
                                                   name="ub_in2", scope=tik.scope_ubuf)

        xcycle = (xlen + x_steplen - 1) // x_steplen
        ycycle = (out_h + y_steplen - 1) // y_steplen
        with self.tik_instance.for_range(0, xcycle) as i_x_cycle:
            self._mode4_xcycle_prepare(xlen, xoffset, i_x_cycle)
            with self.tik_instance.for_range(0, ycycle) as i_ycycle:
                self._mode4_ycycle_prepare(i_ycycle)
                with self.tik_instance.for_range(0, self.cur_hlen) as i_out_y:
                    self._mode4_ystep_prepare(i_out_y)
                    with self.tik_instance.for_range(0, self.nc1_cycle) as i_nc1_cycle:
                        self._mode4_compute_each_loop(i_out_y, i_nc1_cycle)

    def _mode4_xcycle_prepare(self, xlen, xoffset, i_x_cycle):
        self._mode1_xcycle_prepare(xlen, xoffset, i_x_cycle)

    def _mode4_ycycle_prepare(self, i_ycycle):
        self._mode1_ycycle_prepare(i_ycycle)

    def _mode4_cal_data_prepare_fast_for_linear(self):
        self.tmp_i32.set_as(self.ub_x_start[0])
        self.tmp3_i32.set_as(1)
        with self.tik_instance.for_range(1, self.wscale) as i:
            self.tmp2_i32.set_as(self.ub_x_start[i])
            with self.tik_instance.if_scope(self.tmp_i32 == self.tmp2_i32):
                self.tmp3_i32.set_as(self.tmp3_i32 + 1)
            with self.tik_instance.else_scope():
                pass

        with self.tik_instance.if_scope(self.tmp_i32 >= 0):
            # restriction: nc1 * win_ylen_max < 256
            self.tik_instance.vmul(16 * self.tmp3_i32, self.ub_in2, self.ub_in[0, 0, 0, self.tmp_i32, 0],
                                   self.ub_16_f16_val1, nc1 * win_ylen_max, 1, 0, 0,
                                   win_xlen_max * x_steplen, w, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.vector_dup(16 * self.tmp3_i32, self.ub_in2, 0.0, nc1 * win_ylen_max, 1,
                                         win_xlen_max * x_steplen)

        self.tmp2_i32.set_as(self.ub_x_start[self.cur_wlen - 1])
        self.tmp2_i32.set_as(self.tmp2_i32 + win_xlen_max - 1)
        self.tmp2_i32.set_as(self.tmp2_i32 - self.tmp_i32)
        with self.tik_instance.for_range(0, self.tmp2_i32) as i:
            self.tmp_i32.set_as(self.tmp_i32 + 1)
            with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp_i32 < w):
                    self.tik_instance.vmul(16 * self.wscale,
                                           self.ub_in2[0, 0, 0, self.tmp3_i32 + i * self.wscale, 0],
                                           self.ub_in[0, 0, 0, self.tmp_i32, 0], self.ub_16_f16_val1,
                                           nc1 * win_ylen_max, 1, 0, 0, win_xlen_max * x_steplen, w, 0)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

    def _mode4_cal_data_prepare_slow_for_linear(self):
        with self.tik_instance.for_range(0, win_xlen_max) as i_win_tmp:
            with self.tik_instance.for_range(0, self.cur_wlen) as i_tmp:
                self.tmp_i32.set_as(self.ub_x_start[i_tmp])
                self.tmp_i32.set_as(self.tmp_i32 + i_win_tmp)
                with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                    with self.tik_instance.if_scope(self.tmp_i32 < w):
                        # restriction: nc1 * win_ylen_max < 4096
                        self.tik_instance.data_move(self.ub_in2[0, 0, 0, i_win_tmp * x_steplen + i_tmp, 0],
                                                    self.ub_in[0, 0, 0, self.tmp_i32, 0],
                                                    0, nc1 * win_ylen_max, 1, w - 1,
                                                    win_xlen_max * x_steplen - 1, 0)
                    with self.tik_instance.else_scope():
                        pass
                with self.tik_instance.else_scope():
                    pass

    def _mode4_cal_data_prepare_for_linear(self):
        if (self.wscale <= 8 and self.wscale > 1):
            self._mode4_cal_data_prepare_fast_for_linear()
        else:
            self._mode4_cal_data_prepare_slow_for_linear()

    def _mode4_cal_data_prepare_fast_for_nearest(self):
        self.tmp_i32.set_as(self.ub_x_in_round[0])
        self.tmp3_i32.set_as(1)
        with self.tik_instance.for_range(1, self.wscale) as i:
            self.tmp2_i32.set_as(self.ub_x_in_round[i])
            with self.tik_instance.if_scope(self.tmp_i32 == self.tmp2_i32):
                self.tmp3_i32.set_as(self.tmp3_i32 + 1)
            with self.tik_instance.else_scope():
                pass

        with self.tik_instance.if_scope(self.tmp_i32 >= 0):
            # restriction: nc1 * win_ylen_max < 256
            self.tik_instance.vmul(16 * self.tmp3_i32, self.ub_in2, self.ub_in[0, 0, 0, self.tmp_i32, 0],
                                   self.ub_16_f16_val1, nc1 * win_ylen_max, 1, 0, 0,
                                   win_xlen_max * x_steplen + self.wscale, w, 0)
        with self.tik_instance.else_scope():
            pass

        self.tmp2_i32.set_as(self.ub_x_in_round[self.cur_wlen - 1])
        self.tmp2_i32.set_as(self.tmp2_i32 + win_xlen_max - 1)
        self.tmp2_i32.set_as(self.tmp2_i32 - self.tmp_i32)
        with self.tik_instance.for_range(0, self.tmp2_i32) as i:
            self.tmp_i32.set_as(self.tmp_i32 + 1)
            with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp_i32 < w):
                    self.tik_instance.vmul(16 * self.wscale,
                                           self.ub_in2[0, 0, 0, self.tmp3_i32 + i * self.wscale, 0],
                                           self.ub_in[0, 0, 0, self.tmp_i32, 0], self.ub_16_f16_val1,
                                           nc1 * win_ylen_max, 1, 0, 0,
                                           win_xlen_max * x_steplen + self.wscale, w, 0)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

    def _mode4_cal_data_prepare_slow_for_nearest(self):
        with self.tik_instance.for_range(0, win_xlen_max) as i_win_tmp:
            with self.tik_instance.for_range(0, self.cur_wlen) as i_tmp:
                self.tmp_i32.set_as(self.ub_x_in_round[i_tmp])
                self.tmp_i32.set_as(self.tmp_i32 + i_win_tmp)
                with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                    with self.tik_instance.if_scope(self.tmp_i32 < w):
                        # restriction: nc1 * win_ylen_max < 4096
                        self.tik_instance.data_move(self.ub_in2[0, 0, 0, i_win_tmp * x_steplen + i_tmp, 0],
                                                    self.ub_in[0, 0, 0, self.tmp_i32, 0],
                                                    0, nc1 * win_ylen_max, 1, w - 1,
                                                    win_xlen_max * x_steplen + self.wscale - 1, 0)
                    with self.tik_instance.else_scope():
                        pass
                with self.tik_instance.else_scope():
                    pass

    def _mode4_cal_data_prepare_for_nearest(self):
        repeat_tmp = (win_xlen_max * x_steplen + self.wscale) // 8
        remain_tmp = win_xlen_max * x_steplen + self.wscale - 8 * repeat_tmp
        for i in range(0, repeat_tmp):
            # self.ub_out clear
            self.tik_instance.vector_dup(128, self.ub_in2[0, 0, 0, 8 * i, 0], 0.0, nc1, 1,
                                         win_xlen_max * x_steplen + self.wscale)
        if (remain_tmp != 0):
            self.tik_instance.vector_dup(16 * remain_tmp, self.ub_in2[0, 0, 0, 8 * repeat_tmp, 0], 0.0, nc1, 1,
                                         win_xlen_max * x_steplen + self.wscale)

        if (self.wscale <= 8 and self.wscale > 1):
            self._mode4_cal_data_prepare_fast_for_nearest()
        else:
            self._mode4_cal_data_prepare_slow_for_nearest()

    def _mode4_cal_data_prepare(self):
        if (type == 2):
            self._mode4_cal_data_prepare_for_linear()
        else:
            self._mode4_cal_data_prepare_for_nearest()

    def _mode4_indata_prepare(self, i_out_y):
        if (type == 2):
            self.tmp2_i32.set_as(self.ub_y_start[i_out_y])
        else:
            self.tmp2_i32.set_as(self.ub_y_in_round[i_out_y])

        with self.tik_instance.if_scope(self.tmp2_i32 != self.pool_start_h):
            with self.tik_instance.if_scope(self.tmp2_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp2_i32 < h - (win_ylen_max - 1)):
                    self.tik_instance.data_move(self.ub_in, self.gm_input[0, 0, self.tmp2_i32, 0, 0], 0, nc1,
                                                win_ylen_max * w, (h - win_ylen_max) * w, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(self.tmp2_i32 < h):
                        self.tik_instance.data_move(self.ub_in, self.gm_input[0, 0, self.tmp2_i32, 0, 0], 0, nc1,
                                                    (h - self.tmp2_i32) * w, self.tmp2_i32 * w,
                                                    (win_ylen_max - h + self.tmp2_i32) * w, 0)
                    with self.tik_instance.else_scope():
                        pass
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.tmp2_i32 > -1 * win_ylen_max):
                    self.tmp2_i32.set_as((-1) * self.tmp2_i32)
                    self.tik_instance.data_move(self.ub_in[0, 0, self.tmp2_i32, 0, 0], self.gm_input, 0, nc1,
                                                (win_ylen_max - self.tmp2_i32) * w,
                                                (h - win_ylen_max + self.tmp2_i32) * w, self.tmp2_i32 * w, 0)
                with self.tik_instance.else_scope():
                    pass
        with self.tik_instance.else_scope():
            pass

    def _mode4_ystep_prepare(self, i_out_y):
        self._mode4_indata_prepare(i_out_y)

        if (type == 2):
            self.tmp2_i32.set_as(self.ub_y_start[i_out_y])
        else:
            self.tmp2_i32.set_as(self.ub_y_in_round[i_out_y])

        with self.tik_instance.if_scope(self.tmp2_i32 != self.pool_start_h):
            self._mode4_cal_data_prepare()
        with self.tik_instance.else_scope():
            pass

        self.pool_start_h.set_as(self.ub_y_start[i_out_y])
        if (type == 2):
            self.tmp_f16.set_as(self.ub_aady_sum[0, 0, 0, i_out_y, 0])
            # The calculation formula sum(ax*f(ax*dx) * ay*f(ay*dy)) equal to sum(ax*f(ax*dx)) * sum(ay*tf(ay*dy))
            self.tik_instance.vmuls(128, self.ub_aadxy_sum, self.ub_aadx_sum, self.tmp_f16, x_repeat, 1, 1, 8, 8)

            # 1.0 / wsum
            self.tik_instance.vrec(128, self.ub_aadxy_sum, self.ub_aadxy_sum, (self.cur_wlen + 7) // 8, 1, 1, 8, 8)
        else:
            pass

    def _mode4_cal_sum_x_for_linear(self, i_win_y, i_out_y):
        with self.tik_instance.for_range(0, win_xlen_max) as i_win_x:
            self.tmp_f16.set_as(self.ub_aady_win[0, 0, i_win_y, i_out_y, 0])
            # The calculation formula of the following code  ax*f(ax*dx) * ay*f(ay*dy)
            self.tik_instance.vmuls(128, self.ub_aadxy_p_f16, self.ub_aadx_win[0, 0, i_win_x, 0, 0],
                                    self.tmp_f16, x_repeat, 1, 1, 8, 8)
            with self.tik_instance.for_range(0, x_repeat) as i:
                with self.tik_instance.if_scope(self.wscale <= 8 and self.wscale > 1):
                    self.tik_instance.vmla(128, self.ub_out[0, 0, 8 * i, 0],
                                           self.ub_in2[0, 0, i_win_y, i_win_x * self.wscale + 8 * i, 0],
                                           self.ub_aadxy_p_f16[0, 0, 0, 8 * i, 0], self.cur_nc1len, 1,
                                           1, 1, x_steplen, win_ylen_max * win_xlen_max * x_steplen, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vmla(128, self.ub_out[0, 0, 8 * i, 0],
                                           self.ub_in2[0, 0, i_win_y, i_win_x * x_steplen + 8 * i, 0],
                                           self.ub_aadxy_p_f16[0, 0, 0, 8 * i, 0], self.cur_nc1len, 1,
                                           1, 1, x_steplen, win_ylen_max * win_xlen_max * x_steplen, 0)

    def _mode4_compute_each_loop_for_linear(self, i_out_y, i_nc1_cycle):
        # calculate self.cur_nc1len
        with self.tik_instance.if_scope(nc1 - 255 * i_nc1_cycle >= 255):
            self.cur_nc1len.set_as(255)
        with self.tik_instance.else_scope():
            self.cur_nc1len.set_as(nc1 - 255 * i_nc1_cycle)

        for i in range(0, x_repeat):
            # self.ub_out clear
            self.tik_instance.vector_dup(128, self.ub_out[0, 0, 8 * i, 0], 0.0, self.cur_nc1len, 1, x_steplen)

        # calculate output(self.cur_nc1len, 1, 8, 16)
        with self.tik_instance.for_range(0, win_ylen_max) as i_win_y:
            self.tmp_i32.set_as(self.ub_y_start[i_out_y])
            self.tmp_i32.set_as(self.tmp_i32 + i_win_y)
            with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp_i32 < h):
                    self._mode4_cal_sum_x_for_linear(i_win_y, i_out_y)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

        for i in range(0, x_repeat):
            # sum/wsum
            self.tik_instance.vmul(128, self.ub_out[0, 0, 8 * i, 0], self.ub_out[0, 0, 8 * i, 0],
                                   self.ub_aadxy_sum[0, 0, 0, 8 * i, 0], self.cur_nc1len, 1, 1, 1, x_steplen,
                                   x_steplen, 0)

        self.tik_instance.data_move(self.gm_output[255 * i_nc1_cycle * out_h * out_w * 16 +
                                                   (self.start_h + i_out_y) * out_w * 16 + self.start_w * 16],
                                    self.ub_out, 0, self.cur_nc1len, self.cur_wlen, x_steplen - self.cur_wlen,
                                    out_w * out_h - self.cur_wlen, 0)

    def _mode4_compute_each_loop_for_nearest(self, i_out_y, i_nc1_cycle):
        # calculate self.cur_nc1len
        with self.tik_instance.if_scope(nc1 - 255 * i_nc1_cycle >= 255):
            self.cur_nc1len.set_as(255)
        with self.tik_instance.else_scope():
            self.cur_nc1len.set_as(nc1 - 255 * i_nc1_cycle)

        self.tik_instance.data_move(self.gm_output[255 * i_nc1_cycle * out_h * out_w * 16 +
                                                   (self.start_h + i_out_y) * out_w * 16 + self.start_w * 16],
                                    self.ub_in2, 0, self.cur_nc1len, self.cur_wlen,
                                    win_xlen_max * x_steplen + self.wscale - self.cur_wlen,
                                    out_w * out_h - self.cur_wlen, 0)

    def _mode4_compute_each_loop(self, i_out_y, i_nc1_cycle):
        if (type == 2):
            self._mode4_compute_each_loop_for_linear(i_out_y, i_nc1_cycle)
        else:
            self._mode4_compute_each_loop_for_nearest(i_out_y, i_nc1_cycle)

    def _mode2_init(self):
        self.nc1_cycle = (self.nc1 + 254) // 255

        # ubsize check: 132672 + (win_xlen_max + win_ylen_max) * 256 < 240 * 1024
        if (self.win_xlen_max + self.win_ylen_max > 350):
            return ERR

        if (self.out_h * self.out_w) > 65535 or (self.h * self.w) > 65535:
            return ERR

        x_steplen = 8
        x_steplen_16 = ((x_steplen + 15) // 16) * 16
        y_steplen = 8
        y_steplen_16 = ((y_steplen + 15) // 16) * 16
        x_repeat = x_steplen // 8
        y_repeat = y_steplen // 8

        self.x_steplen = x_steplen
        self.x_steplen_16 = x_steplen_16
        self.y_steplen = y_steplen
        self.y_steplen_16 = y_steplen_16
        self.x_repeat = x_repeat
        self.y_repeat = y_repeat

        self._aicore_in_use_select(self.out_w)

        # restriction check
        if x_steplen - (x_steplen // 8) * 8 != 0 or y_steplen - (y_steplen // 8) * 8 != 0:
            print("ERR:invalid steplen")
            return ERR

        return 0

    def mode2_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            with self.tik_instance.if_scope(index != self.aicore_use - 1):
                self._mode2_compute_each_core(self.xlen_each_core, (index * self.xlen_each_core))
            with self.tik_instance.else_scope():
                self._mode2_compute_each_core(self.xlen_last_core, (index * self.xlen_each_core))

        self.tik_instance.BuildCCE(kernel_name=self.kern_name, inputs=[self.gm_input], outputs=[self.gm_output])

    def _mode2_compute_each_core(self, xlen, xoffset):
        self._common_declare()
        self.ub_tmp = self.tik_instance.Tensor("float16", (min(255, nc1), 1, x_steplen, 16), name="ub_tmp",
                                               scope=tik.scope_ubuf)
        self.ub_out = self.tik_instance.Tensor("float16", (min(255, nc1), 1, x_steplen, 16), name="ub_out",
                                               scope=tik.scope_ubuf)
        self.cur_nc1len = self.tik_instance.Scalar("int32")

        xcycle = (xlen + x_steplen - 1) // x_steplen
        ycycle = (out_h + y_steplen - 1) // y_steplen
        with self.tik_instance.for_range(0, xcycle) as i_x_cycle:
            self._mode2_xcycle_prepare(xlen, xoffset, i_x_cycle)
            with self.tik_instance.for_range(0, ycycle) as i_ycycle:
                self._mode2_ycycle_prepare(i_ycycle)
                with self.tik_instance.for_range(0, self.cur_hlen) as i_out_y:
                    self._mode2_ystep_prepare(i_out_y)
                    with self.tik_instance.for_range(0, self.nc1_cycle) as i_nc1_cycle:
                        self._mode2_compute_each_loop(i_out_y, i_nc1_cycle)

    def _mode2_xcycle_prepare(self, xlen, xoffset, i_x_cycle):
        self._mode1_xcycle_prepare(xlen, xoffset, i_x_cycle)

    def _mode2_ycycle_prepare(self, i_ycycle):
        self._mode1_ycycle_prepare(i_ycycle)

    def _mode2_ystep_prepare(self, i_out_y):
        self._mode1_ystep_prepare(i_out_y)

    def _mode2_cal_sum_x_for_linear(self, i_win_y, i_out_y, i_nc1_cycle):
        with self.tik_instance.for_range(0, win_xlen_max) as i_win_x:
            self.tmp_f16.set_as(self.ub_aady_win[0, 0, i_win_y, i_out_y, 0])
            # The calculation formula of the following code ax*f(ax*dx) * ay*f(ay*dy)
            self.tik_instance.vmuls(128, self.ub_aadxy_p_f16, self.ub_aadx_win[0, 0, i_win_x, 0, 0],
                                    self.tmp_f16, x_repeat, 1, 1, 8, 8)

            # prepare data
            with self.tik_instance.for_range(0, self.cur_wlen) as i_out_x:
                self.tmp2_i32.set_as(self.ub_x_start[i_out_x])
                self.tmp2_i32.set_as(self.tmp2_i32 + i_win_x)
                with self.tik_instance.if_scope(self.tmp2_i32 >= 0):
                    with self.tik_instance.if_scope(self.tmp2_i32 < w):
                        self.tik_instance.data_move(self.ub_tmp[0, 0, i_out_x, 0],
                                                    self.gm_input[255 * i_nc1_cycle * h * w * 16 +
                                                                  self.tmp_i32 * w * 16 +
                                                                  self.tmp2_i32 * 16],
                                                    0, self.cur_nc1len, 1, w * h - 1, 7, 0)
                    with self.tik_instance.else_scope():
                        pass
                with self.tik_instance.else_scope():
                    pass

            # calculate in point(x, y) and accumulate to self.ub_out
            self.tik_instance.vmul(128, self.ub_tmp, self.ub_tmp, self.ub_aadxy_p_f16, self.cur_nc1len,
                                   1, 1, 1, 8, 8, 0)
            self.tik_instance.vadd(128, self.ub_out, self.ub_out, self.ub_tmp, self.cur_nc1len, 1, 1, 1,
                                   8, 8, 8)

    def _mode2_compute_each_loop_for_linear(self, i_out_y, i_nc1_cycle):
        # calculate self.cur_nc1len
        with self.tik_instance.if_scope(nc1 - 255 * i_nc1_cycle >= 255):
            self.cur_nc1len.set_as(255)
        with self.tik_instance.else_scope():
            self.cur_nc1len.set_as(nc1 - 255 * i_nc1_cycle)

        # self.ub_out clear
        self.tik_instance.vector_dup(128, self.ub_out, 0.0, self.cur_nc1len, 1, 8)

        # calculate output(self.cur_nc1len, 1, 8, 16)
        with self.tik_instance.for_range(0, win_ylen_max) as i_win_y:
            # calculate y in ub_in and check it
            self.tmp_i32.set_as(self.ub_y_start[i_out_y])
            self.tmp_i32.set_as(self.tmp_i32 + i_win_y)
            with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp_i32 < h):
                    self._mode2_cal_sum_x_for_linear(i_win_y, i_out_y, i_nc1_cycle)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

        # sum * scale
        self.tik_instance.vmul(128, self.ub_out, self.ub_out, self.ub_steplen_16f16, self.cur_nc1len, 1, 1, 1, 8, 8,
                               0)
        # sum/wsum
        self.tik_instance.vmul(128, self.ub_out, self.ub_out, self.ub_aadxy_sum, self.cur_nc1len, 1, 1, 1, 8, 8, 0)

        self.tik_instance.data_move(self.gm_output[255 * i_nc1_cycle * out_h * out_w * 16 +
                                                   (self.start_h + i_out_y) * out_w * 16 + self.start_w * 16],
                                    self.ub_out, 0, self.cur_nc1len, self.cur_wlen, 8 - self.cur_wlen,
                                    out_w * out_h - self.cur_wlen, 0)

    def _mode2_cal_in_xdir_for_nearest(self, i_nc1_cycle):
        with self.tik_instance.for_range(0, self.cur_wlen) as i_out_x:
            self.tmp2_i32.set_as(self.ub_x_in_round[i_out_x])
            with self.tik_instance.if_scope(self.tmp2_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp2_i32 < w):
                    self.tik_instance.data_move(self.ub_out[0, 0, i_out_x, 0],
                                                self.gm_input[255 * i_nc1_cycle * h * w * 16 +
                                                              self.tmp_i32 * w * 16 + self.tmp2_i32 * 16],
                                                0, self.cur_nc1len, 1, w * h - 1, 7, 0)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

    def _mode2_compute_each_loop_for_nearest(self, i_out_y, i_nc1_cycle):
        # calculate self.cur_nc1len
        with self.tik_instance.if_scope(nc1 - 255 * i_nc1_cycle >= 255):
            self.cur_nc1len.set_as(255)
        with self.tik_instance.else_scope():
            self.cur_nc1len.set_as(nc1 - 255 * i_nc1_cycle)

        # self.ub_out clear
        self.tik_instance.vector_dup(128, self.ub_out, 0.0, self.cur_nc1len, 1, 8)
        self.tmp_i32.set_as(self.ub_y_in_round[i_out_y])

        with self.tik_instance.if_scope(self.tmp_i32 >= 0):
            with self.tik_instance.if_scope(self.tmp_i32 < h):
                self._mode2_cal_in_xdir_for_nearest(i_nc1_cycle)
            with self.tik_instance.else_scope():
                pass
        with self.tik_instance.else_scope():
            pass

        self.tik_instance.data_move(self.gm_output[255 * i_nc1_cycle * out_h * out_w * 16 +
                                                   (self.start_h + i_out_y) * out_w * 16 + self.start_w * 16],
                                    self.ub_out, 0, self.cur_nc1len, self.cur_wlen, 8 - self.cur_wlen,
                                    out_w * out_h - self.cur_wlen, 0)

    def _mode2_compute_each_loop(self, i_out_y, i_nc1_cycle):
        if (type == 2):
            self._mode2_compute_each_loop_for_linear(i_out_y, i_nc1_cycle)
        else:
            self._mode2_compute_each_loop_for_nearest(i_out_y, i_nc1_cycle)

    def _mode1_ubsize(self, x_steplen, y_steplen):
        ubsize = self._common_ubsize(x_steplen, y_steplen)

        ubsize += (x_steplen * 16 * 2 * 2)
        ubsize = ubsize * 2
        return ubsize

    def _mode1_init(self):
        out_h = self.out_h
        out_w = self.out_w
        win_xlen_max = self.win_xlen_max
        win_ylen_max = self.win_ylen_max

        x_steplen = (UB_BUFF_MAX // 2 - 651) // (293 + 16 * (win_xlen_max + win_ylen_max))
        x_steplen = min(x_steplen, out_w)
        x_steplen = min(x_steplen, 255 * 8)
        # make sure x_steplen was 8-multiples
        x_steplen = ((x_steplen + 7) // 8) * 8
        x_steplen_16 = ((x_steplen + 15) // 16) * 16
        y_steplen = min(x_steplen, ((out_h + 7) // 8) * 8)
        y_steplen_16 = ((y_steplen + 15) // 16) * 16

        # pick out suitable x_steplen and y_steplen
        while ((x_steplen < ((out_w + 7) // 8) * 8) and (x_steplen < 255 * 8)):
            x_steplen_16 = ((x_steplen + 15) // 16) * 16
            y_steplen = min(x_steplen, ((out_h + 7) // 8) * 8)
            y_steplen_16 = ((y_steplen + 15) // 16) * 16
            ubsize = self._mode1_ubsize(x_steplen, y_steplen)
            if (ubsize < UB_BUFF_MAX):
                x_steplen = x_steplen + 8
            else:
                break

        while (x_steplen > 0):
            x_steplen_16 = ((x_steplen + 15) // 16) * 16
            y_steplen = min(x_steplen, ((out_h + 7) // 8) * 8)
            y_steplen_16 = ((y_steplen + 15) // 16) * 16
            ubsize = self._mode1_ubsize(x_steplen, y_steplen)
            if (ubsize > UB_BUFF_MAX):
                x_steplen = x_steplen - 8
            else:
                break

        if (x_steplen <= 0):
            print("WARN:ub buffer was not enough. Zoom out scale was too bigger to support")
            return ERR

        if (out_w + self.aicore_use - 1) // self.aicore_use < x_steplen:
            x_steplen = (out_w + self.aicore_use - 1) // self.aicore_use
            x_steplen = ((x_steplen + 7) // 8) * 8
            x_steplen_16 = ((x_steplen + 15) // 16) * 16

        x_repeat = x_steplen // 8
        y_repeat = y_steplen // 8

        self.x_steplen = x_steplen
        self.x_steplen_16 = x_steplen_16
        self.y_steplen = y_steplen
        self.y_steplen_16 = y_steplen_16
        self.x_repeat = x_repeat
        self.y_repeat = y_repeat

        self._aicore_in_use_select(out_w)

        # restriction check
        if x_steplen - (x_steplen // 8) * 8 != 0:
            print("ERR:invalid x_steplen")
            return ERR

        return 0

    def mode1_compute(self):
        with self.tik_instance.for_range(0, self.aicore_use, block_num=self.aicore_use) as index:
            with self.tik_instance.if_scope(index != self.aicore_use - 1):
                self._mode1_compute_each_core(self.xlen_each_core, (index * self.xlen_each_core))
            with self.tik_instance.else_scope():
                self._mode1_compute_each_core(self.xlen_last_core, (index * self.xlen_each_core))

        self.tik_instance.BuildCCE(kernel_name=self.kern_name, inputs=[self.gm_input], outputs=[self.gm_output])

    def _mode1_compute_each_core(self, xlen, xoffset):
        self._common_declare()

        self.ub_tmp = self.tik_instance.Tensor("float16", (x_steplen, 16), name="ub_tmp", scope=tik.scope_ubuf)
        self.ub_out = self.tik_instance.Tensor("float16", (x_steplen, 16), name="ub_out", scope=tik.scope_ubuf)

        xcycle = (xlen + x_steplen - 1) // x_steplen
        ycycle = (out_h + y_steplen - 1) // y_steplen
        with self.tik_instance.for_range(0, xcycle) as i_x_cycle:
            self._mode1_xcycle_prepare(xlen, xoffset, i_x_cycle)
            with self.tik_instance.for_range(0, ycycle) as i_ycycle:
                self._mode1_ycycle_prepare(i_ycycle)
                with self.tik_instance.for_range(0, self.cur_hlen) as i_out_y:
                    self._mode1_ystep_prepare(i_out_y)
                    with self.tik_instance.for_range(0, nc1) as i_nc1:
                        self._mode1_compute_each_loop(i_out_y, i_nc1)

    def _cal_in_and_round(self, in_round_i32_ub, in_f32_ub, start_sclr, steplen, f1, f2):
        steplen_16 = ((steplen + 15) // 16) * 16
        with self.tik_instance.for_range(0, steplen) as i_out:
            in_round_i32_ub[i_out].set_as(start_sclr + i_out)

        self._i32tof32(in_f32_ub, in_round_i32_ub, steplen_16, self.ub_steplen_f16, self.ub_steplen_f32,
                       self.ub_steplen_i32)
        self.tik_instance.vmuls(16, in_f32_ub, in_f32_ub, f1, steplen_16 // 16, 1, 1, 2, 2)
        self.tik_instance.vadds(16, in_f32_ub, in_f32_ub, f2 / 2.0, steplen_16 // 16, 1, 1, 2, 2)
        # caculate x_in_round
        self._floor_f32toi32(in_round_i32_ub, in_f32_ub, steplen_16)
        # caculate x_in
        self.tik_instance.vadds(16, in_f32_ub, in_f32_ub, -0.5, steplen_16 // 16, 1, 1, 2, 2)

    def _cal_start_tensor(self, start_i32_ub, in_round_i32_ub, in_f32_ub, steplen_16, r1, a1):
        tmp_i32_ub = self.ub_steplen2_i32
        tmp2_i32_ub = self.ub_steplen_i32
        tmp_f32_ub = self.ub_steplen3_f32

        self.tik_instance.vector_dup(16, tmp_i32_ub, r1, 1, 1, 2)
        # x_in_round - rx
        self.tik_instance.vsub(16, start_i32_ub, in_round_i32_ub, tmp_i32_ub, steplen_16 // 16, 1, 1, 1, 2, 2, 0)

        # consider ax*dx must belong to (-win_factor, win_factor),
        # x must belong to [ceil(x_in - win_factor/ax), floor(x_in + win_factor/ax)].
        # the same to y
        self.tik_instance.vadds(16, tmp_f32_ub, in_f32_ub, (-1.0 * win_factor) / a1, steplen_16 // 16, 1, 1, 2, 2)
        self._ceil_f32toi32(tmp2_i32_ub, tmp_f32_ub, steplen_16)
        self.tik_instance.vmax(16, start_i32_ub, start_i32_ub, tmp2_i32_ub, steplen_16 // 16, 1, 1, 1, 2, 2, 2)

    def _aad_zero_set(self, tmp_16f32_ub, tmp2_16f32_ub, tmp3_16f32_ub, val0_len16_f32_ub, val1_len16_f32_ub,
                      steplen, repeat, a1, dirmax):
        self.tik_instance.data_move(tmp3_16f32_ub, tmp2_16f32_ub, 0, 1, steplen * 2, 0, 0, 0)
        #  make sure aadx == 0 when x < 0
        self.tik_instance.vmuls(64, tmp_16f32_ub, tmp3_16f32_ub, -1.0, repeat, 1, 1, 16, 16)
        self.tik_instance.vmuls(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp3_16f32_ub[0, 0, 0, 4, 0], -1.0, repeat, 1, 1,
                                16, 16)
        self.tik_instance.vmax(64, tmp_16f32_ub, tmp_16f32_ub, val0_len16_f32_ub, repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmax(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp_16f32_ub[0, 0, 0, 4, 0], val0_len16_f32_ub,
                               repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmin(64, tmp_16f32_ub, tmp_16f32_ub, val1_len16_f32_ub, repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmin(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp_16f32_ub[0, 0, 0, 4, 0], val1_len16_f32_ub,
                               repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmuls(64, tmp_16f32_ub, tmp_16f32_ub, -2.0 / a1, repeat, 1, 1, 16, 16)
        self.tik_instance.vmuls(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp_16f32_ub[0, 0, 0, 4, 0], -2.0 / a1, repeat, 1,
                                1, 16, 16)
        self.tik_instance.vadd(64, tmp_16f32_ub, tmp3_16f32_ub, tmp_16f32_ub, repeat, 1, 1, 1, 16, 16, 16)
        self.tik_instance.vadd(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp3_16f32_ub[0, 0, 0, 4, 0],
                               tmp_16f32_ub[0, 0, 0, 4, 0], repeat, 1, 1, 1, 16, 16, 16)

        self.tik_instance.data_move(tmp3_16f32_ub, tmp_16f32_ub, 0, 1, steplen * 2, 0, 0, 0)
        #  make sure aadx == 0 when x > (w-1)
        self.tik_instance.vadds(64, tmp_16f32_ub, tmp3_16f32_ub, 0.9 - dirmax, repeat, 1, 1, 16, 16)
        self.tik_instance.vadds(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp3_16f32_ub[0, 0, 0, 4, 0], 0.9 - dirmax, repeat,
                                1, 1, 16, 16)
        self.tik_instance.vmax(64, tmp_16f32_ub, tmp_16f32_ub, val0_len16_f32_ub, repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmax(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp_16f32_ub[0, 0, 0, 4, 0], val0_len16_f32_ub,
                               repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmin(64, tmp_16f32_ub, tmp_16f32_ub, val1_len16_f32_ub, repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmin(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp_16f32_ub[0, 0, 0, 4, 0], val1_len16_f32_ub,
                               repeat, 1, 1, 0, 16, 16, 0)
        self.tik_instance.vmuls(64, tmp_16f32_ub, tmp_16f32_ub, (-2.0 / a1) - 10000000.0, repeat, 1, 1, 16, 16)
        self.tik_instance.vmuls(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp_16f32_ub[0, 0, 0, 4, 0],
                                (-2.0 / a1) - 10000000.0, repeat, 1, 1, 16, 16)
        self.tik_instance.vadd(64, tmp_16f32_ub, tmp3_16f32_ub, tmp_16f32_ub, repeat, 1, 1, 1, 16, 16, 16)
        self.tik_instance.vadd(64, tmp_16f32_ub[0, 0, 0, 4, 0], tmp3_16f32_ub[0, 0, 0, 4, 0],
                               tmp_16f32_ub[0, 0, 0, 4, 0], repeat, 1, 1, 1, 16, 16, 16)

        self.tik_instance.data_move(tmp3_16f32_ub, tmp_16f32_ub, 0, 1, steplen * 2, 0, 0, 0)

    def _cal_aad_win_and_sum(self, aad1_sum_f16_ub, aad1_win_f16_ub, in_f32_ub, start_i32_ub, steplen, repeat, a1,
                             dirmax, win_len_max):
        tmp_16f16_ub = self.ub_steplen_16f16
        tmp_16f32_ub = self.ub_steplen_16f32
        val1_len16_f16_ub = self.ub_16_f16_val1
        val0_len16_f32_ub, val1_len16_f32_ub = self.ub_16_f32_val0, self.ub_16_f32_val1
        tmp_16i32_ub, tmp2_16i32_ub = self.ub_aadxy_p_i32, self.ub_steplen_16i32
        tmp2_16f32_ub, tmp3_16f32_ub, tmp4_16f32_ub = self.ub_steplen2_16f32, self.ub_xy_16f32, self.ub_xy_in_16f32

        tmp_f32_sclr, tmp_i32_sclr = self.tmp_f32, self.tmp_i32

        # calculate a1*f(a1*d1) and accumulate it
        # ub_aadx_sum clear
        self.tik_instance.vector_dup(128, aad1_sum_f16_ub, 0.0, repeat, 1, 8)

        # x_in
        with self.tik_instance.for_range(0, steplen) as i_out:
            tmp_f32_sclr.set_as(in_f32_ub[i_out])
            self.tik_instance.vector_dup(16, tmp4_16f32_ub[0, 0, 0, i_out, 0], tmp_f32_sclr, 1, 1, 1)

        # x
        with self.tik_instance.for_range(0, steplen) as i_out:
            tmp_i32_sclr.set_as(start_i32_ub[i_out])
            self.tik_instance.vector_dup(16, tmp_16i32_ub[0, 0, 0, i_out, 0], tmp_i32_sclr, 1, 1, 2)
        self._i32tof32(tmp2_16f32_ub, tmp_16i32_ub, (steplen * 16), tmp_16f16_ub, tmp_16f32_ub, tmp2_16i32_ub)
        with self.tik_instance.for_range(0, win_len_max) as i_win:
            self._aad_zero_set(tmp_16f32_ub, tmp2_16f32_ub, tmp3_16f32_ub, val0_len16_f32_ub, val1_len16_f32_ub,
                               steplen, repeat, a1, dirmax)
            # d1
            self.tik_instance.vsub(64, tmp3_16f32_ub, tmp4_16f32_ub, tmp3_16f32_ub, repeat, 1, 1, 1, 16, 16, 16)
            self.tik_instance.vsub(64, tmp3_16f32_ub[0, 0, 0, 4, 0], tmp4_16f32_ub[0, 0, 0, 4, 0],
                                   tmp3_16f32_ub[0, 0, 0, 4, 0], repeat, 1, 1, 1, 16, 16, 16)
            # The calculation formula of the following code  f(a1*d1)
            self.tik_instance.vmuls(64, tmp3_16f32_ub, tmp3_16f32_ub, a1, repeat, 1, 1, 16, 16)  # a1*d1
            self.tik_instance.vmuls(64, tmp3_16f32_ub[0, 0, 0, 4, 0], tmp3_16f32_ub[0, 0, 0, 4, 0], a1, repeat, 1, 1,
                                    16, 16)
            self.tik_instance.vconv(64, "none", aad1_win_f16_ub[0, 0, i_win, 0, 0], tmp3_16f32_ub, repeat, 1, 1, 8,
                                    16)
            self.tik_instance.vconv(64, "none", aad1_win_f16_ub[0, 0, i_win, 4, 0], tmp3_16f32_ub[0, 0, 0, 4, 0],
                                    repeat, 1, 1, 8, 16)

            self.tik_instance.vabs(128, aad1_win_f16_ub[0, 0, i_win, 0, 0], aad1_win_f16_ub[0, 0, i_win, 0, 0], repeat,
                                   1, 1, 8, 8)
            self.tik_instance.vmin(128, aad1_win_f16_ub[0, 0, i_win, 0, 0], aad1_win_f16_ub[0, 0, i_win, 0, 0],
                                   val1_len16_f16_ub, repeat, 1, 1, 0, 8, 8, 0)
            self.tik_instance.vadds(128, aad1_win_f16_ub[0, 0, i_win, 0, 0], aad1_win_f16_ub[0, 0, i_win, 0, 0], -1.0,
                                    repeat, 1, 1, 8, 8)
            self.tik_instance.vabs(128, aad1_win_f16_ub[0, 0, i_win, 0, 0], aad1_win_f16_ub[0, 0, i_win, 0, 0], repeat,
                                   1, 1, 8, 8)

            # accumulate it to aad1_sum_f16_ub
            self.tik_instance.vadd(128, aad1_sum_f16_ub, aad1_sum_f16_ub, aad1_win_f16_ub[0, 0, i_win, 0, 0], repeat, 1,
                                   1, 1, 8, 8, 8)

            self.tik_instance.vadds(64, tmp2_16f32_ub, tmp2_16f32_ub, 1.0, repeat, 1, 1, 16, 16)
            self.tik_instance.vadds(64, tmp2_16f32_ub[0, 0, 0, 4, 0], tmp2_16f32_ub[0, 0, 0, 4, 0], 1.0, repeat, 1, 1,
                                    16, 16)

    def _mode1_xcycle_prepare(self, xlen, xoffset, i_x_cycle):
        # calculate self.cur_wlen
        with self.tik_instance.if_scope(xlen - x_steplen * i_x_cycle >= x_steplen):
            self.cur_wlen.set_as(x_steplen)
        with self.tik_instance.else_scope():
            self.cur_wlen.set_as(xlen - x_steplen * i_x_cycle)

        # calculate x_in_f16 x_in_round
        self.start_w.set_as(xoffset + x_steplen * i_x_cycle)
        self._cal_in_and_round(self.ub_x_in_round, self.ub_x_in_f32, self.start_w, x_steplen, fx, fy)

        if (type == 2):
            self._cal_start_tensor(self.ub_x_start, self.ub_x_in_round, self.ub_x_in_f32, x_steplen_16, rx, ax)
            self._cal_aad_win_and_sum(self.ub_aadx_sum, self.ub_aadx_win, self.ub_x_in_f32,
                                      self.ub_x_start, x_steplen, x_repeat, ax, w, win_xlen_max)
        else:
            pass

    def _mode1_ycycle_prepare(self, i_ycycle):
        # calculate self.cur_hlen
        with self.tik_instance.if_scope(out_h - i_ycycle * y_steplen >= y_steplen):
            self.cur_hlen.set_as(y_steplen)
        with self.tik_instance.else_scope():
            self.cur_hlen.set_as(out_h - i_ycycle * y_steplen)

        # calculate y_in_f16 y_in_round
        self.start_h.set_as(y_steplen * i_ycycle)
        self._cal_in_and_round(self.ub_y_in_round, self.ub_y_in_f32, self.start_h, y_steplen, fy, fx)
        if (type == 2):
            self._cal_start_tensor(self.ub_y_start, self.ub_y_in_round, self.ub_y_in_f32, y_steplen_16, ry, ay)
            self._cal_aad_win_and_sum(self.ub_aady_sum, self.ub_aady_win, self.ub_y_in_f32,
                                      self.ub_y_start, y_steplen, y_repeat, ay, h, win_ylen_max)
        else:
            pass

    def _mode1_ystep_prepare(self, i_out_y):
        if (type == 2):
            self.tmp_f16.set_as(self.ub_aady_sum[0, 0, 0, i_out_y, 0])
            # The calculation formula sum(ax*f(ax*dx) * ay*f(ay*dy)) equal to sum(ax*f(ax*dx)) * sum(ay*tf(ay*dy))
            self.tik_instance.vmuls(128, self.ub_aadxy_sum, self.ub_aadx_sum, self.tmp_f16, x_repeat, 1, 1, 8, 8)

            # avoid to overflow by vrec when 0 < self.ub_aadxy_sum < 1.0 / 65500.
            # The calculation formula of the following code scale = 1.0 / self.ub_aadxy_sum
            # The calculation formula of the following code out = (sum * scale) / (wsum * scale)
            self.tik_instance.vrec(128, self.ub_steplen_16f16, self.ub_aadxy_sum, (self.cur_wlen + 7) // 8, 1, 1, 8, 8)
            self.tik_instance.vmul(128, self.ub_aadxy_sum, self.ub_aadxy_sum, self.ub_steplen_16f16,
                                   (self.cur_wlen + 7) // 8, 1, 1, 1, 8, 8, 8)
            # The calculation formula of the following code 1.0 / wsum
            self.tik_instance.vrec(128, self.ub_aadxy_sum, self.ub_aadxy_sum, (self.cur_wlen + 7) // 8, 1, 1, 8, 8)
        else:
            pass

    def _mode1_cal_sum_x_for_linear(self, i_win_y, i_out_y, i_nc1):
        with self.tik_instance.for_range(0, win_xlen_max) as i_win_x:
            self.tmp_f16.set_as(self.ub_aady_win[0, 0, i_win_y, i_out_y, 0])
            # The calculation formula of the following code ax*f(ax*dx) * ay*f(ay*dy)
            self.tik_instance.vmuls(128, self.ub_aadxy_p_f16, self.ub_aadx_win[0, 0, i_win_x, 0, 0],
                                    self.tmp_f16, x_repeat, 1, 1, 8, 8)

            # prepare data
            with self.tik_instance.for_range(0, self.cur_wlen) as i_out_x:
                self.tmp2_i32.set_as(self.ub_x_start[i_out_x])
                self.tmp2_i32.set_as(self.tmp2_i32 + i_win_x)
                with self.tik_instance.if_scope(self.tmp2_i32 >= 0):
                    with self.tik_instance.if_scope(self.tmp2_i32 < w):
                        self.tik_instance.data_move(self.ub_tmp[i_out_x, 0], self.gm_input[
                            i_nc1 * h * w * 16 + self.tmp_i32 * w * 16 + self.tmp2_i32 * 16], 0, 1, 1,
                                                    0, 0, 0)
                    with self.tik_instance.else_scope():
                        pass
                with self.tik_instance.else_scope():
                    pass

            # calculate in point(x, y) and accumulate to self.ub_out
            self.tik_instance.vmul(128, self.ub_tmp, self.ub_tmp, self.ub_aadxy_p_f16,
                                   (self.cur_wlen + 7) // 8, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadd(128, self.ub_out, self.ub_out, self.ub_tmp, (self.cur_wlen + 7) // 8,
                                   1, 1, 1, 8, 8, 8)

    def _mode1_compute_each_loop_for_linear(self, i_out_y, i_nc1):
        # self.ub_out clear
        self.tik_instance.vector_dup(128, self.ub_out, 0.0, (self.cur_wlen + 7) // 8, 1, 8)
        # calculate output(1, 1, self.cur_wlen, 16)
        with self.tik_instance.for_range(0, win_ylen_max) as i_win_y:
            # calculate y in ub_in and check it
            self.tmp_i32.set_as(self.ub_y_start[i_out_y])
            self.tmp_i32.set_as(self.tmp_i32 + i_win_y)
            with self.tik_instance.if_scope(self.tmp_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp_i32 < h):
                    self._mode1_cal_sum_x_for_linear(i_win_y, i_out_y, i_nc1)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

        # sum * scale
        self.tik_instance.vmul(128, self.ub_out, self.ub_out, self.ub_steplen_16f16, (self.cur_wlen + 7) // 8, 1, 1,
                               1, 8, 8, 8)
        # sum / wsum
        self.tik_instance.vmul(128, self.ub_out, self.ub_out, self.ub_aadxy_sum, (self.cur_wlen + 7) // 8, 1, 1, 1,
                               8, 8, 8)

        self.tik_instance.data_move(
            self.gm_output[i_nc1 * out_h * out_w * 16 + (self.start_h + i_out_y) * out_w * 16 + self.start_w * 16],
            self.ub_out, 0, 1, self.cur_wlen, x_steplen - self.cur_wlen, out_w - self.cur_wlen, 0)

    def _mode1_cal_in_xdir_for_nearest(self, i_nc1):
        with self.tik_instance.for_range(0, self.cur_wlen) as i_out_x:
            self.tmp2_i32.set_as(self.ub_x_in_round[i_out_x])
            with self.tik_instance.if_scope(self.tmp2_i32 >= 0):
                with self.tik_instance.if_scope(self.tmp2_i32 < w):
                    self.tik_instance.data_move(self.ub_out[i_out_x, 0], self.gm_input[
                        i_nc1 * h * w * 16 + self.tmp_i32 * w * 16 + self.tmp2_i32 * 16], 0, 1, 1, 0, 0, 0)
                with self.tik_instance.else_scope():
                    pass
            with self.tik_instance.else_scope():
                pass

    def _mode1_compute_each_loop_for_nearest(self, i_out_y, i_nc1):
        # self.ub_out clear
        self.tik_instance.vector_dup(128, self.ub_out, 0.0, (self.cur_wlen + 7) // 8, 1, 8)
        self.tmp_i32.set_as(self.ub_y_in_round[i_out_y])
        with self.tik_instance.if_scope(self.tmp_i32 >= 0):
            with self.tik_instance.if_scope(self.tmp_i32 < h):
                self._mode1_cal_in_xdir_for_nearest(i_nc1)
            with self.tik_instance.else_scope():
                pass
        with self.tik_instance.else_scope():
            pass

        self.tik_instance.data_move(
            self.gm_output[i_nc1 * out_h * out_w * 16 + (self.start_h + i_out_y) * out_w * 16 + self.start_w * 16],
            self.ub_out, 0, 1, self.cur_wlen, x_steplen - self.cur_wlen, out_w - self.cur_wlen, 0)

    def _mode1_compute_each_loop(self, i_out_y, i_nc1):
        if (type == 2):
            self._mode1_compute_each_loop_for_linear(i_out_y, i_nc1)
        else:
            self._mode1_compute_each_loop_for_nearest(i_out_y, i_nc1)

    def tik_output_debug(self):
        data_np_a = np.ones([n, (c + 15) // 16, h, w, 16]).astype(np.float)
        data_a_tran = data_np_a.astype(np.float16)
        print("data_in:", data_a_tran)

        feed_dict = {
            "gm_input": data_a_tran,
        }

        out, = self.tik_instance.tikdb.start_debug(feed_dict, False)
        print("out", out)


def resample(x, y, height, width, antialias, type, kernel_name="resample", test=False):
    """
    zoom out or zoom in at the H/W dimension.
    :param x: input data
    :param y: output data
    :param height: output H
    :param width: output W
    :param antialias: the antialias
    :param type: the type of resample
    """
    shape = x.get("shape")
    out_shape = y.get("shape")
    obj = tik_resample(shape, type, antialias, out_shape[2], out_shape[3], kernel_name)

    obj.tiling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {
        1: obj.mode1_compute,
        2: obj.mode2_compute,
        4: obj.mode4_compute
    }

    switch[obj.mode]()
    if not test:
        return 0

    obj.tik_output_debug()
    return 0
