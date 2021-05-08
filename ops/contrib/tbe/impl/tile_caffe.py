from te import tik
import os
import numpy as np

G_UB_CAPACITY = 240 * 1024 // 2


class TileCaffe:
    def __init__(self, input_shape, ori_shape, output_shape, axis, tiles, KERNEL_NAME):
        self.tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
        self.kernel_name = KERNEL_NAME

        self.src_n = input_shape[0]
        self.src_c = ori_shape[1]
        self.src_h = input_shape[2]
        self.src_w = input_shape[3]

        self.dst_n = output_shape[0]
        self.dst_h = output_shape[2]
        self.dst_w = output_shape[3]

        self.axis = axis
        self.tiles = tiles
        self.src_c16 = input_shape[1]
        self.dst_c16 = output_shape[1]

        self.src_gm = self.tik_inst.Tensor("float16",
                                           (self.src_n, self.src_c16, self.src_h, self.src_w, 16),
                                           tik.scope_gm, "src_gm")
        self.dst_gm = self.tik_inst.Tensor("float16",
                                           (self.dst_n, self.dst_c16, self.dst_h, self.dst_w, 16),
                                           tik.scope_gm, "dst_gm")

    def compute(self):
        if self.axis == 1 and self.src_c % 16 != 0:
            self._cal_notalign()
        else:
            self._cal_align()
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.src_gm],
                               outputs=[self.dst_gm], enable_l2=True)
        return self.tik_inst

    def _transpose_general(self, src, dst, h, w):
        if h > w:
            repeat_time = h // 16
            src_rep_stride = 0 if repeat_time == 1 else w
            dst_rep_stride = 0 if repeat_time == 1 else 1

            with self.tik_inst.for_range(0, w // 16) as loop_index:
                src_list = [src[i * w + loop_index * 16] for i in range(16)]
                dst_list = [dst[i * h + loop_index * 16 * h] for i in range(16)]
                self.tik_inst.vnchwconv(True, True, dst_list, src_list, repeat_time, dst_rep_stride,
                                        src_rep_stride)
        else:
            repeat_time = w // 16
            src_rep_stride = 0 if repeat_time == 1 else 1
            dst_rep_stride = 0 if repeat_time == 1 else h

            with self.tik_inst.for_range(0, h // 16) as loop_index:
                src_list = [src[i * w + loop_index * 16 * w] for i in range(16)]
                dst_list = [dst[i * h + loop_index * 16] for i in range(16)]
                self.tik_inst.vnchwconv(True, True, dst_list, src_list, repeat_time, dst_rep_stride,
                                        src_rep_stride)

    def _trans_cal(self, n_cor, h_cor, addtime, zero, addlost, infor_time, in_offset,
                   tiles_ub_trans, tiles_ub, flag):
        with self.tik_inst.for_range(0, infor_time) as intimes:
            self.tik_inst.data_move(tiles_ub_trans[intimes, 0],
                                    self.src_gm[n_cor, 0, h_cor, in_offset + intimes, 0], 0,
                                    self.src_c16, 1, self.src_h * self.src_w - 1, 0)
        # transconv
        self._transpose_general(tiles_ub_trans, tiles_ub, 16, self.dst_c16 * 16)
        # tile
        with self.tik_inst.for_range(0, self.tiles) as tile_time:
            with self.tik_inst.if_scope(addtime > 0):
                self.tik_inst.vadds(128, tiles_ub[tile_time * self.src_c * 16], tiles_ub, zero,
                                    addtime, 1, 1, 8, 8)
            with self.tik_inst.else_scope():
                pass
            if addlost > 0:
                self.tik_inst.vadds(16, tiles_ub[tile_time * self.src_c * 16 + addtime * 128],
                                    tiles_ub[addtime * 128], zero, addlost // 16, 0, 0, 1, 1)
        # transconv
        self._transpose_general(tiles_ub, tiles_ub_trans, self.dst_c16 * 16, 16)

        # output
        if flag == 0:
            with self.tik_inst.for_range(0, self.dst_c16) as outtimes:
                self.tik_inst.data_move(self.dst_gm[n_cor, outtimes, h_cor, in_offset, 0],
                                        tiles_ub_trans[0, 16 * outtimes], 0, 16, 1,
                                        self.dst_c16 - 1, 0)
        else:
            with self.tik_inst.for_range(0, infor_time) as outtimes:
                self.tik_inst.data_move(self.dst_gm[n_cor, 0, h_cor, in_offset + outtimes, 0],
                                        tiles_ub_trans[outtimes, 0], 0, self.dst_c16, 1, 0,
                                        self.src_h * self.src_w - 1)

    def _cal_notalign(self):
        addsum = self.src_c * 16
        addti = addsum // 128
        addlost = addsum % 128

        tiles_ub = self.tik_inst.Tensor("float16", (self.dst_c16 * 16, 16), tik.scope_ubuf,
                                        "tiles_ub")
        tiles_ub_trans = self.tik_inst.Tensor("float16", (16, self.dst_c16 * 16), tik.scope_ubuf,
                                              "tiles_ub_trans")
        zero = self.tik_inst.Scalar("float16")
        addtime = self.tik_inst.Scalar("int32")
        addtime.set_as(addti)
        zero.set_as(0.0)
        with self.tik_inst.for_range(0, self.src_n) as n_cor:
            with self.tik_inst.for_range(0, self.src_h) as h_cor:
                with self.tik_inst.for_range(0, self.src_w // 16) as w_cor:
                    self._trans_cal(n_cor, h_cor, addtime, zero, addlost, 16, w_cor * 16,
                                    tiles_ub_trans, tiles_ub, 0)
                # if has width has remainders
                with self.tik_inst.if_scope(self.src_w % 16 > 0):
                    self._trans_cal(n_cor, h_cor, addtime, zero, addlost, self.src_w % 16,
                                    (self.src_w // 16) * 16, tiles_ub_trans, tiles_ub, 1)
                with self.tik_inst.else_scope():
                    pass

    def _get_once_tranport(self):
        once_tranport = 0
        if self.axis == 0:
            once_tranport = self.src_n * self.src_c16 * self.src_h * self.src_w * 16
        if self.axis == 1:
            once_tranport = self.src_c16 * self.src_h * self.src_w * 16
        if self.axis == 2:
            once_tranport = self.src_h * self.src_w * 16
        if self.axis == 3:
            once_tranport = self.src_w * 16
        return once_tranport

    def _small_than_once(self, once_tranport):
        dupli = once_tranport // G_UB_CAPACITY
        mod = once_tranport % G_UB_CAPACITY
        for_time = self.src_n * self.src_c16 * self.src_h * self.src_w * 16 // once_tranport
        dst_ub = self.tik_inst.Tensor("float16", (G_UB_CAPACITY,), tik.scope_ubuf, "dst_ub")

        with self.tik_inst.for_range(0, for_time) as time:
            if mod != 0:
                self.tik_inst.data_move(dst_ub,
                                        self.src_gm[time * once_tranport + dupli * G_UB_CAPACITY],
                                        0, 1, G_UB_CAPACITY // 16, 0, 0)
                with self.tik_inst.for_range(0, self.tiles) as til:
                    self.tik_inst.data_move(self.dst_gm[once_tranport * (
                            til + time * (self.tiles)) + dupli * G_UB_CAPACITY], dst_ub, 0, 1,
                                            mod // 16, 0, 0)
            with self.tik_inst.for_range(0, dupli) as du:
                self.tik_inst.data_move(dst_ub,
                                        self.src_gm[time * once_tranport + du * G_UB_CAPACITY], 0,
                                        1, G_UB_CAPACITY // 16, 0, 0)
                with self.tik_inst.for_range(0, self.tiles) as til:
                    self.tik_inst.data_move(self.dst_gm[once_tranport * (
                            til + time * (self.tiles)) + du * G_UB_CAPACITY], dst_ub, 0, 1,
                                            G_UB_CAPACITY // 16, 0, 0)

    def _big_than_once(self, once_tranport):
        UB_available = (G_UB_CAPACITY // once_tranport) * once_tranport
        total_num = self.src_n * self.src_c16 * self.src_h * self.src_w * 16
        TimesOfDstUB = total_num // UB_available
        lastOfDstUB = total_num % UB_available

        # inputsize smaller than UB
        if TimesOfDstUB == 0:
            self.big_mode1(total_num, once_tranport)

        # inputsize bigger than UB and once_transport smaller than UB
        if TimesOfDstUB > 0 and lastOfDstUB == 0:
            self.big_mode2(TimesOfDstUB, UB_available, once_tranport)

        if TimesOfDstUB > 0 and lastOfDstUB > 0:
            self.big_mode3(UB_available, TimesOfDstUB, lastOfDstUB, once_tranport)

    def big_mode1(self, total_num, once_tranport):
        dst_ub = self.tik_inst.Tensor("float16", (total_num,), tik.scope_ubuf, "dst_ub")
        self.tik_inst.data_move(dst_ub, self.src_gm, 0, 1, total_num // 16, 0, 0)
        with self.tik_inst.for_range(0, self.tiles) as til:
            self.tik_inst.data_move(self.dst_gm[til * once_tranport], dst_ub, 0,
                                    self.src_n * self.src_c16 * self.src_h * self.src_w * 16 //
                                    once_tranport,
                                    once_tranport // 16, 0, once_tranport * (self.tiles - 1) // 16)

    def big_mode2(self, TimesOfDstUB, UB_available, once_tranport):
        dst_ub = self.tik_inst.Tensor("float16", (UB_available,), tik.scope_ubuf, "dst_ub")

        with self.tik_inst.for_range(0, TimesOfDstUB) as time:
            self.tik_inst.data_move(dst_ub, self.src_gm[UB_available * time], 0, 1,
                                    UB_available // 16, 0, 0)
            with self.tik_inst.for_range(0, self.tiles) as til:
                self.tik_inst.data_move(
                    self.dst_gm[UB_available * time * self.tiles + til * once_tranport], dst_ub, 0,
                    UB_available // once_tranport, once_tranport // 16, 0,
                    once_tranport * (self.tiles - 1) // 16)

    def big_mode3(self, UB_available, TimesOfDstUB, lastOfDstUB, once_tranport):
        dst_ub = self.tik_inst.Tensor("float16", (UB_available,), tik.scope_ubuf, "dst_ub")
        time = self.tik_inst.Scalar("int32")
        with self.tik_inst.for_range(0, TimesOfDstUB + 1) as time_increase:
            time.set_as(TimesOfDstUB - time_increase)
            with self.tik_inst.if_scope(time == TimesOfDstUB):
                self.tik_inst.data_move(dst_ub, self.src_gm[UB_available * time], 0, 1,
                                        lastOfDstUB // 16, 0, 0)
                with self.tik_inst.for_range(0, self.tiles) as til:
                    self.tik_inst.data_move(
                        self.dst_gm[UB_available * time * self.tiles + til * once_tranport], dst_ub,
                        0, lastOfDstUB // once_tranport, once_tranport // 16, 0,
                        once_tranport * (self.tiles - 1) // 16)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(dst_ub, self.src_gm[UB_available * time], 0, 1,
                                        UB_available // 16, 0, 0)
                with self.tik_inst.for_range(0, self.tiles) as til:
                    self.tik_inst.data_move(
                        self.dst_gm[UB_available * time * self.tiles + til * once_tranport], dst_ub,
                        0, UB_available // once_tranport, once_tranport // 16, 0,
                        once_tranport * (self.tiles - 1) // 16)

    def _cal_align(self):
        once_tranport = self._get_once_tranport()

        if G_UB_CAPACITY > once_tranport:
            self._big_than_once(once_tranport)
        else:
            self._small_than_once(once_tranport)

    def tiling_mode_select(self):
        self.mode = 1

    def global_init(self):
        pass

    def tik_output_debug(self):
        data_shape = (self.src_n, self.src_c, self.src_h, self.src_w)
        data_path = './inputdata/tile_' + str(self.src_n) + '_' + str(self.src_c) + '_' + str(
            self.src_h) + '_' + str(self.src_w) + '_input1_1.bin'

        if os.path.exists(data_path):
            src_data = np.fromfile(data_path, dtype='float32').reshape(data_shape)
        else:
            print("not exsist")
            src_data = np.random.rand(self.src_n, self.src_c16, self.src_h, self.src_w, 16).astype(
                np.float16)

        feed_dict = {'src_gm': src_data}
        ret, = self.tik_inst.tikdb.start_debug(feed_dict=feed_dict, interactive=True)

        output_path = "./res_stand/res_tile_" + str(self.src_n) + '_' + str(self.src_c) + '_' + str(
            self.src_h) + '_' + str(self.src_w) + "_1.bin"
        ret.tofile(output_path)


def tile_caffe(x, y, axis, tiles, KERNEL_NAME='tile', test=False):
    '''
    Copies data in a specified dimension and increases the dimension size to specified times.
    Args:
        param self.axis: specified dimension
        param self.tiles: increases time
        param KERNEL_NAME: the kernel_name of tik function

    '''
    input_shape = x.get("shape")
    ori_shape = x.get("ori_shape")
    output_shape = y.get("shape")

    obj = TileCaffe(input_shape, ori_shape, output_shape, axis, tiles, KERNEL_NAME)
    obj.tiling_mode_select()
    if obj.mode == 0:
        raise RuntimeError("can not select a valid tiling mode.")

    obj.global_init()

    switch = {1: obj.compute}

    switch[obj.mode]()
    if not test:
        return 0

    return obj.tik_output_debug()
