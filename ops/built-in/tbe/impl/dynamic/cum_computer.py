from numpy import block
from impl.common_util import get_data_size
from impl.constant_util import BLOCK_SIZE, VECTOR_BYTE_SIZE
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_context


class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 16
    MAX_INT64 = 2**64 - 1
    NUM_64 = 64
    MAX_BURST_LEN = 65535
    DEFAULT_BURST_LEN = 8
    MAX_REPEAT_STRIDE = 8
    # a vector can cal 255 repeat
    MAX_REPEAT_TIMES = 255
    # a vector can cal 8 block
    MAX_BLOCK_NUMBER = 8
    MAX_COMPUTE_BLOCK = MAX_BLOCK_NUMBER * MAX_REPEAT_TIMES
    MAX_COMPUTE_SIZE = BLOCK_SIZE * MAX_REPEAT_TIMES * MAX_BLOCK_NUMBER


class CumComputer:

    def __init__(self, input_x, axis, y, exclusive, reverse, kernel_name):
        self.dtype_x = input_x.get("dtype")
        self.shape_axis = axis.get("shape")
        self.dtype_axis = axis.get("dtype")
        self.exclusive = exclusive
        self.reverse = reverse
        self.kernel_name = kernel_name

        self.tik_instance = tik.Tik()
        self.tik_profiling = tik.Dprofile()

        #get ai_core num
        self.ai_core_num = self.tik_profiling.get_aicore_num()
        #get ub size
        self.ub_size_bytes = self.tik_profiling.get_unified_buffer_size()
        self.dtype_size = get_data_size(self.dtype_x)

        self.input_x_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64,),
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.output_gm = self.tik_instance.Tensor(self.dtype_x, (Constant.MAX_INT64,),
                                                  name="output_gm",
                                                  scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor("int64", (Constant.TILING_ARG_NUM,),
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        self._init_tiling_scalars()
        self._cum_compute_tiling()

    def cum_computer(self):
        with self.tik_instance.if_scope(self.tiling_mode == 0):
            self._compute_mode_default()
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            self._compute_mode_small_inner()

        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.ai_core_num,
            "ub_size": self.ub_size_bytes
        })
        axis = self.tik_instance.Tensor(self.dtype_axis, self.shape_axis, name="axis", scope=tik.scope_gm)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_x_gm, axis],
                                   outputs=[self.output_gm],
                                   flowtable=[self.tiling_gm])

        return self.tik_instance

    def _init_tiling_scalars(self):
        self.tiling_mode = self.tik_instance.Scalar("int64", name="tiling_num_act_core", init_value=0)
        self.tiling_num_act_core = self.tik_instance.Scalar("int64", name="tiling_num_act_core", init_value=0)
        self.tiling_num_outer_total = self.tik_instance.Scalar("int64", name="tiling_num_outer_total", init_value=0)
        self.tiling_ceil_outer_loop_times = self.tik_instance.Scalar("int64",
                                                                     name="tiling_ceil_outer_loop_times",
                                                                     init_value=0)
        self.tiling_floor_outer_loop_times = self.tik_instance.Scalar("int64",
                                                                      name="tiling_floor_outer_loop_times",
                                                                      init_value=0)
        self.tiling_num_outer_cores_tail = self.tik_instance.Scalar("int64",
                                                                    name="tiling_num_outer_cores_tail",
                                                                    init_value=0)
        self.tiling_nums_per_outer_loop = self.tik_instance.Scalar("int64",
                                                                   name="tiling_nums_per_outer_loop",
                                                                   init_value=0)
        self.tiling_inner_floor_loop_times = self.tik_instance.Scalar("int64",
                                                                      name="tiling_inner_floor_loop_times",
                                                                      init_value=0)
        self.tiling_num_per_inner_loop = self.tik_instance.Scalar("int64",
                                                                  name="tiling_num_each_inner_loop",
                                                                  init_value=0)
        self.tiling_num_inner_last_loop = self.tik_instance.Scalar("int64",
                                                                   name="tiling_num_inner_last_loop",
                                                                   init_value=0)
        self.tiling_num_per_core = self.tik_instance.Scalar("int64", name="tiling_num_per_core", init_value=0)
        self.tiling_offset_back_loop = self.tik_instance.Scalar("int64", name="tiling_offset_back_loop", init_value=0)
        self.tiling_axis_shape = self.tik_instance.Scalar("int64", name="tiling_axis_shape", init_value=0)
        self.tiling_inner_tail_repeat_times = self.tik_instance.Scalar("int64",
                                                                       name="tiling_inner_tail_repeat_times",
                                                                       init_value=0)
        self.tiling_equal_shape_before_axis = self.tik_instance.Scalar("int64",
                                                                       name="tiling_equal_shape_before_axis",
                                                                       init_value=0)
        self.tiling_equal_shape_after_axis = self.tik_instance.Scalar("int64",
                                                                      name="tiling_equal_shape_after_axis",
                                                                      init_value=0)

    def _request_two_ub(self, input_1_size, input_2_size):
        """
        request ub
        """
        # ub tensor
        input_1_ub = self.tik_instance. \
            Tensor(self.dtype_x,
                    (input_1_size,),
                    name="input_x_ub",
                    scope=tik.scope_ubuf)
        input_2_ub = self.tik_instance. \
            Tensor(self.dtype_x,
                    (input_2_size,),
                    name="last_res",
                    scope=tik.scope_ubuf)
        return input_1_ub, input_2_ub

    def _cum_compute_tiling(self):
        """
        The function of tiling
        """
        self.tiling_ub = self.tik_instance.Tensor("int64", (16,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 4, 0, 0)
        self.tiling_mode.set_as(self.tiling_ub[0])
        self.tiling_num_act_core.set_as(self.tiling_ub[1])
        self.tiling_num_outer_total.set_as(self.tiling_ub[2])
        self.tiling_ceil_outer_loop_times.set_as(self.tiling_ub[3])
        self.tiling_floor_outer_loop_times.set_as(self.tiling_ub[4])
        self.tiling_num_outer_cores_tail.set_as(self.tiling_ub[5])
        self.tiling_nums_per_outer_loop.set_as(self.tiling_ub[6])
        self.tiling_inner_floor_loop_times.set_as(self.tiling_ub[7])
        self.tiling_num_per_inner_loop.set_as(self.tiling_ub[8])
        self.tiling_num_inner_last_loop.set_as(self.tiling_ub[9])
        self.tiling_num_per_core.set_as(self.tiling_ub[10])
        self.tiling_offset_back_loop.set_as(self.tiling_ub[11])
        self.tiling_axis_shape.set_as(self.tiling_ub[12])
        self.tiling_inner_tail_repeat_times.set_as(self.tiling_ub[13])
        self.tiling_equal_shape_before_axis.set_as(self.tiling_ub[14])
        self.tiling_equal_shape_after_axis.set_as(self.tiling_ub[15])

    def _dup_0_ub(self, input_ub, repeat_time):
        """
        dup 0 ub
        """
        mask = VECTOR_BYTE_SIZE // self.dtype_size
        self.tik_instance.vector_dup(mask, input_ub, 0, repeat_time, 1, Constant.MAX_REPEAT_STRIDE)

    def _gm_2_ub(self, input_x_ub, gm_idx, burst):
        """
        move data from input gm to ub 
        """
        with self.tik_instance.if_scope(burst > 0):
            self.tik_instance.data_move(input_x_ub, self.input_x_gm[gm_idx], 0, 1, burst, 0, 0)

    def _ub_2_gm(self, src_ub, gm_idx, burst):
        """
        move data from ub to output gm
        """
        with self.tik_instance.if_scope(burst > 0):
            self.tik_instance.data_move(self.output_gm[gm_idx], src_ub, 0, 1, burst, 0, 0)

    def _ub_2_gm_last_block_back(self, src_ub, gm_init_idx, last_block_offset, data_len):
        """
        special move for last 32B, addr back
        """
        last_block_gm_idx = gm_init_idx + last_block_offset
        last_block = self.tik_instance.Tensor(self.dtype_x, (BLOCK_SIZE // self.dtype_size,), tik.scope_ubuf,
                                              "last_block")

        self.tik_instance.data_move(last_block,
                                    self.output_gm[last_block_gm_idx - BLOCK_SIZE // self.dtype_size + data_len], 0, 1,
                                    1, 0, 0)
        tmp_scalar = self.tik_instance.Scalar(self.dtype_x)
        with self.tik_instance.for_range(0, data_len) as i:
            tmp_scalar.set_as(src_ub[last_block_offset + i])
            last_block[BLOCK_SIZE // self.dtype_size - data_len + i].set_as(tmp_scalar)
        self.tik_instance.data_move(self.output_gm[last_block_gm_idx - BLOCK_SIZE // self.dtype_size + data_len],
                                    last_block, 0, 1, 1, 0, 0)

    def _handle_first_axis_last_block(self, last_res, gm_index_init, burst_len, last_block_offset, last_block_len):
        if self.exclusive:
            self._dup_0_ub(last_res, self.tiling_inner_tail_repeat_times)
            self._ub_2_gm(last_res, gm_index_init, burst_len)
            self._ub_2_gm_last_block_back(last_res, gm_index_init, last_block_offset, last_block_len)
            self._gm_2_ub(last_res, gm_index_init, burst_len + 1)
        else:
            self._gm_2_ub(last_res, gm_index_init, burst_len + 1)
            self._ub_2_gm(last_res, gm_index_init, burst_len)
            self._ub_2_gm_last_block_back(last_res, gm_index_init, last_block_offset, last_block_len)

    def _handle_other_axis_last_block(self, input_x_ub, last_res, gm_idx, burst_len, last_block_offset, last_block_len):
        if self.exclusive:
            self._ub_2_gm(last_res, gm_idx, burst_len)
            self._ub_2_gm_last_block_back(last_res, gm_idx, last_block_offset, last_block_len)
            self._gm_2_ub(input_x_ub, gm_idx, burst_len + 1)
            self._cal_data(input_x_ub, last_res, self.tiling_inner_tail_repeat_times)
        else:
            self._gm_2_ub(input_x_ub, gm_idx, burst_len + 1)
            self._cal_data(input_x_ub, last_res, self.tiling_inner_tail_repeat_times)
            self._ub_2_gm(last_res, gm_idx, burst_len)
            self._ub_2_gm_last_block_back(last_res, gm_idx, last_block_offset, last_block_len)

    def _handle_first_axis(self, last_res, gm_index_init, burst_len):
        """
        handle first axis
        if exclusive, dup 0 to first axis
        else move data from input_gm to output_gm
        """
        if self.exclusive:
            self._dup_0_ub(last_res, self.tiling_inner_tail_repeat_times)
            self._ub_2_gm(last_res, gm_index_init, burst_len)
            self._gm_2_ub(last_res, gm_index_init, burst_len)
        else:
            self._gm_2_ub(last_res, gm_index_init, burst_len)
            self._ub_2_gm(last_res, gm_index_init, burst_len)

    def _handle_other_aixs(self, input_x_ub, last_res, gm_idx, burst_len, cal_repeat_time):
        """
        handle other axis
        `last_res = input_x + last_res`
        if exclusive, move last_res to gm then cal
        else cal then move to gm
        """
        if self.exclusive:
            self._ub_2_gm(last_res, gm_idx, burst_len)
            self._gm_2_ub(input_x_ub, gm_idx, burst_len)
            self._cal_data(input_x_ub, last_res, cal_repeat_time)
        else:
            self._gm_2_ub(input_x_ub, gm_idx, burst_len)
            self._cal_data(input_x_ub, last_res, cal_repeat_time)
            self._ub_2_gm(last_res, gm_idx, burst_len)

    def _handle_inner_tail(self, outer_loop_index):
        input_x_ub, last_res = self._request_two_ub(Constant.MAX_COMPUTE_SIZE // self.dtype_size,
                                                    Constant.MAX_COMPUTE_SIZE // self.dtype_size)

        burst_len = self.tiling_num_inner_last_loop * self.dtype_size // BLOCK_SIZE
        last_block_offset = burst_len * BLOCK_SIZE // self.dtype_size
        last_block_len = self.tiling_num_inner_last_loop * self.dtype_size % BLOCK_SIZE // self.dtype_size
        gm_index_init = outer_loop_index * self.tiling_nums_per_outer_loop + \
                        self.tiling_inner_floor_loop_times * self.tiling_num_per_inner_loop
        if self.reverse:
            gm_index_init += (self.tiling_axis_shape - 1) * self.tiling_equal_shape_after_axis

        with self.tik_instance.if_scope(last_block_len > 0):
            self._handle_first_axis_last_block(last_res, gm_index_init, burst_len, last_block_offset, last_block_len)
        with self.tik_instance.else_scope():
            self._handle_first_axis(last_res, gm_index_init, burst_len)

        with self.tik_instance.for_range(1, self.tiling_axis_shape) as axis_cycle:
            if self.reverse:
                gm_idx = gm_index_init - axis_cycle * self.tiling_equal_shape_after_axis
            else:
                gm_idx = gm_index_init + axis_cycle * self.tiling_equal_shape_after_axis

            with self.tik_instance.if_scope(last_block_len > 0):
                self._handle_other_axis_last_block(input_x_ub, last_res, gm_idx, burst_len, last_block_offset,
                                                   last_block_len)
            with self.tik_instance.else_scope():
                self._handle_other_aixs(input_x_ub, last_res, gm_idx, burst_len, self.tiling_inner_tail_repeat_times)

    def _handle_inner_loop(self, outer_loop_index, inner_loop_index):
        input_x_ub, last_res = self._request_two_ub(Constant.MAX_COMPUTE_SIZE // self.dtype_size,
                                                    Constant.MAX_COMPUTE_SIZE // self.dtype_size)
        gm_index_init = (outer_loop_index * self.tiling_nums_per_outer_loop) + (inner_loop_index *
                                                                                self.tiling_num_per_inner_loop)
        if self.reverse:
            gm_index_init += (self.tiling_axis_shape - 1) * self.tiling_equal_shape_after_axis

        self._handle_first_axis(last_res, gm_index_init, Constant.MAX_COMPUTE_BLOCK)
        with self.tik_instance.for_range(1, self.tiling_axis_shape) as axis_cycle:
            if self.reverse:
                gm_idx = gm_index_init - axis_cycle * self.tiling_equal_shape_after_axis
            else:
                gm_idx = gm_index_init + axis_cycle * self.tiling_equal_shape_after_axis
            self._handle_other_aixs(input_x_ub, last_res, gm_idx, Constant.MAX_COMPUTE_BLOCK, Constant.MAX_REPEAT_TIMES)

    def _handle_per_loop(self, outer_loop_index):
        with self.tik_instance.if_scope(self.tiling_num_inner_last_loop != 0):
            self._handle_inner_tail(outer_loop_index)
        with self.tik_instance.for_range(0, self.tiling_inner_floor_loop_times) as inner_loop_cycle:
            self._handle_inner_loop(outer_loop_index, inner_loop_cycle)

    def _handle_one_core(self, outer_loop_index, outer_loop_num_per_core):
        """
        handle one core
        """
        with self.tik_instance.for_range(0, outer_loop_num_per_core) as c_cycle:
            outer_loop_index = outer_loop_index + c_cycle
            self._handle_per_loop(outer_loop_index)

    def _handle_out_loop(self, block_i):
        """
        Multi-core processing data of the entire block

        Parameters
        ----------
        block_i: block index

        Returns
        -------
        None

        """
        self._handle_one_core(block_i * self.tiling_floor_outer_loop_times, self.tiling_floor_outer_loop_times)
        with self.tik_instance.if_scope(block_i < self.tiling_num_outer_cores_tail):
            self._handle_one_core(block_i + self.ai_core_num * self.tiling_floor_outer_loop_times, 1)

    def _compute_mode_default(self):
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as block_i:
            self._handle_out_loop(block_i)

    # tiling mode 1 funcs begin
    def _ub_2_gm_last_block_front(self, src_ub, gm_idx, data_len):
        """
        special move for last 32B, move output gm to ub before calculate
        """
        last_block = self.tik_instance.Tensor(self.dtype_x, (BLOCK_SIZE // self.dtype_size,), tik.scope_ubuf,
                                              "last_block")

        self.tik_instance.data_move(last_block, self.output_gm[gm_idx], 0, 1, 1, 0, 0)
        tmp_scalar = self.tik_instance.Scalar(self.dtype_x)
        with self.tik_instance.for_range(0, data_len) as i:
            tmp_scalar.set_as(src_ub[i])
            last_block[i].set_as(tmp_scalar)
        self.tik_instance.data_move(self.output_gm[gm_idx], last_block, 0, 1, 1, 0, 0)

    def _ub_2_gm_smaller_inner(self, src_ub, gm_idx):
        if self.reverse:
            self._ub_2_gm_last_block_front(src_ub, gm_idx, self.tiling_equal_shape_after_axis)
        else:
            self._ub_2_gm(src_ub, gm_idx, 1)

    def _compute_mode_small_inner(self):
        with self.tik_instance.for_range(0, self.tiling_equal_shape_before_axis) as outer_loop_index:
            input_x_ub, last_res = self._request_two_ub(Constant.MAX_COMPUTE_SIZE // self.dtype_size,
                                                        Constant.MAX_COMPUTE_SIZE // self.dtype_size)
            gm_init_idx = outer_loop_index * self.tiling_nums_per_outer_loop

            if self.reverse:
                gm_init_idx += (self.tiling_axis_shape - 1) * self.tiling_equal_shape_after_axis

            if self.exclusive:
                self._dup_0_ub(last_res, 1)
                self._ub_2_gm_smaller_inner(last_res, gm_init_idx)
                self._gm_2_ub(last_res, gm_init_idx, 1)
            else:
                self._gm_2_ub(last_res, gm_init_idx, 1)
                self._ub_2_gm_smaller_inner(last_res, gm_init_idx)

            with self.tik_instance.if_scope(self.tiling_axis_shape > 1):
                with self.tik_instance.for_range(1, self.tiling_axis_shape) as axis_cycle:
                    if self.reverse:
                        gm_idx = gm_init_idx - axis_cycle * self.tiling_equal_shape_after_axis
                    else:
                        gm_idx = gm_init_idx + axis_cycle * self.tiling_equal_shape_after_axis

                    if self.exclusive:
                        self._ub_2_gm_smaller_inner(last_res, gm_idx)
                        self._gm_2_ub(input_x_ub, gm_idx, 1)
                        self._cal_data(input_x_ub, last_res, 1)
                    else:
                        self._gm_2_ub(input_x_ub, gm_idx, 1)
                        self._cal_data(input_x_ub, last_res, 1)
                        self._ub_2_gm_smaller_inner(last_res, gm_idx)


class CumsumComputer(CumComputer):

    def _cal_data(self, input_x_ub, last_res, repeat_time):
        mask = VECTOR_BYTE_SIZE // self.dtype_size
        self.tik_instance.vadd(mask, last_res, input_x_ub, last_res, repeat_time, 1, 1, 1, Constant.MAX_REPEAT_STRIDE,
                               Constant.MAX_REPEAT_STRIDE, Constant.MAX_REPEAT_STRIDE)