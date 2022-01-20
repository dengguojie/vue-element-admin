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
dynamic_lstm_grad_cell
"""
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from tbe.dsl.base.operation import add_compile_info
from te.utils import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    INT32_MAX_NUM = 2 * 32 - 1
    TILLING_ARG_NUM = 12
    T_STATE_NUM = 1
    INT64 = 'int64'
    INT32 = 'int32'
    FORWARD = 'UNIDIRECTIONAL'
    TILLING_PARA_INDEX_MAP = {
        't_size': 0,
        'eleEachCore': 1,
        'outLoopNum': 2,
        'outLoopEleNum': 3,
        'innerLoopNum': 4,
        'innerLoopEleNum': 5,
        'lastLoopEleNum': 6,
        'ubSize': 7,
        'hiddenSize': 8,
        'batchSize': 9,
        'useCoreNum': 10,
        'fuseSize': 11,
    }


# 'pylint: disable=too-many-instance-attributes
class LstmCellGradInput:
    """
    Class: use to store LstmCellGradInput input parameters
    Modify : 2019-12-28
    """

    # 'pylint: disable=too-many-arguments,unused-argument
    def __init__(self, mask, init_c, cell_state, dht_out, dht, dct, input_gate, forget_gate,
                 update_gate, output_gate, tanh_ct, gate_order, direction, kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        cell_state: dict
            cell state at the last moment
        dht_out: dict
            output state gradient at time t
        dht: dict
            hidden state gradient at time t
        dct: dict
            cell state gradient at time t

        input_gate: dict
            forward it buffer value at time t
        forget_gate: dict
            forward ft buffer value at time t
        update_gate: dict
            forward jt buffer value at time t
        output_gate: dict
            forward ot buffer value at time t
        tanh_ct: dict
            forward tanh_ct buffer value at time t
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        if dht_out is not None:
            self.dht_out_shape = dht_out.get("shape")
            self.dht_out_dtype = dht_out.get("dtype")
        else:
            self.dht_out_shape = None
            self.dht_out_dtype = None
        if mask is None:
            self.mask_shape = None
        else:
            self.mask_shape = mask.get('shape')
        self.dht_shape = dht.get("shape")
        self.dht_dtype = dht.get("dtype")
        self.dct_shape = dct.get("shape")
        self.dct_dtype = dct.get("dtype")
        self.it_shape = input_gate.get("shape")
        self.it_dtype = input_gate.get("dtype")
        self.ft_shape = forget_gate.get("shape")
        self.ft_dtype = forget_gate.get("dtype")
        self.jt_shape = update_gate.get("shape")
        self.jt_dtype = update_gate.get("dtype")
        self.ot_shape = output_gate.get("shape")
        self.ot_dtype = output_gate.get("dtype")
        self.tanh_ct_shape = tanh_ct.get("shape")
        self.tanh_ct_dtype = tanh_ct.get("dtype")
        self.c_shape = cell_state.get("shape")
        self.c_dtype = cell_state.get("dtype")

        self.batch_size = None
        self.hidden_size = None
        self.fuse_size = None

        self.t_size = None
        self.t_state = None
        self.dgate_shape = None
        self.dgate_dtype = output_gate.get('dtype')
        self.direction = direction
        self.gate_order = gate_order

        self.kernel_name = kernel_name

        self.check_input_param()

        self.tik_instance = tik.Tik(tik.Dprofile())
        self.aicore_num = self.tik_instance.d_profiling.get_aicore_num()
        self.use_core_num = self.tik_instance.Scalar(Constant.INT64, name='use_core_num')

        self.init_gm_tensor()

    def check_input_param(self):
        """
        Check the input parameter

        Parameters
        ----------
        None

        Returns:
        None
        """
        if sorted(self.gate_order) != sorted('ijfo'):
            raise RuntimeError('gate_order illegal')
        shape_list = (self.c_shape, self.it_shape, self.jt_shape, self.ft_shape, self.ot_shape, self.tanh_ct_shape)
        no_t_shape = self.c_shape[1:]
        for shape in shape_list:
            para_check.check_shape(shape, min_rank=4, max_rank=5, param_name="dht_out")
            shape = shape if len(shape) == 4 else shape[1:]
            if shape != no_t_shape:
                raise RuntimeError("the input shapes are not same")

        check_list = ("float16", "float32")
        dtype_list = (self.c_dtype, self.dht_dtype, self.dct_dtype,
                      self.it_dtype, self.jt_dtype, self.ft_dtype,
                      self.ot_dtype, self.tanh_ct_dtype)

        if self.dht_out_dtype is not None:
            dtype_list += (self.dht_out_dtype,)

        for dtype in dtype_list:
            para_check.check_dtype(dtype.lower(), check_list, param_name="dht_out")
            if dtype != self.c_dtype:
                raise RuntimeError("the input dtypes are not same")

    def init_gm_tensor(self):
        """
        Declare tensor on gm

        Parameters
        ----------
        None

        Returns:
        None
        """
        if self.dht_out_dtype is not None:
            self.gm_dht_out = self.tik_instance.Tensor(
                self.dht_out_dtype,
                (Constant.INT32_MAX_NUM,),
                name="gm_dht_out",
                scope=tik.scope_gm)
        self.gm_dht = self.tik_instance.Tensor(
            self.dht_dtype, (Constant.INT32_MAX_NUM,), name="gm_dht", scope=tik.scope_gm)
        self.gm_dct = self.tik_instance.Tensor(
            self.dct_dtype, (Constant.INT32_MAX_NUM,), name="gm_dct", scope=tik.scope_gm)
        self.gm_it = self.tik_instance.Tensor(
            self.it_dtype, (Constant.INT32_MAX_NUM,), name="gm_it", scope=tik.scope_gm)
        self.gm_ft = self.tik_instance.Tensor(
            self.ft_dtype, (Constant.INT32_MAX_NUM,), name="gm_ft", scope=tik.scope_gm)
        self.gm_jt = self.tik_instance.Tensor(
            self.jt_dtype, (Constant.INT32_MAX_NUM,), name="gm_jt", scope=tik.scope_gm)
        self.gm_ot = self.tik_instance.Tensor(
            self.ot_dtype, (Constant.INT32_MAX_NUM,), name="gm_ot", scope=tik.scope_gm)
        self.gm_tanh_ct = self.tik_instance.Tensor(
            self.tanh_ct_dtype,
            (Constant.INT32_MAX_NUM,),
            name="gm_tanh_ct",
            scope=tik.scope_gm)
        self.gm_c = self.tik_instance.Tensor(
            self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_c", scope=tik.scope_gm)
        self.gm_init_c = self.tik_instance.Tensor(
            self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_init_c", scope=tik.scope_gm)
        if self.mask_shape is not None:
            self.gm_mask = self.tik_instance.Tensor(
                self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_mask", scope=tik.scope_gm)
        self.gm_t_state = self.tik_instance.Tensor(
            Constant.INT32, (Constant.T_STATE_NUM,), name="gm_t_state", scope=tik.scope_gm)
        self.tilling_gm = self.tik_instance.Tensor(
            Constant.INT64, (Constant.TILLING_ARG_NUM,), name="tilling_gm", scope=tik.scope_gm)
        # output gm
        self.gm_dct1 = self.tik_instance.Tensor(
            self.c_dtype, (Constant.INT32_MAX_NUM,), name="gm_dct1", scope=tik.scope_gm)

        self.gm_dgate = self.tik_instance.Tensor(
            self.dgate_dtype,
            (Constant.INT32_MAX_NUM,),
            name="gm_dgate",
            scope=tik.scope_gm)


class LstmCellGrad(LstmCellGradInput):
    """
    Class: use to store LstmCellGrad input parameters
    Modify : 2019-12-28
    """

    # 'pylint: disable=too-many-arguments,too-many-locals
    def __init__(self, mask, init_c, cell_state, dht_out, dht, dct, input_gate, forget_gate,
                 update_gate, output_gate, tanh_ct, gate_order, direction, kernel_name):
        """
        init LstmCellGradInput base parameters

        Parameters
        ----------
        cell_state: dict
            cell state at the last moment
        dht_out: dict
            output state gradient at time t
        dht: dict
            hidden state gradient at time t
        dct: dict
            cell state gradient at time t

        input_gate: dict
            forward it buffer value at time t
        forget_gate: dict
            forward ft buffer value at time t
        update_gate: dict
            forward jt buffer value at time t
        output_gate: dict
            forward ot buffer value at time t
        tanh_ct: dict
            forward tanh_ct buffer value at time t
        kernel_name: str
            op kernel name

        Returns
        -------
        None
        """
        # 'pylint: disable=super-with-arguments
        super(LstmCellGrad, self).__init__(mask, init_c, cell_state, dht_out, dht, dct, input_gate,
                                           forget_gate, update_gate, output_gate, tanh_ct, gate_order,
                                           direction, kernel_name)
        self.ele_each_core = 0
        self.out_loop_ele_num = 0
        self.out_loop_num = 0
        self.inner_loop_ele_num = 0
        self.last_loop_ele_num = 0
        self.inner_loop_num = 0
        self.ub_size = 0
        if self.mask_shape is None:
            # ub tensor count
            self.ub_pice_num = 20
        else:
            self.ub_pice_num = 21

        # get vector compute parameters
        dtype_bytes_size = tbe_platform.get_bit_len(self.dht_dtype) // 8
        int64_bytes_size = 8
        self.v_mask_max = 128 // (dtype_bytes_size // 2)
        self.v_repeat_max = 255
        self.v_ele_each_block = 32 // dtype_bytes_size

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        t_state_and_tilling_size = ((Constant.T_STATE_NUM + 15) // 16 * 16 * dtype_bytes_size + (
                Constant.TILLING_ARG_NUM + 15) // 16 * 16 * int64_bytes_size) * 2
        ub_max_ele_num = (self.ub_size_bytes - t_state_and_tilling_size) // dtype_bytes_size
        align = 256
        self.max_block_ele_num = (ub_max_ele_num // self.ub_pice_num // 2 // align) * align
        self.max_mem_size = dtype_bytes_size * self.max_block_ele_num

        self.ub_dot_conv = None
        self.ub_dit_conv = None
        self.ub_djt_conv = None
        self.ub_dft_conv = None

        self.ub_dht_out = None
        self.ub_dht = None
        self.ub_dht_add = None
        self.ub_ot = None
        self.ub_dot = None
        self.ub_tanh_ct = None
        self.ub_dc = None
        self.ub_dct = None
        self.ub_it = None
        self.ub_jt = None
        self.ub_djt = None
        self.ub_dit = None
        self.ub_dft = None
        self.ub_c = None
        self.ub_ft = None
        self.ub_dct1 = None
        self.ub_mask = None
        self.ub_t_state = None
        self.tmp_data1 = None
        self.ub_t_size = None

    def get_tik_instance(self):
        """
        Return tik instance for tik debug

        Parameters
        ----------
        None

        Returns:
        tik_instance:
            tik instance
        """
        return self.tik_instance

    def get_tilling_params(self):
        """
        set tilling params
        """

        tilling_ub = self.tik_instance.Tensor(Constant.INT64, (Constant.TILLING_ARG_NUM,), name="tilling_ub", \
        scope=tik.scope_ubuf)
        burst = (Constant.TILLING_ARG_NUM * 8 + 15) // 16 * 16 // 32
        self.tik_instance.data_move(tilling_ub, self.tilling_gm, 0, 1, burst, 0, 0)
        self.t_size = self.tik_instance.Scalar(Constant.INT64, name='t_size')
        self.t_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('t_size')])
        self.ele_each_core = self.tik_instance.Scalar(Constant.INT64, name='ele_each_core')
        self.ele_each_core.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('eleEachCore')])
        self.out_loop_num = self.tik_instance.Scalar(Constant.INT64, name='out_loop_num')
        self.out_loop_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('outLoopNum')])
        self.out_loop_ele_num = self.tik_instance.Scalar(Constant.INT64, name='out_loop_ele_num')
        self.out_loop_ele_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('outLoopEleNum')])
        self.inner_loop_num = self.tik_instance.Scalar(Constant.INT64, name='inner_loop_num')
        self.inner_loop_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('innerLoopNum')])
        self.inner_loop_ele_num = self.tik_instance.Scalar(Constant.INT64, name='inner_loop_ele_num')
        self.inner_loop_ele_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('innerLoopEleNum')])
        self.last_loop_ele_num = self.tik_instance.Scalar(Constant.INT64, name='last_loop_ele_num')
        self.last_loop_ele_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('lastLoopEleNum')])
        self.ub_size = self.tik_instance.Scalar(Constant.INT64, name='ub_size')
        self.ub_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('ubSize')])
        self.hidden_size = self.tik_instance.Scalar(Constant.INT64, name='hidden_size')
        self.hidden_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('hiddenSize')])
        self.batch_size = self.tik_instance.Scalar(Constant.INT64, name='batch_size')
        self.batch_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('batchSize')])
        self.fuse_size = self.tik_instance.Scalar(Constant.INT64, name='fuse_size')
        self.fuse_size.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('fuseSize')])
        self.use_core_num.set_as(tilling_ub[Constant.TILLING_PARA_INDEX_MAP.get('useCoreNum')])

    def init_ub(self):
        """
        Declare tensor on UB buffer

        Parameters
        ----------
        None

        Returns:
        None
        """
        self.ub_mask = self.tik_instance.Tensor(
            self.dht_dtype, (self.ub_size,),
            name='ub_mask',
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size
        )
        self.ub_dht = self.tik_instance.Tensor(
            self.dht_dtype, (self.ub_size,),
            name="ub_dht",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        if self.dht_out_shape is not None:
            self.ub_dht_out = self.tik_instance.Tensor(
                self.dht_out_dtype, (self.ub_size,),
                name="ub_dht_out",
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
            self.ub_dht_add = self.tik_instance.Tensor(
                self.dht_out_dtype, (self.ub_size,),
                name="ub_dht_add",
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_ot = self.tik_instance.Tensor(
            self.ot_dtype, (self.ub_size,), name="ub_ot", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_dot = self.tik_instance.Tensor(
            self.ot_dtype, (self.ub_size,), name="ub_dot", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_tanh_ct = self.tik_instance.Tensor(
            self.tanh_ct_dtype, (self.ub_size,),
            name="ub_tanh_ct",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dc = self.tik_instance.Tensor(
            self.dct_dtype, (self.ub_size,), name="ub_dc", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_dct = self.tik_instance.Tensor(
            self.dct_dtype, (self.ub_size,),
            name="ub_dct",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_it = self.tik_instance.Tensor(
            self.it_dtype, (self.ub_size,), name="ub_it", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_jt = self.tik_instance.Tensor(
            self.jt_dtype, (self.ub_size,), name="ub_jt", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_djt = self.tik_instance.Tensor(
            self.jt_dtype, (self.ub_size,), name="ub_djt", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dit = self.tik_instance.Tensor(
            self.it_dtype, (self.ub_size,), name="ub_dit", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dft = self.tik_instance.Tensor(
            self.ft_dtype, (self.ub_size,), name="ub_dft", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_c = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,), name="ub_c", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)
        self.ub_ft = self.tik_instance.Tensor(
            self.ft_dtype, (self.ub_size,), name="ub_ft", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.ub_dct1 = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,), name="ub_dct1", scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        self.tmp_data1 = self.tik_instance.Tensor(
            self.c_dtype, (self.ub_size,),
            name="temp_data1",
            scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

        if self.it_dtype == "float32":
            # vconv dot
            self.ub_dot_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_dot_conv",
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

            # vconv dit
            self.ub_dit_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_dit_conv",
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

            # vconv djt
            self.ub_djt_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_djt_conv",
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

            # vconv dft
            self.ub_dft_conv = self.tik_instance.Tensor(
                "float16", (self.ub_size,),
                name="ub_dft_conv",
                scope=tik.scope_ubuf, max_mem_size=self.max_mem_size)

    def vector_compute(self, index, mask, repeat):
        """
        Calculate the smallest data shard

        Parameters
        ----------
        src: int
            source address offset
        dst: int
            destination address offset
        mask: int
            vector compute mask
        repeat:
            vector compute repeat times
        Returns:
        None
        """
        # mask mul dy dh dc
        if self.mask_shape is not None:
            self.tik_instance.vmul(mask, self.ub_dht[index], self.ub_dht[index], self.ub_mask[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmul(mask, self.ub_dct[index], self.ub_dct[index], self.ub_mask[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
        # compute process for dot
        if self.dht_out_shape is not None:
            if self.mask_shape is not None:
                self.tik_instance.vmul(mask, self.ub_dht_out[index], self.ub_dht_out[index], self.ub_mask[index],
                                       repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vadd(mask, self.ub_dht_add[index],
                                   self.ub_dht_out[index], self.ub_dht[index],
                                   repeat, 1, 1, 1, 8, 8, 8)
        else:
            self.ub_dht_add = self.ub_dht

        # compute process for dot
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_tanh_ct[index],
                               self.ub_dht_add[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_dot[index],
                               self.ub_ot[index], repeat, 1, 1, 1, 8, 8, 8)

        self.tik_instance.vmuls(mask, self.tmp_data1[index], self.ub_ot[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dot[index], self.ub_dot[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dc
        self.tik_instance.vmul(mask, self.ub_dht_add[index],
                               self.ub_dht_add[index], self.ub_ot[index],
                               repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dc[index], self.ub_tanh_ct[index],
                               self.ub_tanh_ct[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.ub_dc[index], self.ub_dc[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.ub_dc[index], self.ub_dc[index], 1,
                                repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dc[index], self.ub_dc[index],
                               self.ub_dht_add[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vadd(mask, self.ub_dc[index], self.ub_dc[index],
                               self.ub_dct[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dit
        self.tik_instance.vmuls(mask, self.tmp_data1[index], self.ub_it[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dc[index],
                               self.ub_jt[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dit[index],
                               self.ub_it[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dit[index], self.ub_dit[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for djt
        self.tik_instance.vmul(mask, self.tmp_data1[index], self.ub_jt[index],
                               self.ub_jt[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmuls(mask, self.tmp_data1[index],
                                self.tmp_data1[index], -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_djt[index], self.ub_it[index],
                               self.ub_dc[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_djt[index], self.ub_djt[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # fake
        # compute process for dft
        self.tik_instance.vmuls(mask, self.tmp_data1[index], self.ub_ft[index],
                                -1.0, repeat, 1, 1, 8, 8)
        self.tik_instance.vadds(mask, self.tmp_data1[index],
                                self.tmp_data1[index], 1, repeat, 1, 1, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dc[index],
                               self.ub_c[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dft[index],
                               self.ub_ft[index], repeat, 1, 1, 1, 8, 8, 8)
        self.tik_instance.vmul(mask, self.ub_dft[index], self.ub_dft[index],
                               self.tmp_data1[index], repeat, 1, 1, 1, 8, 8, 8)

        # compute process for dct-1
        self.tik_instance.vmul(mask, self.ub_dct1[index], self.ub_dc[index],
                               self.ub_ft[index], repeat, 1, 1, 1, 8, 8, 8)
        if self.it_dtype == "float32":
            self.tik_instance.vconv(mask, "", self.ub_dot_conv[index],
                                    self.ub_dot[index], repeat, 1, 1, 4, 8)
            self.tik_instance.vconv(mask, "", self.ub_dit_conv[index],
                                    self.ub_dit[index], repeat, 1, 1, 4, 8)
            self.tik_instance.vconv(mask, "", self.ub_djt_conv[index],
                                    self.ub_djt[index], repeat, 1, 1, 4, 8)
            self.tik_instance.vconv(mask, "", self.ub_dft_conv[index],
                                    self.ub_dft[index], repeat, 1, 1, 4, 8)

    def compute_each_loop(self, ele_num):
        """
        Calculate each loop

        Parameters
        ----------
        start_index: int
            source address offset

        Returns:
        None
        """
        # vector compute
        loop_num = ele_num // (self.v_mask_max * self.v_repeat_max)
        with self.tik_instance.if_scope(loop_num > 0):
            with self.tik_instance.for_range(0, loop_num) as index:
                compute_index = self.v_mask_max * self.v_repeat_max * index
                self.vector_compute(compute_index, self.v_mask_max,
                                    self.v_repeat_max)

        repeat_times = (
                ele_num % (self.v_mask_max * self.v_repeat_max) // self.v_mask_max)
        with self.tik_instance.if_scope(repeat_times > 0):
            compute_index = self.v_mask_max * self.v_repeat_max * loop_num
            self.vector_compute(compute_index, self.v_mask_max, repeat_times)

        tile_mask = ele_num % self.v_mask_max
        with self.tik_instance.if_scope(tile_mask > 0):
            compute_index = (
                    self.v_mask_max * self.v_repeat_max * loop_num +
                    repeat_times * self.v_mask_max)
            self.vector_compute(compute_index, tile_mask, 1)

    def calc_t_offset(self):
        """
        Calculate t_offset

        Parameters
        ----------

        Returns:
        t_offset
        """
        self.ub_t_state = self.tik_instance.Tensor(
            Constant.INT32, (4,),
            name="ub_t_state",
            scope=tik.scope_ubuf
        )
        self.tik_instance.data_move(self.ub_t_state, self.gm_t_state, 0, 1, 1, 0, 0)
        self.t_state = self.tik_instance.Scalar(Constant.INT32, name='t_state', init_value=self.ub_t_state[0])
        if self.direction == Constant.FORWARD:
            t_offset = self.t_size - self.t_state - 1
        else:
            t_offset = self.t_state
        return t_offset

    def input_data_move_in(self, start_index, t_offset, c_t_offset, ele_num):
        """
        Move the input data to ub

        Parameters
        ----------
        start_index: int
            source address offset

        Returns:
        None
        """
        # move in vector data
        v_burst_lens = ele_num // self.v_ele_each_block
        if self.mask_shape is not None:
            self.tik_instance.data_move(self.ub_mask, self.gm_mask[t_offset], 0, 1, v_burst_lens, 0, 0)
        if self.dht_out_shape is not None:
            self.tik_instance.data_move(self.ub_dht_out,
                                        self.gm_dht_out[t_offset], 0, 1,
                                        v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dht, self.gm_dht[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_ot, self.gm_ot[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_tanh_ct,
                                    self.gm_tanh_ct[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_it, self.gm_it[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_jt, self.gm_jt[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_ft, self.gm_ft[t_offset], 0, 1,
                                    v_burst_lens, 0, 0)
        self.tik_instance.data_move(self.ub_dct, self.gm_dct[start_index], 0, 1,
                                    v_burst_lens, 0, 0)
        with self.tik_instance.if_scope(self.t_state == self.t_size - 1):
            self.tik_instance.data_move(self.ub_c, self.gm_init_c[start_index], 0, 1, v_burst_lens, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.ub_c, self.gm_c[c_t_offset], 0, 1, v_burst_lens, 0, 0)

    def compute_each_core(self, core_index, out_loop_index, t_offset):
        """
        Calculate the data on each core

        Parameters
        ----------
        core_index: int
            the index of aicore
        core_index: int
            the index of out loop

        Returns:
        None
        """
        self.init_ub()
        loop_offset = (
                core_index * self.ele_each_core +
                out_loop_index * self.out_loop_ele_num)
        with self.tik_instance.if_scope(self.inner_loop_num > 0):
            with self.tik_instance.for_range(0, self.inner_loop_num) as index:
                start_index = loop_offset + index * self.inner_loop_ele_num
                if self.direction == Constant.FORWARD:
                    self.input_data_move_in(start_index, t_offset * self.fuse_size + start_index,
                                            (t_offset - 1) * self.fuse_size + start_index, self.inner_loop_ele_num)
                else:
                    self.input_data_move_in(start_index, t_offset * self.fuse_size + start_index,
                                            (t_offset + 1) * self.fuse_size + start_index, self.inner_loop_ele_num)
                self.compute_each_loop(self.inner_loop_ele_num)

                # move vector compute result to l2 and gm
                self.move_vector_data_out(start_index, self.inner_loop_ele_num)

        with self.tik_instance.if_scope(self.last_loop_ele_num > 0):
            start_index = (
                    loop_offset + self.inner_loop_num * self.inner_loop_ele_num)
            if self.direction == Constant.FORWARD:
                self.input_data_move_in(start_index, t_offset * self.fuse_size + start_index,
                                        (t_offset - 1) * self.fuse_size + start_index, self.last_loop_ele_num)
            else:
                self.input_data_move_in(start_index, t_offset * self.fuse_size + start_index,
                                        (t_offset + 1) * self.fuse_size + start_index, self.inner_loop_ele_num)
            self.compute_each_loop(self.last_loop_ele_num)

            # move vector compute result to l2 and gm
            self.move_vector_data_out(start_index, self.last_loop_ele_num)

    def move_vector_data_out(self, index, ele_num):
        """
        Move the vector compute result to gm

        Parameters
        ----------
        index: int
            move out index
        ele_num: int
            the element number of result

        Returns:
        None
        """
        burst_len = ele_num // self.v_ele_each_block
        if self.it_dtype == "float32":
            djt_src = self.ub_djt_conv
            dit_src = self.ub_dit_conv
            dot_src = self.ub_dot_conv
            dft_src = self.ub_dft_conv
            dgate_burst_len = burst_len // 2
        else:
            djt_src = self.ub_djt
            dit_src = self.ub_dit
            dot_src = self.ub_dot
            dft_src = self.ub_dft
            dgate_burst_len = burst_len

        offset = self.batch_size * self.hidden_size
        gate_data_map = {'i': dit_src, 'j': djt_src, 'f': dft_src, 'o': dot_src}
        self.tik_instance.data_move(self.gm_dgate[index], gate_data_map[self.gate_order[0]], 0, 1,
                                    dgate_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_dgate[index + offset], gate_data_map[self.gate_order[1]], 0,
                                    1, dgate_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_dgate[index + offset * 2], gate_data_map[self.gate_order[2]],
                                    0, 1, dgate_burst_len, 0, 0)
        self.tik_instance.data_move(self.gm_dgate[index + offset * 3], gate_data_map[self.gate_order[3]],
                                    0, 1, dgate_burst_len, 0, 0)

        self.tik_instance.data_move(self.gm_dct1[index], self.ub_dct1, 0, 1,
                                    burst_len, 0, 0)

    def compute(self):
        """
        Calculate the data

        Parameters
        ----------
        None

        Returns:
        None
        """
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as index0:
            self.get_tilling_params()
            t_offset = self.calc_t_offset()
            with self.tik_instance.for_range(0, 2, thread_num=2) as index1:
                with self.tik_instance.if_scope(index0 < self.use_core_num):
                    self.compute_each_core(index0, index1, t_offset)

        if self.dht_out_shape is not None and self.mask_shape is not None:
            input_list = (self.gm_init_c, self.gm_c, self.gm_dht_out, self.gm_dht, self.gm_dct,
                          self.gm_it, self.gm_jt, self.gm_ft, self.gm_ot,
                          self.gm_tanh_ct, self.gm_t_state, self.gm_mask)
        elif self.dht_out_shape is not None and self.mask_shape is None:
            input_list = (self.gm_init_c, self.gm_c, self.gm_dht_out, self.gm_dht, self.gm_dct,
                          self.gm_it, self.gm_jt, self.gm_ft, self.gm_ot,
                          self.gm_tanh_ct, self.gm_t_state)
        elif self.dht_out_shape is None and self.mask_shape is not None:
            input_list = (self.gm_init_c, self.gm_c, self.gm_dht, self.gm_dct,
                          self.gm_it, self.gm_jt, self.gm_ft, self.gm_ot,
                          self.gm_tanh_ct, self.gm_t_state, self.gm_mask)
        else:
            input_list = (self.gm_init_c, self.gm_c, self.gm_dht, self.gm_dct,
                          self.gm_it, self.gm_jt, self.gm_ft, self.gm_ot,
                          self.gm_tanh_ct, self.gm_t_state)

        add_compile_info("vars", {"device_aicore_num": self.aicore_num,
                                  "ub_size": tbe_platform.get_soc_spec(tbe_platform.UB_SIZE),
                                  "mask_input": 0 if self.mask_shape is None else 1})
        self.tik_instance.BuildCCE(
            kernel_name=self.kernel_name,
            inputs=input_list,
            outputs=(self.gm_dgate, self.gm_dct1),
            flowtable=(self.tilling_gm,),
            enable_l2=False)


# 'pylint: disable=unused-argument,too-many-arguments,invalid-name,too-many-locals
@register_operator("DynamicLSTMGradCell")
def dynamic_lstm_grad_cell(init_c, c, dy, dh, dc, i, j, f, o, tanhct, t_state, mask, dgate, dct1,
                           forget_bias=1, activation="tanh", direction="UNIDIRECTIONAL",
                           gate_order="ijfo", kernel_name="dynamic_lstm_grad_cell"):
    """
    Calculate the gradient of the four gates and the state of c at t-1

    Parameters
    ----------
    c: dict
        cell state at the last moment
    dht: dict
        hidden state gradient at time t
    dct: dict
        cell state gradient at time t
    it: dict
        forward it buffer value at time t
    jt: dict
        forward jt buffer value at time t
    ft: dict
        forward ft buffer value at time t
    ot: dict
        forward ot buffer value at time t
    tanh_ct: dict
        forward tanh_ct buffer value at time t
    forget_bias: int
        the bias of forget gate
    activation: str
        activation method
    kernel_name: str
        op kernel name

    Returns:
    None
    """
    dht_out = dy
    dht = dh
    dct = dc
    it = i
    jt = j
    ft = f
    ot = o

    lstm_cell_grad = LstmCellGrad(mask, init_c, c, dht_out, dht, dct, it, ft, jt, ot, tanhct, gate_order, direction,
                                  kernel_name)
    lstm_cell_grad.compute()

    return lstm_cell_grad
