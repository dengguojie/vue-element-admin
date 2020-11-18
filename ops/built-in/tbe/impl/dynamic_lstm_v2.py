"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic_lstm_v2
"""
import operator

from te.lang.cce import broadcast
from te.lang.cce import cast_to
from te.lang.cce import vabs
from te.lang.cce import vadd
from te.lang.cce import vadds
from te.lang.cce import vdiv
from te.lang.cce import vexp
from te.lang.cce import vmul
from te.lang.cce import vmuls
from te.lang.cce import vrec
from te.lang.cce import vsub
from te.domain.rl_bank import rl_bank
from te.platform import scope_ca
from te.platform import scope_cb
from te.platform import scope_cbuf
from te.platform import scope_cc
from te.platform import scope_ubuf
from te.platform.fusion_manager import fusion_manager
from te.platform.cce_conf import api_check_support
from te.tik import Dprofile
from te.tik import Tik
from te.tik import scope_gm
from te.tvm import api as tvm
from te.tvm.schedule import create_schedule
from te.utils import para_check
from te.utils.error_manager import error_manager_vector

def sigmoid_compute(input_x):
    """
    calculating sigmoid
    """
    data_input = input_x
    dtype = input_x.dtype
    exp_support = api_check_support(
        "te.lang.cce.vexp", "float32")
    mul_support = api_check_support(
        "te.lang.cce.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_manager_vector.raise_err_specific_reson("DynamicLSTM",
                                                      "Input dtype only support float16 while input dtype is float32")

    const_num_neg_one = tvm.const(-1, dtype=dtype)
    const_num_one = tvm.const(1, dtype=dtype)
    tmp_negative = vmuls(data_input, const_num_neg_one)
    if dtype == "float32" and not exp_support:
        tmp_negative = cast_to(tmp_negative, "float16")
    tmp_exp = vexp(tmp_negative)
    if dtype == "float32" and not exp_support:
        tmp_exp = cast_to(tmp_exp, "float32")
    tmp_sum = vadds(tmp_exp, const_num_one)
    if dtype == "float32":
        inp_shape = tmp_sum.shape
        tensor_one = broadcast(tvm.const(1, dtype), inp_shape)
        res = vdiv(tensor_one, tmp_sum)
    else:
        res = vrec(tmp_sum)

    return res


def tanh_compute(input_x):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)
    """
    input_dtype = input_x.dtype
    # positive min float32 value
    min_fp_data = 2 ** (-126)
    const_dtype = input_dtype
    # positive min float16 value
    if input_dtype == "float16":
        min_fp_data = 2 ** (-14)

    has_improve_precision = False

    input_abs = vabs(input_x)
    power_val = vmuls(input_abs, tvm.const(-2, const_dtype))
    exp_val = vexp(power_val)

    up_val_tmp = vmul(exp_val, input_x)
    up_val = vsub(input_x, up_val_tmp)

    input_x_tmp = vadds(input_abs, min_fp_data)
    down_val_tmp = vadds(exp_val, tvm.const(1, const_dtype))
    down_val = vmul(down_val_tmp, input_x_tmp)

    res = vdiv(up_val, down_val)

    if has_improve_precision:
        res = cast_to(res, "float16")

    return res


def get_emit_insn_map(tensor):
    """
    get tensor's emit_insn key
    """
    insn_map = {"elewise_single_cast": "vector_conv",
                "elewise_single_VS_max": "vector_maxs",
                "elewise_single_VS_min": "vector_mins",
                "elewise_single_log": "vector_ln",
                "elewise_single_exp": "vector_exp",
                "elewise_single_rec": "vector_rec",
                "elewise_single_relu": "vector_relu",
                "elewise_single_abs": "vector_abs",
                "elewise_single_not": "vector_not",
                "elewise_single_sqrt": "vector_sqrt",
                "elewise_single_rsqrt": "vector_rsqrt",
                "elewise_binary_mul": "vector_mul",
                "elewise_single_VS_mul": "vector_muls",
                "elewise_binary_div": "vector_div",
                "elewise_binary_add": "vector_add",
                "elewise_single_VS_add": "vector_adds",
                "elewise_binary_min": "vector_min",
                "elewise_binary_max": "vector_max",
                "elewise_binary_vcmpv_gt": "vector_gt",
                "elewise_binary_vcmpv_ge": "vector_ge",
                "elewise_binary_vcmpv_lt": "vector_lt",
                "elewise_binary_vcmpv_le": "vector_le",
                "elewise_binary_vcmpv_eq": "vector_eq",
                "elewise_binary_vcmpv_ne": "vector_ne",
                "elewise_binary_or": "vector_or",
                "elewise_binary_and": "vector_and",
                "elewise_multiple_mla": "vector_multiple",
                "elewise_multiple_madd": "vector_multiple",
                "elewise_multiple_maddrelu": "vector_multiple",
                "broadcast_for_tensor": "broadcast_for_tensor",
                "elewise_binary_sub": "vector_sub",
                "broadcast": "broadcast"}
    if tensor.op.tag.find("|") != -1:
        str_list = tensor.op.tag.split("|")
        insn = insn_map.get(str_list[0])
    else:
        insn = insn_map.get(tensor.op.tag)
    return insn


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_prama_dtype(input_x, weight, bias, h0, c0, y, output_h, output_c):
    """
    check parameters dtype
    :return:
    """
    pass


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_prama_shape(input_x, weight, bias, cont, h0, c0, wci,
                      wcf, wco, mask, y, output_h, output_c):
    """
    check parameters
    """
    pass


# pylint: disable=too-many-arguments,too-many-branches,too-many-locals
def check_attr(cell_type, direction, cell_depth, use_peephole, keep_prob,
               cell_clip, num_proj, time_major, activation):
    """
    check parameters
    """
    pass


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
# pylint: disable=too-many-arguments,too-many-locals,invalid-name
# pylint: disable=too-many-function-args,too-many-statements
def dynamic_lstm_v2(input_x, weight, bias, cont, w_xc_x_static, h0, c0, wci, wcf,
                wco, mask, y, output_h, output_c,
                num_output=0, expose_hidden=False, time_major=True, forget_bias=0.0,
                kernel_name="dynamic_lstm"):
    """
    dynamic_lstm_v2
    """

    check_prama_dtype(input_x, weight, bias, h0, c0, y, output_h, output_c)

    check_prama_shape(input_x, weight, bias, cont, h0, c0, wci,
                      wcf, wco, mask, y, output_h, output_c)

    shape_x_input = input_x.get("shape")
    shape_w_input = weight.get("shape")

    input_dtype = input_x.get("dtype").lower()
    bias_dtype = bias.get("dtype").lower()

    t_size = shape_x_input[0]
    m_size = shape_x_input[2]
    k_size = shape_w_input[0]
    n_size = shape_w_input[1]

    block_size = 4
    hidden_size = n_size // 4
    in_x = k_size - hidden_size

    shape_x = (t_size, in_x, m_size, 16, 16)
    shape_w = (1, k_size, block_size, hidden_size, 16, 16)
    shape_hc = (t_size, hidden_size, m_size, 16, 16)
    shape_bias = (1, block_size, hidden_size, 1, 1, 16)
    shape_hc_init = (1, hidden_size, m_size, 16, 16)

    is_global_init = False
    if h0 is not None:
        is_global_init = True

    # due to FE/GE not support, now set default value
    is_gate_output = False

    tik_instance = Tik(Dprofile())

    input_x = tik_instance.Tensor(shape=shape_x, dtype=input_dtype,
                                  scope=scope_gm, name='input_x')
    weight = tik_instance.Tensor(shape=shape_w, dtype=input_dtype,
                                 scope=scope_gm, name='weight')
    bias = tik_instance.Tensor(shape=shape_bias, scope=scope_gm,
                               dtype=bias_dtype, name='bias')
    sync = tik_instance.Tensor(shape=(128, ), dtype='int64', scope=scope_gm, name='sync',
                               is_workspace=True, is_atomic_add=True)
    cont_gm = tik_instance.Tensor(shape=cont['shape'], dtype=cont['dtype'], scope=scope_gm, name='cont_gm')

    if is_global_init:
        s_init_h_gm = tik_instance.Tensor(shape=shape_hc_init,
                                          dtype=input_dtype,
                                          scope=scope_gm,
                                          name='s_init_h_gm')
        s_init_c_gm = tik_instance.Tensor(shape=shape_hc_init, dtype=bias_dtype, scope=scope_gm, name='s_init_c_gm')
                                    
    update_h_gm = tik_instance.Tensor(shape=shape_hc, dtype=input_dtype,
                                      scope=scope_gm, name='update_h_gm')
    update_c_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                      scope=scope_gm, name='update_c_gm')
    update_h_gm_as_y = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                           scope=scope_gm,
                                           name='update_h_gm_as_y')

    if is_gate_output:
        f_t_sigmoid_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                             scope=scope_gm,
                                             name='f_t_sigmoid_gm')
        i_t_sigmoid_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                             scope=scope_gm,
                                             name='i_t_sigmoid_gm')
        o_t_sigmoid_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                             scope=scope_gm,
                                             name='o_t_sigmoid_gm')
        j_t_tanh_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                          scope=scope_gm,
                                          name='j_t_tanh_gm')
        c_t_tanh_gm = tik_instance.Tensor(shape=shape_hc, dtype=bias_dtype,
                                          scope=scope_gm,
                                          name='c_t_tanh_gm')

    if is_global_init:
        build_input_list = [input_x, weight, bias, cont_gm, s_init_h_gm, s_init_c_gm]
    else:
        build_input_list = [input_x, weight, bias, cont_gm]

    if is_gate_output:
        build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm,
                             i_t_sigmoid_gm, j_t_tanh_gm, f_t_sigmoid_gm,
                             o_t_sigmoid_gm, c_t_tanh_gm]
    else:
        build_output_list = [update_h_gm_as_y, update_h_gm, update_c_gm]

    # for RL tune getting tik input&output tensor
    fusion_manager.set_tik_tensor(build_input_list, build_output_list)

    last = 1
    cut_t = 1
    # RL default tiling
    cut_m = m_size
    loop_t = t_size // cut_t
    loop_m = m_size // cut_m

    with tik_instance.for_range(0, loop_t) as loop_i:
        with tik_instance.for_range(0, loop_m) as loop_j:

            input_x_var = input_x[loop_i * cut_t: loop_i * cut_t + cut_t,
                          :,
                          loop_j * cut_m: loop_j * cut_m + cut_m,
                          :, :]
            cont_var = cont_gm[loop_i, :]
            if is_global_init:
                s_init_c_gm_var = s_init_c_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
                s_init_h_gm_var = s_init_h_gm[:, :, loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
            else:
                s_init_c_gm_var = None
                s_init_h_gm_var = None

            state_h_last = update_h_gm[
                           loop_i * cut_t - last: loop_i * cut_t + cut_t - last:,
                           :, loop_j * cut_m: loop_j * cut_m + cut_m, :, :]
            state_c_last = update_c_gm[
                           loop_i * cut_t - last: loop_i * cut_t + cut_t - last:,
                           :,
                           loop_j * cut_m: loop_j * cut_m + cut_m,
                           :, :]

            update_h_gm_var = update_h_gm[
                              loop_i * cut_t: loop_i * cut_t + cut_t,
                              :,
                              loop_j * cut_m: loop_j * cut_m + cut_m,
                              :,
                              :]
            update_c_gm_var = update_c_gm[
                              loop_i * cut_t: loop_i * cut_t + cut_t,
                              :,
                              loop_j * cut_m: loop_j * cut_m + cut_m,
                              :,
                              :]
            update_h_gm_as_y_var = update_h_gm_as_y[
                                   loop_i * cut_t: loop_i * cut_t + cut_t:,
                                   :,
                                   loop_j * cut_m: loop_j * cut_m + cut_m,
                                   :, :]

            if is_gate_output:
                f_t_sigmoid_gm_var = f_t_sigmoid_gm[
                                     loop_i * cut_t: loop_i * cut_t + cut_t,
                                     :,
                                     loop_j * cut_m: loop_j * cut_m + cut_m,
                                     :, :]
                i_t_sigmoid_gm_var = i_t_sigmoid_gm[
                                     loop_i * cut_t: loop_i * cut_t + cut_t,
                                     :,
                                     loop_j * cut_m: loop_j * cut_m + cut_m,
                                     :, :]
                o_t_sigmoid_gm_var = o_t_sigmoid_gm[
                                     loop_i * cut_t: loop_i * cut_t + cut_t,
                                     :,
                                     loop_j * cut_m: loop_j * cut_m + cut_m,
                                     :, :]
                j_t_tanh_gm_var = j_t_tanh_gm[
                                  loop_i * cut_t: loop_i * cut_t + cut_t,
                                  :,
                                  loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
                c_t_tanh_gm_var = c_t_tanh_gm[
                                  loop_i * cut_t: loop_i * cut_t + cut_t,
                                  :,
                                  loop_j * cut_m: loop_j * cut_m + cut_m,
                                  :, :]
            else:
                f_t_sigmoid_gm_var = None
                i_t_sigmoid_gm_var = None
                o_t_sigmoid_gm_var = None
                j_t_tanh_gm_var = None
                c_t_tanh_gm_var = None

            input_list = [input_x_var, weight, bias, s_init_h_gm_var,
                          s_init_c_gm_var, state_h_last, state_c_last, cont_var, sync]

            if is_gate_output:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var, i_t_sigmoid_gm_var,
                               j_t_tanh_gm_var, f_t_sigmoid_gm_var,
                               o_t_sigmoid_gm_var, c_t_tanh_gm_var]
            else:
                output_list = [update_h_gm_var, update_c_gm_var,
                               update_h_gm_as_y_var]

            with tik_instance.if_scope(loop_i == 0):
                is_first_round = True
                tik_instance.call_module(
                    dynamic_rnn_tik,
                    input_list,
                    output_list,
                    [is_gate_output, is_first_round, is_global_init,
                     forget_bias])

            with tik_instance.if_scope(loop_i > 0):
                is_first_round = False
                tik_instance.call_module(
                    dynamic_rnn_tik,
                    input_list,
                    output_list,
                    [is_gate_output, is_first_round, is_global_init,
                     forget_bias])

    config_map = {
        "dump_cce_code": False,
    }

    tik_instance.BuildCCE(kernel_name,
                          build_input_list,
                          build_output_list,
                          config=config_map)

# pylint: disable=too-many-arguments,too-many-locals,invalid-name
# pylint: disable=too-many-function-args,too-many-statements
def dynamic_rnn_tik(input_list, custom_list):
    """
    dynamic rnn tik
    """
    input_x = input_list[0]
    weight = input_list[1]
    bias = input_list[2]
    s_init_h_gm = input_list[3]
    s_init_c_gm = input_list[4]
    s_state_h_gm_last = input_list[5]
    s_state_c_gm_last = input_list[6]
    seq_length_gm = input_list[7]
    sync0 = input_list[8]


    is_gate_output = custom_list[0]
    is_first_round = custom_list[1]
    is_global_init = custom_list[2]
    forget_bias = custom_list[3]

    return dynamic_rnn_core(input_x, weight, bias, seq_length_gm, s_init_h_gm, s_init_c_gm,
                            s_state_h_gm_last, s_state_c_gm_last, sync0, 
                            is_gate_output, is_first_round, is_global_init,
                            forget_bias)


# pylint: disable=too-many-arguments,too-many-locals,invalid-name
# pylint: disable=too-many-statements
def dynamic_rnn_core(input_x, weight, bias, seq_length, s_init_h_gm, s_init_c_gm,
                     s_state_h_gm_last, s_state_c_gm_last, sync0,
                     is_gate_output, is_first_round, is_global_init, forget_bias):
    """
    dynamic rnn core tvm
    """
    shape_x_input = input_x.shape
    shape_w_input = weight.shape

    t_size = 1
    m_size = shape_x_input[2].value
    k_size = shape_w_input[1].value
    n_size = shape_w_input[2].value * shape_w_input[3].value

    block_size = 4
    hidden_size = n_size // block_size
    in_x = k_size - hidden_size

    shape_a_z_bigz = (t_size, m_size, k_size, 16, 16)
    shape_b = (t_size, k_size, block_size, hidden_size, 16, 16)
    shape_c = (t_size, block_size, hidden_size, m_size, 16, 16)
    shape_bias = (t_size, block_size, hidden_size, 1, 1, 16)
    shape_h = (t_size, k_size - in_x, m_size, 16, 16)
    shape_i = (t_size, hidden_size, m_size, 16, 16)

    k0_size = 16
    input_dtype = input_x.dtype
    bias_dtype = bias.dtype

    fp16_input_output = False
    if bias_dtype == 'float16':
        fp16_input_output = True

    # compute

    if is_first_round:
        if is_global_init:
            s_state_h_ub = tvm.compute(shape_h,
                                       lambda _, i, j, k, l: s_init_h_gm[
                                           0, i, j, k, l], name="s_init_h")
            s_state_c_ub = tvm.compute(shape_i,
                                       lambda _, i, j, k, l: s_init_c_gm[0, i, j, k, l], name='s_init_c')
        else:
            s_state_h_ub = \
                tvm.compute(shape_h,
                            lambda *indices: tvm.const(0.0, dtype=input_dtype),
                            name='s_state_h_ub',
                            tag="broadcast")

            s_state_c_ub = \
                tvm.compute(shape_i,
                            lambda *indices: tvm.const(0.0, dtype=bias_dtype),
                            name='s_state_c_ub',
                            tag="broadcast")
    else:
        s_state_h_ub = tvm.compute(shape_h,
                                   lambda _, i, j, k, l: s_state_h_gm_last[
                                       0, i, j, k, l], name="s_state_h_ub")
        s_state_c_ub = tvm.compute(shape_i,
                                   lambda _, i, j, k, l: s_state_c_gm_last[
                                       0, i, j, k, l], name="s_state_c_ub")
    
    # handle cont mul h  caffe
    tmp_shape = [1, 1, (seq_length.shape[1] + 15) // 16, 16, 1]
    tensor_seq_length_ub = tvm.compute(tmp_shape, lambda i, j, l, m, n: seq_length[l, m], name='tensor_seq_length_ub')
    tensor_seq_length_bc_ub = broadcast(tensor_seq_length_ub, shape_h)
    s_state_h_mul_cont_ub = vmul(s_state_h_ub, tensor_seq_length_bc_ub)
    

    # input and s_start_h is Nz, need trans to zZ
    # so change axis 1 and 2
    a_ub = tvm.compute(shape_a_z_bigz,
                       lambda *indice:
                       tvm.select(indice[2] < in_x,
                                  input_x[indice[0],
                                          indice[2],
                                          indice[1],
                                          indice[3],
                                          indice[4]],
                                  s_state_h_mul_cont_ub[0,
                                               indice[2] - in_x,
                                               indice[1],
                                               indice[3],
                                               indice[4]]
                                  ),
                       name="a_ub", tag="concat")

    a_l1 = tvm.compute(shape_a_z_bigz,
                       lambda *indices: a_ub(*indices),
                       name='a_l1',
                       tag="out_to_l1")
    b_l1 = tvm.compute(shape_b,
                       lambda *indices: weight(*indices),
                       name='b_l1',
                       tag="out_to_l1")

    a_l0a = tvm.compute(shape_a_z_bigz, lambda *indices: a_l1(*indices),
                        name="a_l0a", tag="l1_to_l0")
    b_l0b = tvm.compute(shape_b,
                        lambda *indices: b_l1(*indices),
                        name="b_l0b",
                        tag="l1_to_l0")

    k1 = tvm.reduce_axis((0, k_size), name='k1')
    k0 = tvm.reduce_axis((0, k0_size), name='k0')

    c_l0c = tvm.compute(shape_c,
                        lambda t, nb_0, nb_1, mb, mp, np:
                        tvm.sum((a_l0a[t, mb, k1, mp, k0] * \
                                 b_l0b[t, k1, nb_0, nb_1, np, k0]),
                                axis=[k1, k0]),
                        name='c_l0c',
                        tag="matmul")

    c_ub = tvm.compute(shape_c, lambda *indices: c_l0c(*indices), name="c_ub")

    bias_ub = tvm.compute(shape_bias,
                          lambda *indices: bias(*indices),
                          name='bias_ub')

    bias_ub_mid = bias_ub
    bias_bc_ub = broadcast(bias_ub_mid, shape_c)
    c_ub_bias = vadd(c_ub, bias_bc_ub)

    # split matmul res
    i_t_index = 0
    j_t_index = 3
    f_t_index = 1
    o_t_index = 2

    j_t = tvm.compute(shape_i,
                    lambda t, i, j, k, l: c_ub_bias(t, j_t_index, i, j, k, l),
                    name="j_t",
                    tag="split_com")
    shape_fio = (t_size, 3, hidden_size, m_size, 16, 16)
    f_i_o = tvm.compute(shape_fio, lambda t, x, i, j, k, l: c_ub_bias(t, x, i, j, k, l), name='f_i_o', tag="split_com")
    f_i_o_sigmoid = sigmoid_compute(f_i_o)
    f_t_sigmoid = tvm.compute(shape_i,
                lambda t, i, j, k, l: f_i_o_sigmoid(t, f_t_index, i, j, k, l),
                name="f_t_sigmoid", tag="split_com")
    i_t_sigmoid = tvm.compute(shape_i,
            lambda t, i, j, k, l: f_i_o_sigmoid(t, i_t_index, i, j, k, l),
            name="i_t_sigmoid", tag="split_com")
    o_t_sigmoid = tvm.compute(shape_i,
            lambda t, i, j, k, l: f_i_o_sigmoid(t, o_t_index, i, j, k, l),
            name="o_t_sigmoid", tag="split_com")
    j_t_tanh = tanh_compute(j_t)

    if ''.join([str(i) for i in shape_h]) == ''.join([str(i) for i in shape_i]):
        tensor_cont_ub = tensor_seq_length_bc_ub
    else:
        tensor_cont_ub = tensor_seq_length_ub
    tensor_seq_length_ub_conv = tensor_cont_ub
    if tensor_cont_ub.dtype != f_t_sigmoid.dtype:
        tensor_seq_length_ub_conv = tvm.compute(
            tensor_cont_ub.shape, lambda *i: tensor_cont_ub(*i).astype(f_t_sigmoid.dtype),
            name='tensor_seq_length_ub_conv', tag='elewise_single_cast'
            )
    tensor_seq_length_ub_bc_conv = tensor_seq_length_ub_conv
    if ''.join([str(i) for i in shape_h]) != ''.join([str(i) for i in shape_i]):
        tensor_seq_length_ub_bc_conv = broadcast(tensor_seq_length_ub_conv, shape_i)

    f_t_sigmoid_mul_cont = vmul(f_t_sigmoid, tensor_seq_length_ub_bc_conv)
    f_t_sigmoid_ub = f_t_sigmoid_mul_cont
    i_t_sigmoid_ub = i_t_sigmoid
    o_t_sigmoid_ub = o_t_sigmoid
    j_t_tanh_ub = j_t_tanh

    if is_gate_output:
        f_t_sigmoid_mid = f_t_sigmoid_mul_cont
        i_t_sigmoid_mid = i_t_sigmoid
        o_t_sigmoid_mid = o_t_sigmoid
        j_t_tanh_mid = j_t_tanh

        if fp16_input_output:
            f_t_sigmoid_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: f_t_sigmoid_mul_cont(*indices).astype('float16'),
                            name="f_t_sigmoid_fp16_drnn_cast",
                            tag="elewise_single_cast")
            i_t_sigmoid_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: i_t_sigmoid(*indices).astype('float16'),
                            name="i_t_sigmoid_fp16_drnn_cast",
                            tag="elewise_single_cast")
            o_t_sigmoid_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: o_t_sigmoid(*indices).astype('float16'),
                            name="o_t_sigmoid_fp16_drnn_cast",
                            tag="elewise_single_cast")
            j_t_tanh_fp16 = \
                tvm.compute(shape_i,
                            lambda *indices: j_t_tanh(*indices).astype('float16'),
                            name="j_t_tanh_fp16_drnn_cast",
                            tag="elewise_single_cast")
            f_t_sigmoid_mid = f_t_sigmoid_fp16
            i_t_sigmoid_mid = i_t_sigmoid_fp16
            o_t_sigmoid_mid = o_t_sigmoid_fp16
            j_t_tanh_mid = j_t_tanh_fp16

        f_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: f_t_sigmoid_mid(*indices),
                                     name="f_t_sigmoid_gm",
                                     tag="ub_to_out")
        i_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: i_t_sigmoid_mid(*indices),
                                     name="i_t_sigmoid_gm",
                                     tag="ub_to_out")
        o_t_sigmoid_gm = tvm.compute(shape_i,
                                     lambda *indices: o_t_sigmoid_mid(*indices),
                                     name="o_t_sigmoid_gm",
                                     tag="ub_to_out")
        j_t_tanh_gm = tvm.compute(shape_i,
                                  lambda *indices: j_t_tanh_mid(*indices),
                                  name="j_t_tanh_gm",
                                  tag="ub_to_out")

        f_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: f_t_sigmoid_gm(*indices),
                                       name="f_t_sigmoid_back",
                                       tag="out_to_ub")
        i_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: i_t_sigmoid_gm(*indices),
                                       name="i_t_sigmoid_back",
                                       tag="out_to_ub")
        o_t_sigmoid_back = tvm.compute(shape_i,
                                       lambda *indices: o_t_sigmoid_gm(*indices),
                                       name="o_t_sigmoid_back",
                                       tag="out_to_ub")
        j_t_tanh_back = tvm.compute(shape_i,
                                    lambda *indices: j_t_tanh_gm(*indices),
                                    name="j_t_tanh_back",
                                    tag="out_to_ub")

        if fp16_input_output:
            f_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: f_t_sigmoid_back(*indices).astype('float32'),
                                                name="f_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
            i_t_sigmoid_back_fp32 = tvm.compute(shape_i,
                                                lambda *indices: i_t_sigmoid_back(*indices).astype('float32'),
                                                name="i_t_sigmoid_back_fp32_drnn_cast",
                                                tag="elewise_single_cast")
            o_t_sigmoid_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: o_t_sigmoid_back(*indices).astype('float32'),
                name="o_t_sigmoid_back_fp32_drnn_cast",
                tag="elewise_single_cast")
            j_t_tanh_back_fp32 = tvm.compute(
                shape_i,
                lambda *indices: j_t_tanh_back(*indices).astype('float32'),
                name="j_t_tanh_back_fp32_drnn_cast",
                tag="elewise_single_cast")

        f_t_sigmoid_ub = f_t_sigmoid_back
        i_t_sigmoid_ub = i_t_sigmoid_back
        o_t_sigmoid_ub = o_t_sigmoid_back
        j_t_tanh_ub = j_t_tanh_back

        if fp16_input_output:
            f_t_sigmoid_ub = f_t_sigmoid_back_fp32
            i_t_sigmoid_ub = i_t_sigmoid_back_fp32
            o_t_sigmoid_ub = o_t_sigmoid_back_fp32
            j_t_tanh_ub = j_t_tanh_back_fp32

    # auto cast support both fp16 fp32
    c_t_tmp1 = vmul(s_state_c_ub, f_t_sigmoid_ub)
    c_t_tmp2 = vmul(j_t_tanh_ub, i_t_sigmoid_ub)
    update_c = vadd(c_t_tmp1, c_t_tmp2)
    update_c_gm = tvm.compute(shape_i,
                                lambda *indices: update_c(*indices),
                                name="update_c_gm",
                                tag="ub_to_out")
    update_c_fp16_back = tvm.compute(shape_i,
                                         lambda *indices: update_c_gm(*indices),
                                         name="update_c_fp16_back",
                                         tag="out_to_ub")
    c_t_tanh = tanh_compute(update_c_fp16_back)
    c_t_tanh_ub = c_t_tanh

    if is_gate_output:
        c_t_tanh_mid = c_t_tanh

        if fp16_input_output:
            c_t_tanh_fp16 = tvm.compute(shape_i,
                                        lambda *indices: c_t_tanh(*indices).astype('float16'),
                                        name="c_t_tanh_fp16_drnn_cast",
                                        tag="elewise_single_cast")
            c_t_tanh_mid = c_t_tanh_fp16
        c_t_tanh_gm = tvm.compute(shape_i,
                                  lambda *indices: c_t_tanh_mid(*indices),
                                  name="c_t_tanh_gm",
                                  tag="ub_to_out")
        c_t_tanh_back = tvm.compute(shape_i,
                                    lambda *indices: c_t_tanh_gm(*indices),
                                    name="c_t_tanh_back",
                                    tag="out_to_ub")

        if fp16_input_output:
            c_t_tanh_back_fp32 = tvm.compute(shape_i,
                                             lambda *indices: c_t_tanh_back(*indices).astype('float32'),
                                             name="c_t_tanh_back_fp32_drnn_cast",
                                             tag="elewise_single_cast")

        c_t_tanh_ub = c_t_tanh_back

        if fp16_input_output:
            c_t_tanh_ub = c_t_tanh_back_fp32

    update_h = vmul(c_t_tanh_ub, o_t_sigmoid_ub)
    update_h_gm_as_y = tvm.compute(shape_i,
                                    lambda *indices: update_h(*indices),
                                    name="update_h_gm_as_y",
                                    tag="ub_to_out")
    update_h_gm_as_y_back = tvm.compute(shape_i,
                                        lambda *indices: update_h_gm_as_y(*indices),
                                        name="update_h_gm_as_y_back",
                                        tag="out_to_ub")
    update_h_gm = tvm.compute(shape_i,
                                lambda *indices: update_h_gm_as_y_back(*indices),
                                name="update_h_gm",
                                tag="ub_to_out")

    # end compute

    if is_gate_output:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y, i_t_sigmoid_gm,
                       j_t_tanh_gm, f_t_sigmoid_gm, o_t_sigmoid_gm, c_t_tanh_gm]
    else:
        return_list = [update_h_gm, update_c_gm, update_h_gm_as_y]
    return_list, s = rl_bank.tik_dsl_bank_proc(return_list, sync_tensor=sync0)
    if s is not None:
        return return_list, s

    # schedule
    s = create_schedule([update_h_gm.op])

    def gen_reversed_subgraph_list(out_tensor, tensor_list):
        """
        traverse tensors by Depth-First-Search
        """
        if out_tensor is None:
            return
        stack = [out_tensor]
        visited_list = []
        while stack:
            cur_tensor = stack.pop()
            visited_list.append(cur_tensor)
            for in_tensor in cur_tensor.op.input_tensors:
                if in_tensor not in visited_list:
                    stack.append(in_tensor)
                    if "elewise" in in_tensor.op.tag or "broadcast" == in_tensor.op.tag:
                        if in_tensor.name.endswith("_drnn_cast"):
                            continue
                        if in_tensor.name in ["s_state_h_ub", "s_state_c_ub"]:
                            continue
                        if in_tensor not in tensor_list:
                            tensor_list.append(in_tensor)

    elewise_tensors = []
    gen_reversed_subgraph_list(update_h_gm, elewise_tensors)

    barrier_tensor = c_ub_bias
    elewise_before_barrier_tensors = [bias_bc_ub]

    # set scope
    s[a_l1].set_scope(scope_cbuf)
    s[b_l1].set_scope(scope_cbuf)
    s[a_l0a].set_scope(scope_ca)
    s[b_l0b].set_scope(scope_cb)
    s[c_l0c].set_scope(scope_cc)
    s[c_ub].set_scope(scope_ubuf)
    s[bias_ub].set_scope(scope_ubuf)
    s[bias_bc_ub].set_scope(scope_ubuf)
    s[tensor_seq_length_ub_bc_conv].set_scope(scope_ubuf)
    s[tensor_seq_length_bc_ub].set_scope(scope_ubuf)
    s[s_state_h_ub].set_scope(scope_ubuf)
    s[tensor_seq_length_ub].set_scope(scope_ubuf)
    if tensor_seq_length_ub.dtype != f_t_sigmoid.dtype:
        s[tensor_seq_length_ub_conv].set_scope(scope_ubuf)
    s[s_state_c_ub].set_scope(scope_ubuf)
    s[a_ub].set_scope(scope_ubuf)

    for tensor in elewise_tensors:
        s[tensor].set_scope(scope_ubuf)

    if is_gate_output:
        if fp16_input_output:
            s[f_t_sigmoid_fp16].set_scope(scope_ubuf)
            s[i_t_sigmoid_fp16].set_scope(scope_ubuf)
            s[o_t_sigmoid_fp16].set_scope(scope_ubuf)
            s[j_t_tanh_fp16].set_scope(scope_ubuf)
            s[c_t_tanh_fp16].set_scope(scope_ubuf)
            s[f_t_sigmoid_back_fp32].set_scope(scope_ubuf)
            s[i_t_sigmoid_back_fp32].set_scope(scope_ubuf)
            s[o_t_sigmoid_back_fp32].set_scope(scope_ubuf)
            s[j_t_tanh_back_fp32].set_scope(scope_ubuf)
            s[c_t_tanh_back_fp32].set_scope(scope_ubuf)

        s[f_t_sigmoid_back].set_scope(scope_ubuf)
        s[i_t_sigmoid_back].set_scope(scope_ubuf)
        s[o_t_sigmoid_back].set_scope(scope_ubuf)
        s[j_t_tanh_back].set_scope(scope_ubuf)
        s[c_t_tanh_back].set_scope(scope_ubuf)

    # fp16 in
    s[update_c_fp16_back].set_scope(scope_ubuf)
    s[update_h_gm_as_y_back].set_scope(scope_ubuf)

    # compute inline
    compute_inline_tensors = [j_t, f_i_o, i_t_sigmoid, f_t_sigmoid, o_t_sigmoid]
    for tensor in compute_inline_tensors:
        s[tensor].compute_inline()

    # matmul tiling
    factor_l1_m, factor_l1_n, factor_l1_k, \
    factor_l0_m, factor_l0_n, factor_l0_k =  1, 1, 12, 1, 1, 12

    l1_n_outer, l1_n_inner = \
        s[c_l0c].split(c_l0c.op.axis[2],
                       factor=factor_l1_n)

    l1_m_outer, l1_m_inner = \
        s[c_l0c].split(c_l0c.op.axis[3],
                       factor=factor_l1_m)
    l1_k_outer, l1_k_inner = \
        s[c_l0c].split(c_l0c.op.reduce_axis[0],
                       factor=factor_l1_k)

    l0_n_outer, l0_n_inner = s[c_l0c].split(l1_n_inner,
                                            factor=factor_l0_n)
    l0_m_outer, l0_m_inner = s[c_l0c].split(l1_m_inner,
                                            factor=factor_l0_m)
    l0_k_outer, l0_k_inner = s[c_l0c].split(l1_k_inner,
                                            factor=factor_l0_k)

    s[c_l0c].reorder(l1_n_outer, c_l0c.op.axis[1],
                     l1_m_outer, l1_k_outer,
                     l0_n_outer, l0_m_outer, l0_k_outer,
                     l0_n_inner, l0_m_inner, c_l0c.op.axis[3 + 1],
                     c_l0c.op.axis[4 + 1], l0_k_inner,
                     c_l0c.op.reduce_axis[1])

    s[a_ub].compute_at(s[c_l0c], l1_k_outer)
    s[s_state_h_mul_cont_ub].compute_at(s[c_l0c], l1_k_outer)

    s[a_l0a].compute_at(s[c_l0c], l0_k_outer)
    s[b_l0b].compute_at(s[c_l0c], l0_k_outer)
    s[a_l1].compute_at(s[c_l0c], l1_k_outer)

    ub_n_outer, ub_n_inner = \
        s[c_ub].split(c_ub.op.axis[2],
                      factor=factor_l1_n)

    ub_m_outer, ub_m_inner = s[c_ub].split(c_ub.op.axis[3],
                                           factor=factor_l1_m)
    s[c_ub].reorder(ub_m_outer, ub_n_outer, c_ub.op.axis[1],
                    ub_n_inner, ub_m_inner, c_ub.op.axis[4],
                    c_ub.op.axis[5])

    s[c_l0c].compute_at(s[c_ub], ub_n_outer)

    # elewise compute_at
    barrier_outer, barrier_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[2],
                                factor=factor_l1_n)
    barrier_m_outer, barrier_m_inner = \
        s[barrier_tensor].split(barrier_tensor.op.axis[3],
                                factor=factor_l1_m)
    s[barrier_tensor].reorder(
        barrier_tensor.op.axis[0], barrier_m_outer, barrier_outer,
        barrier_tensor.op.axis[1], barrier_inner, barrier_m_inner,
        barrier_tensor.op.axis[4],
        barrier_tensor.op.axis[5])

    s[c_ub].compute_at(s[barrier_tensor], barrier_outer)
    s[bias_ub].compute_at(s[barrier_tensor], barrier_outer)

    for tensor in elewise_before_barrier_tensors:
        s[tensor].compute_at(s[barrier_tensor], barrier_outer)

    vn_outer, vn_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 1],
                             factor=factor_l1_n)
    vn_m_outer, vn_m_inner = \
        s[update_h_gm].split(update_h_gm.op.axis[0 + 2],
                             factor=factor_l1_m)
    # second_split_factor default is (hidden_size // factor_l1_n) // 1
    second_split_factor = 8

    vn_o_outer, vn_o_inner = \
        s[update_h_gm].split(vn_outer,
                             factor=second_split_factor)
    s[update_h_gm].reorder(update_h_gm.op.axis[0], vn_m_outer,
                           vn_o_outer, vn_o_inner, vn_inner,
                           vn_m_inner, update_h_gm.op.axis[3],
                           update_h_gm.op.axis[4])

    s[s_state_c_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[s_state_h_ub].compute_at(s[update_h_gm], vn_o_inner)
    s[tensor_seq_length_ub_bc_conv].compute_at(s[update_h_gm], vn_o_inner)
    s[barrier_tensor].compute_at(s[update_h_gm], vn_o_inner)

    for tensor in elewise_tensors:
        if tensor not in elewise_before_barrier_tensors:
            s[tensor].compute_at(s[update_h_gm], vn_o_inner)

    s[update_c_gm].compute_at(s[update_h_gm], vn_o_inner)

    # fp16 in
    s[update_c_fp16_back].compute_at(s[update_h_gm], vn_o_inner)
    s[update_c].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y].compute_at(s[update_h_gm], vn_o_inner)
    s[update_h_gm_as_y_back].compute_at(s[update_h_gm], vn_o_inner)

    if is_gate_output:
        if fp16_input_output:
            s[f_t_sigmoid_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[i_t_sigmoid_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[o_t_sigmoid_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[j_t_tanh_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[c_t_tanh_fp16].compute_at(s[update_h_gm], vn_o_inner)
            s[f_t_sigmoid_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[i_t_sigmoid_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[o_t_sigmoid_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[j_t_tanh_back_fp32].compute_at(s[update_h_gm], vn_o_inner)
            s[c_t_tanh_back_fp32].compute_at(s[update_h_gm], vn_o_inner)

        s[f_t_sigmoid_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[i_t_sigmoid_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[o_t_sigmoid_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[j_t_tanh_gm].compute_at(s[update_h_gm], vn_o_inner)
        s[c_t_tanh_gm].compute_at(s[update_h_gm], vn_o_inner)

        s[f_t_sigmoid_back].compute_at(s[update_h_gm], vn_o_inner)
        s[i_t_sigmoid_back].compute_at(s[update_h_gm], vn_o_inner)
        s[o_t_sigmoid_back].compute_at(s[update_h_gm], vn_o_inner)
        s[j_t_tanh_back].compute_at(s[update_h_gm], vn_o_inner)
        s[c_t_tanh_back].compute_at(s[update_h_gm], vn_o_inner)

    if is_gate_output:
        s[f_t_sigmoid].reused_by(f_t_sigmoid_ub)
        s[f_t_sigmoid_mul_cont].reused_by(f_t_sigmoid_ub)
        s[i_t_sigmoid].reused_by(i_t_sigmoid_ub)
        s[o_t_sigmoid].reused_by(o_t_sigmoid_ub)
        s[j_t_tanh].reused_by(j_t_tanh_ub)
        s[c_t_tanh].reused_by(c_t_tanh_ub)

        s[f_t_sigmoid_ub].reused_by(reuse_data=True)
        s[i_t_sigmoid_ub].reused_by(reuse_data=True)
        s[o_t_sigmoid_ub].reused_by(reuse_data=True)
        s[j_t_tanh_ub].reused_by(reuse_data=True)
        s[c_t_tanh_ub].reused_by(reuse_data=True)

    s[update_c].reused_by(update_c_fp16_back)
    s[update_h].reused_by(update_h_gm_as_y_back)
    s[update_h_gm_as_y_back].reused_by(reuse_data=True)

    s[a_l1].double_buffer()
    s[b_l1].double_buffer()
    s[a_l0a].double_buffer()
    s[b_l0b].double_buffer()
    s[c_l0c].double_buffer()
    s[c_ub].double_buffer()

    # emit_insn
    s[a_l1].emit_insn(a_l1.op.axis[0], 'dma_copy')
    s[b_l1].emit_insn(b_l1.op.axis[0], 'dma_copy')
    s[a_l0a].emit_insn(a_l0a.op.axis[0], 'dma_copy')
    s[b_l0b].emit_insn(b_l0b.op.axis[0], 'dma_copy')

    s[tensor_seq_length_ub].emit_insn(tensor_seq_length_ub.op.axis[0], 'dma_copy')
    s[a_ub].emit_insn(a_ub.op.axis[0], 'dma_copy')

    mad_dict = {"mad_pattern": 0, "k_outer": [l1_k_outer, l0_k_outer]}
    s[c_l0c].emit_insn(l0_n_inner, 'mad', mad_dict)
    s[c_ub].emit_insn(ub_n_inner, 'dma_copy')

    s[bias_bc_ub].emit_insn(bias_bc_ub.op.axis[0], 'unified_broadcast')
    s[tensor_seq_length_bc_ub].emit_insn(tensor_seq_length_bc_ub.op.axis[0], 'unified_broadcast')

    if is_first_round:
        if is_global_init:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')
        else:
            s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'broadcast')
            s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'broadcast')
    else:
        s[s_state_h_ub].emit_insn(s_state_h_ub.op.axis[0], 'dma_copy')
        s[s_state_c_ub].emit_insn(s_state_c_ub.op.axis[0], 'dma_copy')

    s[barrier_tensor].emit_insn(barrier_tensor.op.axis[1], 'vector_add')

    for tensor in elewise_tensors:
        if tensor.op.name == 's_state_h_mul_cont_ub' or tensor.op.name == 'f_t_sigmoid_mul_cont':
            s[tensor].reorder(tensor.op.axis[2], tensor.op.axis[3], tensor.op.axis[1],
                        tensor.op.axis[0], tensor.op.axis[4])
        if tensor != barrier_tensor:
            insn = get_emit_insn_map(tensor)
            s[tensor].emit_insn(tensor.op.axis[0], insn)

    s[bias_ub].emit_insn(bias_ub.op.axis[0], 'dma_copy')

    s[update_c_gm].emit_insn(s[update_c_gm].op.axis[1], 'dma_copy')
    s[update_h_gm].emit_insn(s[update_h_gm].op.axis[3], 'dma_copy')

    block = tvm.thread_axis('blockIdx.x')
    s[update_h_gm].bind(vn_o_outer, block)
    s[update_h_gm].wait_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)
    s[update_h_gm].set_block_sync(axis=vn_m_outer, tensor=sync0[0], bottom=True)

    if is_gate_output:
        if fp16_input_output:
            s[f_t_sigmoid_fp16].emit_insn(s[f_t_sigmoid_fp16].op.axis[1],
                                          'vector_conv')
            s[i_t_sigmoid_fp16].emit_insn(s[i_t_sigmoid_fp16].op.axis[1],
                                          'vector_conv')
            s[o_t_sigmoid_fp16].emit_insn(s[o_t_sigmoid_fp16].op.axis[1],
                                          'vector_conv')
            s[j_t_tanh_fp16].emit_insn(s[j_t_tanh_fp16].op.axis[1],
                                       'vector_conv')
            s[c_t_tanh_fp16].emit_insn(s[c_t_tanh_fp16].op.axis[1],
                                       'vector_conv')
            s[f_t_sigmoid_back_fp32].emit_insn(
                s[f_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[i_t_sigmoid_back_fp32].emit_insn(
                s[i_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[o_t_sigmoid_back_fp32].emit_insn(
                s[o_t_sigmoid_back_fp32].op.axis[1], 'phony_insn')
            s[j_t_tanh_back_fp32].emit_insn(s[j_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')
            s[c_t_tanh_back_fp32].emit_insn(s[c_t_tanh_back_fp32].op.axis[1],
                                            'phony_insn')

        s[f_t_sigmoid_gm].emit_insn(s[f_t_sigmoid_gm].op.axis[1], 'dma_copy')
        s[i_t_sigmoid_gm].emit_insn(s[i_t_sigmoid_gm].op.axis[1], 'dma_copy')
        s[o_t_sigmoid_gm].emit_insn(s[o_t_sigmoid_gm].op.axis[1], 'dma_copy')
        s[j_t_tanh_gm].emit_insn(s[j_t_tanh_gm].op.axis[1], 'dma_copy')
        s[c_t_tanh_gm].emit_insn(s[c_t_tanh_gm].op.axis[1], 'dma_copy')

        s[f_t_sigmoid_back].emit_insn(s[f_t_sigmoid_back].op.axis[1],
                                      'phony_insn')
        s[i_t_sigmoid_back].emit_insn(s[i_t_sigmoid_back].op.axis[1],
                                      'phony_insn')
        s[o_t_sigmoid_back].emit_insn(s[o_t_sigmoid_back].op.axis[1],
                                      'phony_insn')
        s[j_t_tanh_back].emit_insn(s[j_t_tanh_back].op.axis[1], 'phony_insn')
        s[c_t_tanh_back].emit_insn(s[c_t_tanh_back].op.axis[1], 'phony_insn')

    # fp16 in
    s[update_c_fp16_back].emit_insn(update_c_fp16_back.op.axis[0],
                                        'phony_insn')
    s[update_h_gm_as_y].emit_insn(update_h_gm_as_y.op.axis[0], 'dma_copy')
    s[update_h_gm_as_y_back].emit_insn(update_h_gm_as_y_back.op.axis[0],
                                       'phony_insn')

    return return_list, s
