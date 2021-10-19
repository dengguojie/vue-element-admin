#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
drop_out_do_mask_v3_d
"""
import te.platform as tbe_platform

from te import tik
from te.utils import para_check

SHAPE_SIZE_LIMIT = 1 << 30


# pylint: disable = unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def softmax_v2_with_drop_out_do_mask_v3_d(input_tensor, input_mask, output_1, output_2, input_keep_prob, axis=-1,
                                          kernel_name="softmax_v2_with_drop_out_do_mask_v3_d"):
    """
    softmax_v2 + drop_out_do_mask_v3_d

    Parameters
    ----------
    input_tensor : dict,shape and dtype of input_tensor,only support float16
    input_mask : dict,shape and dtype of input_mask
        shape of mask,1D, dtype == uint8
    input_keep_prob : dict,shape and dtype of input_keep_prob
        shape of keep_prob, only 1 parament and equals to (1)
        prob scale (0.0,1.0] NOTICE: type same as dytpe
    output1 : dict,shape and dtype of output1
    output2 : dict,shape and dtype of output2
    kernel_name : str
        cce kernel name, default value is "softmax_v2_with_drop_out_do_mask_v3_d"

    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(input_tensor.get('dtype').lower(), ("float16"))
    para_check.check_dtype_rule(input_mask.get('dtype').lower(), ("uint8"))
    para_check.check_shape_rule(input_tensor.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_rule(input_mask.get('shape'), max_shape_num=SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_tensor.get('shape'), SHAPE_SIZE_LIMIT)
    para_check.check_shape_size(input_mask.get('shape'), SHAPE_SIZE_LIMIT)

    tensor_shape = input_tensor.get("shape")
    mask_shape = input_mask.get("shape")
    tensor_dtype = input_tensor.get("dtype").lower()
    mask_dtype = input_mask.get("dtype").lower()
    tik_inst = tik.Tik(tik.Dprofile(), disable_debug=False)
    tensor_input = tik_inst.Tensor(tensor_dtype,
                                   tensor_shape,
                                   name="tensor_input",
                                   scope=tbe_platform.scope_gm)
    mask_input = tik_inst.Tensor(mask_dtype,
                                 mask_shape,
                                 name="mask_input",
                                 scope=tbe_platform.scope_gm)
    output1 = tik_inst.Tensor(tensor_dtype,
                              tensor_shape,
                              name="output1",
                              scope=tbe_platform.scope_gm)
    output2 = tik_inst.Tensor(tensor_dtype,
                              tensor_shape,
                              name="output2",
                              scope=tbe_platform.scope_gm)
    aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    block_count = tensor_shape[0] * tensor_shape[1] / aicore_num

    with tik_inst.for_range(0, aicore_num, block_num=aicore_num) as blockid:
        with tik_inst.for_range(0, block_count) as i:
            with tik_inst.for_range(0, 16) as j:
                ub_1 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_1")
                ub_2 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_2")
                ub_cast = tik_inst.Tensor("float32", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_cast")
                ub_3 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_3")
                ub_4 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_4")

                ub_reducemax = tik_inst.Tensor("float16", (32,), scope=tbe_platform.scope_ubuf, name="ub_reducemax")
                ub_reduceadd = tik_inst.Tensor("float32", (32,), scope=tbe_platform.scope_ubuf, name="ub_reduceadd")
                ub_reduceadd_fp16 = tik_inst.Tensor("float16", (32,), scope=tbe_platform.scope_ubuf,
                                                    name="ub_reduceadd_fp16")
                ub_dup = tik_inst.Tensor("uint16", (128,), scope=tbe_platform.scope_ubuf, name="ub_dup")
                ub_broadcast = tik_inst.Tensor("uint16", (32 * 16,), scope=tbe_platform.scope_ubuf,
                                               name="ub_broadcast")

                ub_mask = tik_inst.Tensor("uint8", (32, 32, 16), scope=tbe_platform.scope_ubuf, name="ub_mask")
                ub_mask_fp16 = tik_inst.Tensor("float16", (32, 32, 16), scope=tbe_platform.scope_ubuf,
                                               name="ub_mask_fp16")

                tik_inst.data_move(ub_1[0], tensor_input[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   0, 32, 32, 480, 0)
                tik_inst.data_move(ub_mask[0], mask_input[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   0, 32, 16, 240, 0)
                tik_inst.vconv(128, "", ub_mask_fp16[0], ub_mask[0], 128, 1, 1, 8, 4)

                tik_inst.vmax(128, ub_2[0], ub_1[0], ub_1[8192], 64, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[4096], 32, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[2048], 16, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[1024], 8, 1, 1, 1, 8, 8, 8)
                tik_inst.vmax(128, ub_2[0], ub_2[0], ub_2[512], 4, 1, 1, 1, 8, 8, 8)
                tik_inst.vcgmax(128, ub_reducemax[0], ub_2[0], 4, 1, 1, 8)

                tik_inst.vector_dup(128, ub_dup[0], tik_inst.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)
                ub_reducemax_int16 = ub_reducemax.reinterpret_cast_to("uint16")
                tik_inst.vor(16, ub_broadcast[0], ub_reducemax_int16[0], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)
                tik_inst.vor(16, ub_broadcast[256], ub_reducemax_int16[16], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)

                tik_inst.vtranspose(ub_broadcast[0], ub_broadcast[0])
                tik_inst.vtranspose(ub_broadcast[256], ub_broadcast[256])

                ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

                with tik_inst.for_range(0, 4) as idx:
                    tik_inst.vsub(128, ub_2[idx * 128], ub_1[idx * 128], ub_broadcast_fp16[idx * 128],
                                  32, 1, 1, 1, 32, 32, 0)

                tik_inst.vconv(64, "", ub_cast[0], ub_2[0], 255, 1, 1, 8, 4)
                tik_inst.vconv(64, "", ub_cast[16320], ub_2[16320], 1, 1, 1, 8, 4)

                tik_inst.vexp(64, ub_cast[0], ub_cast[0], 255, 1, 1, 8, 8)
                tik_inst.vexp(64, ub_cast[16320], ub_cast[16320], 1, 1, 1, 8, 8)

                tik_inst.vconv(64, "", ub_3[0], ub_cast[0], 255, 1, 1, 4, 8)
                tik_inst.vconv(64, "", ub_3[16320], ub_cast[16320], 1, 1, 1, 4, 8)

                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[8192], 128, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[4096], 64, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[2048], 32, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[1024], 16, 1, 1, 1, 8, 8, 8)
                tik_inst.vadd(64, ub_cast[0], ub_cast[0], ub_cast[512], 8, 1, 1, 1, 8, 8, 8)
                tik_inst.vcadd(16, ub_reduceadd[0], ub_cast[0], 32, 1, 1, 2)

                tik_inst.vrec(32, ub_reduceadd[0], ub_reduceadd[0], 1, 1, 1, 0, 0)

                tik_inst.vconv(32, "", ub_reduceadd_fp16[0], ub_reduceadd[0], 1, 1, 1, 0, 0)

                tik_inst.vector_dup(128, ub_dup[0], tik_inst.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)

                ub_reduceadd_int16 = ub_reduceadd_fp16.reinterpret_cast_to("uint16")
                tik_inst.vor(16, ub_broadcast[0], ub_reduceadd_int16[0], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)
                tik_inst.vor(16, ub_broadcast[256], ub_reduceadd_int16[16], ub_dup[0], 16, 1, 1, 0, 1, 0, 0)

                tik_inst.vtranspose(ub_broadcast[0], ub_broadcast[0])
                tik_inst.vtranspose(ub_broadcast[256], ub_broadcast[256])

                ub_broadcast_fp16 = ub_broadcast.reinterpret_cast_to("float16")

                with tik_inst.for_range(0, 4) as idx:
                    tik_inst.vmul(128, ub_4[idx * 128], ub_3[idx * 128], ub_broadcast_fp16[idx * 128],
                                  32, 1, 1, 1, 32, 32, 0)

                tik_inst.vmuls(128, ub_2[0], ub_4[0],
                               tik_inst.Scalar(init_value=1 / input_keep_prob, dtype="float16"), 128, 1, 1, 8, 8)
                tik_inst.vmul(128, ub_3[0], ub_mask_fp16[0], ub_2[0], 128, 1, 1, 1, 8, 8, 8)

                tik_inst.data_move(output1[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   ub_4[0], 0, 32, 32, 0, 480)
                tik_inst.data_move(output2[blockid * block_count * 262144 + i * 262144 + j * 512],
                                   ub_3[0], 0, 32, 32, 0, 480)

    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[tensor_input, mask_input], outputs=[output1, output2])
    return tik_inst
