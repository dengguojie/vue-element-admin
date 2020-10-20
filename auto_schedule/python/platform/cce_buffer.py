# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
register the cce buffer info
"""
from __future__ import absolute_import as _abs

from te.platform import cce_conf
from te.platform import cce_params

from te.tvm._ffi.function import register_func
from te.tvm import make as _make
from te.tvm import api as tvm

# pylint: disable=invalid-name
# add default product, default value is Ascend310
# get the CceProductParams instance
cur_cce_product_params = cce_conf.te_set_version("Ascend310")


@register_func("te.cce.cur_buf_params")
def cur_product_params(name):
    """ api for c++ pass to get current product params"""
    ret = cur_cce_product_params.getParams(name)
    return tvm.const(ret, 'int32')


# The memory information for the compiler
# ub 32B aligned, L1 and L0 2*16^2B aligned
L1_UNIT_BITS = 2*16*16*8
L1_MAX_SIMD_BITS = 2*16*16*8


@register_func("tvm.info.mem.%s" % cce_params.scope_cbuf)
def mem_info_l1_buffer():
    """
    make node info L1 buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=L1_UNIT_BITS,
                      max_simd_bits=L1_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("L1_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


@register_func("tvm.info.mem.%s" % cce_params.scope_cbuf_fusion)
def mem_info_l1_fusion_buffer():
    """
    make node info L1 buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=L1_UNIT_BITS,
                      max_simd_bits=L1_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("L1_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


UB_UNIT_BITS = 32*8
UB_MAX_SIMD_BITS = 32*8


@register_func("tvm.info.mem.%s" % cce_params.scope_ubuf)
def mem_info_ub_buffer():
    """
    make node info UB buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=UB_UNIT_BITS,
                      max_simd_bits=UB_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("UB_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


# The memory information for the compiler
# smask 32B aligned
SMASK_UNIT_BITS = 32*8
SMASK_MAX_SIMD_BITS = 32*8


@register_func("tvm.info.mem.%s" % cce_params.scope_smask)
def mem_info_smask_buffer():
    """
    make node info SMASK buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=SMASK_UNIT_BITS,
                      max_simd_bits=SMASK_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("SMASK_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


L0A_UNIT_BITS = 2*16*16*8
L0A_MAX_SIMD_BITS = 2*16*16*8


@register_func("tvm.info.mem.%s" % cce_params.scope_ca)
def mem_info_l0a_buffer():
    """
    make node info L0A buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=L0A_UNIT_BITS,
                      max_simd_bits=L0A_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("L0A_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


L0B_UNIT_BITS = 2*16*16*8
L0B_MAX_SIMD_BITS = 2*16*16*8


@register_func("tvm.info.mem.%s" % cce_params.scope_cb)
def mem_info_l0b_buffer():
    """
    make node info L0B buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=L0B_UNIT_BITS,
                      max_simd_bits=L0B_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("L0B_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


L0C_UNIT_BITS = 2*16*16*8
L0C_MAX_SIMD_BITS = 2*16*16*8


@register_func("tvm.info.mem.%s" % cce_params.scope_cc)
def mem_info_l0c_buffer():
    """
    make node info L0C buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=L0C_UNIT_BITS,
                      max_simd_bits=L0C_MAX_SIMD_BITS,
                      max_num_bits=cce_conf.get_soc_spec("L0C_SIZE")*8,
                      head_address=tvm.const(0, 'int32'))


REG_UNIT_BITS = 16
REG_MAX_SIMD_BITS = 64
REG_MAX_NUM_BITS = 64*3200


@register_func("tvm.info.mem.%s" % cce_params.scope_reg)
def mem_info_reg_buffer():
    """
    make node info Reg buffer
    """
    return _make.node("MemoryInfo",
                      unit_bits=REG_UNIT_BITS,
                      max_simd_bits=REG_MAX_SIMD_BITS,
                      max_num_bits=REG_MAX_NUM_BITS,
                      head_address=tvm.const(0, 'int32'))


AICPU_UNIT_BITS = 16
AICPU_MAX_SIMD_BITS = 64
AICPU_MAX_NUM_BITS = 16*1024*1024  # AICPU stack memory limit is 2M


@register_func("tvm.info.mem.%s" % cce_params.scope_aicpu)
def mem_info_ai_cpu():
    """
    make node info Ai_CPU
    """
    return _make.node("MemoryInfo",
                      unit_bits=AICPU_UNIT_BITS,
                      max_simd_bits=AICPU_MAX_SIMD_BITS,
                      max_num_bits=AICPU_MAX_NUM_BITS,
                      head_address=tvm.const(0, 'int32'))
