# -*- coding:utf-8 -*-
from te import tvm
from te.platform import cce_params as param
from te import platform as tbe_platform
from te.platform.cce_conf import CceProductParams

ARGS_STR_V = tvm.call_pure_intrin("int32", "tvm_cce_string_print", "PIPE_V")
ARGS_STR_V_ALL = tvm.call_pure_intrin("int32", "tvm_cce_string_print", "PIPE_ALL")


class GlobalParams:
    def __init__(self, inputs, outputs):
        # get device version
        version = tbe_platform.cce_conf.get_soc_spec("SOC_VERSION")
        if version in ("Ascend910"):
            self.devices = "1980"
            self.core_counts = 32
        elif version in ("Ascend310"):
            self.devices = "1910"
            self.core_counts = 2
        else:
            raise RuntimeError("Only support target:cloud_v100/mini_v100")

        # use multi-core resource
        self.block = tvm.thread_axis("blockIdx.x")
        # get dma sid
        self.sid = int(CceProductParams().getParams("Sid_copy_gm_to_cbuf"))
        self.pad_mode = tvm.call_pure_intrin("int32", "tvm_cce_string_print", "PAD_NONE")
        self.dtype = inputs.dtype

        # get input\output shape
        size_in = []
        for i in inputs.shape:
            size_in.append(i.value)
        self.size_in = size_in
        size_out = []
        for i in outputs.shape:
            size_out.append(i.value)
        self.size_out = size_out
        self.h_in = size_in[-3]
        self.w_in = size_in[-2]
        self.h_out = size_out[-3]
        self.w_out = size_out[-2]

        # calculate scale
        self.scale_w = float(self.w_in - 1) / float(self.w_out - 1) if self.w_out > 1 else 0.0
        self.scale_h = float(self.h_in - 1) / float(self.h_out - 1) if self.h_out > 1 else 0.0
        # c0 declare
        self.f32_c0 = 8
        self.f16_c0 = 16
        self.c0 = 8 if self.dtype == "float32" else 16

        self.inputs = inputs
        self.outputs = outputs


class LargeDataFp16Param:
    def __init__(self, expand_loop, l1_half, f16_stride, f16_size, gap_limit, reduce_size,
                 in_l1, burst_limit, f16_out, out_ub_f32):
        self.expand_loop = expand_loop
        self.l1_half = l1_half
        self.f16_stride = f16_stride
        self.f16_size = f16_size
        self.gap_limit = gap_limit
        self.reduce_size = reduce_size
        self.in_l1 = in_l1
        self.burst_limit = burst_limit
        self.f16_out = f16_out
        self.out_ub_f32 = out_ub_f32


class LargeDataFp32Param:
    def __init__(self, expand_loop, l1_half, f32_stride, f32_size, gap_limit, reduce_size,
                 in_l1, burst_limit, f32_out, f32_in):
        self.expand_loop = expand_loop
        self.l1_half = l1_half
        self.f32_stride = f32_stride
        self.f32_size = f32_size
        self.gap_limit = gap_limit
        self.reduce_size = reduce_size
        self.in_l1 = in_l1
        self.burst_limit = burst_limit
        self.f32_out = f32_out
        self.f32_in = f32_in


class NormalSituationParam:
    def __init__(self, para):
        self.h_per_core = para.h_out // para.core_counts + (1 if para.h_out % para.core_counts > 0 else 0)
        self.is_same_percore = 0 if para.h_out % para.core_counts == 0 else 1
        self.core_num = para.h_out // self.h_per_core + self.is_same_percore

        if (para.w_in + 1) * 16 < 512 * 8 and para.dtype == "float32":
            self.is_input_in_ub = True
        else:
            self.is_input_in_ub = False

        self.loop_levelx = para.w_out // 256 + (1 if para.w_out % 256 > 0 else 0)
        self.loop_levely = self.h_per_core // 256 + (1 if self.h_per_core % 256 > 0 else 0)


def apply_store_buffer(ib, dtype, shape, name="store_buf", scope=param.scope_ubuf):
    """
        apply storage space
    """
    buf_var = ib.allocate(dtype, shape, name=name, scope=scope)
    return tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)


def apply_reg_buffer(ib, dtype, shape, name="reg_buf"):
    """
        apply register space
    """
    return ib.allocate(dtype, shape, name=name, scope=param.scope_reg)
