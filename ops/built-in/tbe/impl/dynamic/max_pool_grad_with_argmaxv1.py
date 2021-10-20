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
max_pool_grad_with_argmax_v1
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.dynamic.max_pool_grad_with_argmax_cut_one_h_v1 import MaxpoolGradCustom

# min shape of attr
ATTR_SHAPE_MIN = 4
# size of vector calc one repeat
ONE_REPEAT = 256
# size of one block
BLOCK_SIZE = 32
# max repeat of vector calc
V_MAX_REPEAT = 255
# max num of fp16 in one repeat
FP16_MAX = 128
# max num of fp32 in one repeat
FP32_MAX = 64
# max num of fp16 mask handle one time
MASK_MAX = 8
DT_INT32 = 3
DT_INT64 = 9
TILING_MODE0 = 0
TILING_MODE1 = 1
TILING_MODE2 = 2
TILING_MODE3 = 3
TILING_MODE4 = 4
TILING_MODE5 = 5
UB_SIZE = 262144
L1_SIZE = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)


# 'pylint: disable=invalid-name,too-many-arguments,useless-super-delegation,super-with-arguments
# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-return-statements,consider-using-in
@register_operator("MaxPoolGradWithArgmaxV1")
@para_check.check_input_type(dict, dict, dict, dict, (list, tuple), (list, tuple), (list, tuple), int, (list, tuple),
                             bool, str)
def max_pool_grad_with_argmax_v1(x, grad, argmax, y, ksize, strides, pads,
                                 dtype=DT_INT32,
                                 dilation=(1, 1, 1, 1), ceil_mode=False,
                                 kernel_name="max_pool_grad_with_argmax_v1"):
    """
    the main function of the maxpoolGradWithArgmax
    Parameters
    ----------
    x: input of maxpool, useless for maxpool gard
    grad: input of maxpoolgard or output of maxpool
    argmax:output of maxpool mask or index
    y: output of maxpoolgard
    ksize: kernel or windows size,minimum length is 4,
           just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like
    [1, poolingStrideH, poolingStrideW, 1]
    pads: pad list_int
    kernel_name: kernel_name
    Returns
    -------
    tik_instance: tik_instance
    :param ceil_mode:
    """
    check_param(x, grad, argmax, y, ksize, strides, pads, dtype, dilation,
                ceil_mode,
                kernel_name)
    maxpoolgard = MaxpoolGard(grad, argmax, x, ksize, strides, pads, dilation,
                              ceil_mode)
    return maxpoolgard.tik_instance_function(kernel_name)


class MaxpoolGard(MaxpoolGradCustom):
    """
    parameter for max_pool_grad_with_pool
    """

    def __init__(self, grad, argmax, input_x, ksize, strides, padding,
                 dilation, ceil_mode):
        """
        init compare and bit pack parameters
        Parameters
        ----------
        x: input of maxpool, useless for maxpool gard
        grad: input of maxpoolgard or output of maxpool
        argmax:output of maxpool mask or index
        strides: stride , minimum length is 4, just like
        [1, poolingStrideH, poolingStrideW, 1]
        padding: pad mode, just support "SANME" or "VALID"
        Returns
        -------
        None
        """
        super(MaxpoolGard, self).__init__(grad, argmax, input_x, ksize, strides, padding, dilation, ceil_mode)
        self.pad_h, self.pad_w = self.padding[1:3]
        self.ori_stride_h, self.ori_stride_w = self.strides[1:3]
        self.dilation_h, self.dilation_w = self.dilation[1:3]

    def tik_instance_function(self, kernel_name):
        """
        tik_instance function
        """
        with self.tik_instance.for_range(0, self.blocknum, block_num=self.blocknum) as block_id:
            self.get_tiling_params()
            with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE0):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(block_id < self.real_block):
                        self.tik_instance_cut_nc1_cut_one_h(block_id)

            with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE1):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(block_id < self.real_block):
                        self.tik_instance_cut_nc1_cut_w(block_id)

            with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE2):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(block_id < self.real_block):
                        self.tik_instance_cut_nc1_cut_h(block_id)

            with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE3):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(block_id < self.real_block):
                        self.tik_instance_cut_nc1h_cut_one_h(block_id)

            with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE4):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(block_id < self.real_block):
                        self.tik_instance_cut_nc1h_cut_w(block_id)

            with self.tik_instance.if_scope(self.tiling_mode == TILING_MODE5):
                with self.tik_instance.new_stmt_scope():
                    with self.tik_instance.if_scope(block_id < self.real_block):
                        self.tik_instance_cut_nc1h_cut_h(block_id)

        self.tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[self.data_input_origin, self.data_input,
                                                                    self.data_mask],
                                   outputs=[self.data_output], flowtable=[self.tiling_gm])

        ceil_mode_int = 1 if self.ceil_mode else 0
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.blocknum, "ub_size": UB_SIZE,
                                                            "l1_size": L1_SIZE, "kernel_h": self.kernel_h,
                                                            "kernel_w": self.kernel_w, "stride_h": self.ori_stride_h,
                                                            "stride_w": self.ori_stride_w, "pad_h": self.pad_h,
                                                            "pad_w": self.pad_w, "dilation_h": self.dilation_h,
                                                            "dilation_w": self.dilation_w, "ceil_mode": ceil_mode_int,
                                                            "dtype_size": self.dtype_size})


def check_output_dim_with_ksize_stride(ksize, strides):
    """
    The common check rule for output dim and ksize and strides
    """
    para_check.check_shape_size(ksize)
    para_check.check_shape_size(strides)
    if len(ksize) < ATTR_SHAPE_MIN or len(strides) < ATTR_SHAPE_MIN:
        raise RuntimeError(
            "The shape length of ksize or strides must be more than 4")
    if ksize[0] != 1 or ksize[3] != 1:
        raise RuntimeError(
            "MaxPoolGradWithArgmax only supports pooling across width/height,"
            "and other ksize dimension should be one")
    if strides[0] != 1 or strides[3] != 1:
        raise RuntimeError(
            "MaxPoolGradWithArgmax only supports pooling across width/height,"
            "and other strides dimension should be one")
    if ksize[1] * ksize[2] > 255:
        raise RuntimeError(
            "invalid window params, window_h*window_w should be <=255")


# 'pylint: disable=unused-argument
def check_param(x, grad, argmax, y, ksize, strides, padding, dtype, dilation,
                ceil_mode, kernel_name):
    """
    check the parameters is valid, if one is invalid,then raise error
    Parameters
    ----------
    x: dict,shape and datatype
    grad: dict,shape and datatype
    argmax: dict,shape and datatype
    y: dict,shape and datatype
    ksize: kernel or windows size,minimum length is 4,
          just like [1, poolingWindowH, poolingWindowW, 1]
    strides: stride , minimum length is 4, just like
    [1, poolingStrideH, poolingStrideW, 1]
    padding: pad mode
    Returns
    -------
    None
    """
    y_dtype = x.get("dtype").lower()
    y_dtype_arg = y.get("dtype").lower()
    grad_dtype = grad.get("dtype").lower()
    argmax_dtype = argmax.get("dtype").lower()
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(grad_dtype, ("float16", "float32", "int32"))
    para_check.check_dtype_rule(argmax_dtype, ("uint16"))
    para_check.check_dtype_rule(y_dtype, ("float16", "float32", "int32"))

    if y_dtype != grad_dtype or y_dtype_arg != y_dtype:
        raise RuntimeError(
            "The dtype of tensor must be same")

    if dtype != DT_INT32 and dtype != DT_INT64:
        raise RuntimeError(
            "The dtype of input max indice must be int32 or int64")

    check_output_dim_with_ksize_stride(ksize, strides)
