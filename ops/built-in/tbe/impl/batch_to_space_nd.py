# Copyright 2021 Huawei Technologies Co., Ltd
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
batch_to_space_nd
"""
from te.utils import para_check
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# pylint: disable=invalid-name,unused-argument,too-many-locals,unnecessary-pass,too-many-return-statements
def check_supported(x, block_shape, crops, y, kernel_name="batch_to_space_nd"):
    """check supported dynamiclly.
    only spported format NHWC,NCHW,NC1HWC0,NDHWC,NCDHW,NDC1HWC0
    ori_format:NHWC
        ori shape must be 4([-1,-1,-1,-1]), block_shape must be 1([2]), crops must be 2([2,2])
        ori shape must be 3([-1,-1,-1]), block_shape must be 1([1]), crops must be 2([1,2])
    ori format:NCHW
        ori shape must be 4([-1,-1,-1,-1]), block_shape must be 1([3]), crops must be 2([3,2])
    ori format:NDHWC
        ori shape must be 5([-1,-1,-1,-1,-1]), block_shape must be 1([3]), crops must be 2([3,2])
    ori format:NCDHW
        ori shape must be 5([-1,-1,-1,-1,-1]), block_shape must be 1([4]), crops must be 2([4,2])
    """
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")
    block_s = block_shape.get("shape")
    crop_s = crops.get("shape")
    if ori_format not in ("NHWC", "NCHW", "NDHWC", "NCDHW"):
        return False
    if len(block_s) != 1 or len(crop_s) != 2 or crop_s[1] != 2:
        return False
    if ori_format in ("NHWC",):
        if len(ori_shape) != 4 or block_s[0] != 2 or crop_s[0] != 2:
            if len(ori_shape) != 3 or block_s[0] != 1 or crop_s[0] != 1:
                return False
    elif ori_format in ("NCHW",):
        if len(ori_shape) != 4 or block_s[0] != 3 or crop_s[0] != 3:
            return False
    elif ori_format in ("NDHWC",):
        if len(ori_shape) != 5 or block_s[0] != 3 or crop_s[0] != 3:
            return False
    elif ori_format in ("NCDHW",):
        if len(ori_shape) != 5 or block_s[0] != 4 or crop_s[0] != 4:
            return False

    return True


def op_select_format(x, block_shape, crops, y, kernel_name="batch_to_space_nd"):
    """select format dynamiclly.
    op_select_format support desc:
        1. when ori_format is 'NHWC' or 'NCHW', input_format is 'NC1HWC0'
            for example:
                ori:
                    x              shape = [16,16,16,16]           format = 'NHWC'
                    block_shape    shape = [2,]                    format = 'ND'
                    crops          shape = [2,2]                   format = 'ND'
                    y              shape = [None,None,None,16]     format = 'NHWC'
                format transformer:
                    x              shape = [16,1,16,16,16]         format = 'NC1HWC0'
                    block_shape    shape = [2,]                    format = 'ND'
                    crops          shape = [2,2]                   format = 'ND'
                    y              shape = [None,1,None,None,16]   format = 'NC1HWC0'
        2. when ori_format is 'NDHWC' or 'NCDHW', input_format is 'NDC1HWC0'
            for example:
                ori:
                    x              shape = [16,16,16,16,16]              format = 'NDHWC'
                    block_shape    shape = [3,]                          format = 'ND'
                    crops          shape = [3,2]                         format = 'ND'
                    y              shape = [None,None,None,None,16]      format = 'NDHWC'
                format transformer:
                    x              shape = [16,16,1,16,16,16]            format = 'NDC1HWC0'
                    block_shape    shape = [3,]                          format = 'ND'
                    crops          shape = [3,2]                         format = 'ND'
                    y              shape = [None,None,1,None,None,16]    format = 'NDC1HWC0'
    """
    input_dtype = "float16, float, float16, float"
    input_format = "NC1HWC0, NC1HWC0, NC1HWC0, NC1HWC0"
    ori_format = x.get("ori_format")
    if ori_format in ("NDHWC", "NCDHW"):
        input_dtype = "float16, float, float16, float"
        input_format = "NDC1HWC0, NDC1HWC0, NDC1HWC0, NDC1HWC0"

    attr_dtype = "int32, int32, int64, int64"
    attr_format = "ND, ND, ND, ND"
    input0 = gen_param(classify="input0",
                       name="x",
                       datatype=input_dtype,
                       format=input_format,
                       unknownshape_format=input_format)
    input1 = gen_param(classify="input1",
                       name="block_shape",
                       datatype=attr_dtype,
                       format=attr_format,
                       unknownshape_format=attr_format)
    input2 = gen_param(classify="input2",
                       name="crops",
                       datatype=attr_dtype,
                       format=attr_format,
                       unknownshape_format=attr_format)
    output0 = gen_param(classify="output0",
                        name="y",
                        datatype=input_dtype,
                        format=input_format,
                        unknownshape_format=input_format)

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def batch_to_space_nd(x, block_shape, crops, y, kernel_name="batch_to_space_nd"):
    """BatchToSpaceND for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    block_shape: dict
        the dict of block_shape tensor.
    crops: dict
        the dict of crops tensor.
    y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "batch_to_space_nd".

    Returns
    -------
    None.
    """
    pass
