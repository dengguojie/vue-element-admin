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
common_util
"""
from te import tik
from impl import constant_util as constant
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe


def get_vector_repeat_times(tik_instance, total_size):
    """
    get vector instruct repeat times

    Parameters
    ----------
    tik_instance: tik_instance
    total_size: the byte of the data

    Returns
    -------
    repeats: repeat times of vector instructs
    """
    repeats = tik_instance.Scalar(constant.DATA_TYPE_INT32)
    repeats.set_as(total_size % constant.VECTOR_BYTE_SIZE)
    with tik_instance.if_scope(repeats == 0):
        repeats.set_as(total_size // constant.VECTOR_BYTE_SIZE)
    with tik_instance.else_scope():
        repeats.set_as(total_size // constant.VECTOR_BYTE_SIZE + 1)

    return repeats


def get_datamove_nburst(tik_instance, total_size):
    """
    get datamove nburst

    Parameters
    ----------
    tik_instance: tik_instance
    total_size: the byte of the data

    Returns
    -------
    nburst: one burst indicating the continueous transfer length
                     in terms of the block size.
    """
    nburst = tik_instance.Scalar(constant.DATA_TYPE_INT32)
    nburst.set_as(total_size % constant.BLOCK_SIZE)
    with tik_instance.if_scope(nburst == 0):
        nburst.set_as(total_size / constant.BLOCK_SIZE)
    with tik_instance.else_scope():
        nburst.set_as(total_size / constant.BLOCK_SIZE + 1)

    return nburst


def get_data_size(datatype):
    """
    get data size unit, means one element of this input datatype takes up nbyte space

    Parameters
    ----------
    datatype: datatype supports float32,float16,int32,int16,int8,uint32,uint16,uint8

    Returns
    -------
    data_size: one element of this input datatype takes up nbyte space
    """
    datatype_map = {constant.DATA_TYPE_FP32: constant.DATA_SIZE_FOUR,
                    constant.DATA_TYPE_FP16: constant.DATA_SIZE_TWO,
                    constant.DATA_TYPE_INT32: constant.DATA_SIZE_FOUR,
                    constant.DATA_TYPE_INT16: constant.DATA_SIZE_TWO,
                    constant.DATA_TYPE_INT8: constant.DATA_SIZE_ONE,
                    constant.DATA_TYPE_UINT32: constant.DATA_SIZE_FOUR,
                    constant.DATA_TYPE_UINT16: constant.DATA_SIZE_TWO,
                    constant.DATA_TYPE_UINT8: constant.DATA_SIZE_ONE,
                    constant.DATA_TYPE_UINT64: constant.DATA_SIZE_EIGHT,
                    constant.DATA_TYPE_INT64: constant.DATA_SIZE_EIGHT
                    }
    data_size = datatype_map.get(datatype)
    if data_size is None:
        raise RuntimeError("datatype %s is not support!" % (datatype))

    return data_size


def move_out_non32_alignment(input_dict):
    """
  move data from ub to gm when non32 alignment
  usage scenarios: multi core moves out of the scene for the last time,
  in order to prevent covering the data of other core

  Parameters
  ----------
    input_dict: input_dict is a dict, the keys as follow:
            instance: tik instance
            out_ub: a ub tensor
            out_gm: a gm tensor
            gm_offset: a scalar,gm offset
            element_num: element number
            dsize: data size of each type,fp32,data size is 4,
                   fp16 data size is 2 and so on
  Returns
  -------
  None
  """
    instance = input_dict.get("instance")
    out_ub = input_dict.get("out_ub")
    out_gm = input_dict.get("out_gm")
    gm_offset = input_dict.get("gm_offset")
    element_num = input_dict.get("element_num")
    dsize = input_dict.get("dsize")
    each_burst_num = constant.BLOCK_SIZE // dsize
    out_ub_tmp = instance.Tensor(out_ub.dtype, (each_burst_num,),
                                 name="out_ub_tmp",
                                 scope=tik.scope_ubuf)
    nbursts = instance.Scalar("int32")
    nbursts.set_as((element_num * dsize) // constant.BLOCK_SIZE)
    scalar = instance.Scalar(out_ub.dtype)
    mod = instance.Scalar("int32")
    mod.set_as((element_num * dsize) % constant.BLOCK_SIZE)

    # 32b alignment
    with instance.if_scope(mod == 0):
        instance.data_move(out_gm[gm_offset], out_ub, constant.SID,
                           constant.DEFAULT_NBURST,
                           nbursts, constant.STRIDE_ZERO, constant.STRIDE_ZERO)

    # less than 32b
    with instance.if_scope(nbursts == 0):
        offset = each_burst_num - element_num
        instance.data_move(out_ub_tmp,
                           out_gm[gm_offset - offset],
                           constant.SID, constant.DEFAULT_NBURST,
                           constant.DEFAULT_BURST_LEN,
                           constant.STRIDE_ZERO,
                           constant.STRIDE_ZERO)

        with instance.for_range(0, element_num) as out_cycle:
            scalar.set_as(out_ub[out_cycle])
            out_ub_tmp[offset + out_cycle].set_as(scalar)
        instance.data_move(out_gm[gm_offset - offset],
                           out_ub_tmp,
                           constant.SID, constant.DEFAULT_NBURST,
                           constant.DEFAULT_BURST_LEN,
                           constant.STRIDE_ZERO, constant.STRIDE_ZERO)
    # bigger than 32b
    with instance.else_scope():
        instance.data_move(out_gm[gm_offset], out_ub, constant.SID,
                           constant.DEFAULT_NBURST,
                           nbursts, constant.STRIDE_ZERO, constant.STRIDE_ZERO)
        offset = element_num - each_burst_num
        scalar = instance.Scalar(out_ub.dtype)
        with instance.for_range(0, each_burst_num) as time:
            scalar.set_as(out_ub[offset + time])
            out_ub[time].set_as(scalar)
        instance.data_move(out_gm[gm_offset + offset], out_ub,
                           constant.SID, constant.DEFAULT_NBURST,
                           constant.DEFAULT_BURST_LEN,
                           constant.STRIDE_ZERO, constant.STRIDE_ZERO)


def get_block_element(datatype):
    """
    get the count of element that one block has.
    :param datatype: data type
    :return: count
    """
    data_type_size = get_data_size(datatype)
    return constant.BLOCK_SIZE // data_type_size


def get_attr(attr_value, attr_name, dtype_compute, ir_dtype):
    """
    get the attr

    Parameters
    ----------
    attr_value: value of attr
    attr_name: name of attr
    dtype_compute: is the dtype used for calculation
    ir_dtype: the type of attr is ir

    Returns
    -------
    attr_var
    """
    if attr_value is None:
        attr_dtype = {"src_dtype": ir_dtype}
        attr_var = tbe.var_attr(attr_name, dtype=dtype_compute, addition=attr_dtype)
    else:
        attr_var = tvm.const(attr_value, dtype_compute)
    return attr_var


def get_vlrelu(x, attr_value, attr_name, attr_dtype):
    """
    get vlrelu

    Parameters
    ----------
    x: x tensor
    attr: value of attr
    attr_name: name of attr
    attr_dtype: dtype of attr

    Returns
    -------
    res_vlrelu, attr_value
    """
    if attr_value is None:
        dtype = x.dtype
        scalar = tvm.const(0, dtype)
        tmp_max_x = tbe.vmaxs(x, scalar)
        tmp_min_x = tbe.vmins(x, scalar)
        attr_value = get_attr(attr_value, attr_name, dtype, attr_dtype)
        tmp_mul_x = tbe.vmuls(tmp_min_x, attr_value)
        res_vlrelu = tbe.vadd(tmp_max_x, tmp_mul_x)
    else:
        res_vlrelu = tbe.vlrelu(x, attr_value)
    return res_vlrelu, attr_value
