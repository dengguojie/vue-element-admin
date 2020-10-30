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
Get the tiling
"""
import copy
import numpy as np
import tvm
import json
from te.domain.tiling.tiling_helper import TILING_INSTANCE
from te.domain.tiling.op_param_encode.params_encoder import ParamsEncoder
from te.domain.tiling.auto_tiling_log import AUTOTILINGLOG


# global variable
BANK_CACHE = ""

# define the max value
MAX_UINT32 = 4294967295
MAX_UINT16 = 65535
MAX_UINT8 = 255
MAX_UINT4 = 15
MAX_BOOL = 1

# define the tiling_type
AUTO_TILING_TYPE = 0
CCE_TILING_TYPE = 1
REPOSITORY_TILING_TYPE = 2
PRIORITY_TILING_TYPE = 3
CUSTOM_TILING_TYPE = 4
COST_TILING_TYPE = 5
TUNING_TILING_TYPE = 6

# define the support tiling type
SUPPORT_TILING_TYPE = {
    "auto_tiling": AUTO_TILING_TYPE,
    "cce_tiling": CCE_TILING_TYPE,
    "repository_tiling": REPOSITORY_TILING_TYPE,
    "custom_tiling": CUSTOM_TILING_TYPE,
    "priority_tiling": PRIORITY_TILING_TYPE,
    "tuning_tiling": TUNING_TILING_TYPE,
    "cost_model_tiling": COST_TILING_TYPE
}


def bank_cache_get(info_dict):
    """Get the tiling from bank_cache

    Parameters
    ----------

    Returns
    -------
    ret : bool
    tiling : dict
        The result.
    """
    bank_cache_dict = {}
    if BANK_CACHE:
        try:
            bank_cache_dict = json.loads(BANK_CACHE)
        except json.decoder.JSONDecodeError:
            AUTOTILINGLOG.warn(
                "GA BANK_CACHE: {} must be json format!".format(BANK_CACHE))
    else:
        return False, {}
    # bank_cache kernel_name like
    # te_fused_op_conv2d_51f2e282e2efb9f2_bab2c8511d0e0ea_56f30_3_0051498
    # get te_fused_op_conv2d_51f2e282e2efb9f2_bab2c8511d0e0ea to compare
    b_kernel_name = bank_cache_dict.get("kernel_name", "").rsplit('_', 3)[0]
    info_kernel_name = info_dict["kernel_name"]
    if isinstance(info_dict["kernel_name"], tvm.expr.StringImm):
            info_kernel_name = info_dict["kernel_name"].value

    # kernel_name like
    # te_fused_op_conv2d_51f2e282e2efb9f2_bab2c8511d0e0ea_56f30
    # get te_fused_op_conv2d_51f2e282e2efb9f2_bab2c8511d0e0ea to compare
    i_kernel_name = info_kernel_name.rsplit('_', 1)[0]
    if b_kernel_name == i_kernel_name and bank_cache_dict.get("tuning_mode", "") == "GA":
        tiling = bank_cache_dict["tiling_dict"]
        return True, tiling
    else:
        return False, {}


def get_tiling(info_dict):
    """Get the tiling from module

    Parameters
    ----------
    info_dict: dict
        the params of operator

    Returns
    -------
    tiling : dict
        The result.
    """
    # check the params
    check_params(info_dict)
    op_type = copy.deepcopy(info_dict.get("op_type"))
    tiling_type = info_dict.get("tiling_type")
    dynamicShapeFlag = info_dict.get("dynamic_shape_flag")
    if tiling_type not in [None, "cost_model_tiling"]:
        raise ValueError("the tiling_type is error, only support %s, but tiling_stype is %s" % \
                (str(["None", "cost_model_tiling"]), str(tiling_type)))
    # encode the params of operator
    editor = ParamsEncoder(op_type)
    input_str = editor.encode_array(info_dict)
    # Read the config of tiling type, get the type of tiling
    if tiling_type == "cost_model_tiling":
        mode_old = TILING_INSTANCE.get_tiling_type()
        TILING_INSTANCE.set_tiling_type("cost_model_tiling")
    mode = TILING_INSTANCE.get_tiling_type()
    # default mode is auto_tiling
    if mode is None:
        mode = "auto_tiling"

    # check mode
    if mode not in SUPPORT_TILING_TYPE.keys():
        raise ValueError("the tiling_type is error, only support %s, but mode is %s" % \
            (str(SUPPORT_TILING_TYPE.keys()), str(mode)))
    if dynamicShapeFlag and (mode == "cce_tiling" or mode == "priority_tiling"):
        raise ValueError("the tiling_type is error, dyanmic shape not support %s mode" % (str(mode)))

    tiling_type_num = SUPPORT_TILING_TYPE.get(mode)
    ret, tiling = bank_cache_get(info_dict)
    if ret:
        return tiling

    if tiling_type_num != TUNING_TILING_TYPE:
        tiling_result = tvm.get_global_func("_get_tiling")
        ret = tiling_result(input_str, tiling_type_num)
        tiling = editor.decode(ret)
        if tiling_type == "cost_model_tiling":
            TILING_INSTANCE.set_tiling_type(mode_old)
            mode = TILING_INSTANCE.get_tiling_type()
        TILING_INSTANCE.set_params(info_dict)
    else:
        tiling = TILING_INSTANCE.get_tiling(info_dict)
    AUTOTILINGLOG.debug("[auto_tiling]info_dict:{}".format(info_dict))
    AUTOTILINGLOG.debug("[auto_tiling]tiling:{}, tiling_type:{}".format(tiling, mode))
    return tiling

def check_params(info_dict):
    """check the tiling from module

    Parameters
    ----------
    info_dict: dict
        the params of operator

    Returns
    -------

    """
    # check the info_dict
    if not isinstance(info_dict, dict):
        raise TypeError("info_dict should be dict, but the input is %s" % type(info_dict))

    # check the op_type info
    if "op_type" not in info_dict.keys():
        raise KeyError("the keyword 'op_type' is missing in input params")
