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
Compute Manager for fusion
"""
import importlib
import itertools
import json
import inspect

from te.platform import operation
from te.platform import cce_policy
from te.platform.operation import get_operator as dyn_get_operator
from te.tvm.schedule import create_schedule as _create_schedule
from te.tvm._api_config import api_config


# pylint: disable=useless-object-inheritance, too-many-instance-attributes
class FusionManager(object):
    """Manage computes which are registered
    Save and call compute function by their registered key
    """
    def __init__(self):
        """init"""
        self._build_cfg = "enable"
        self._current_op_name = ""
        self._current_op_func_name = ""
        self._current_op_pattern = "Opaque"
        self._op_build_type = {}
        self._op_args = {}
        self._op_kwds = {}
        self._op_compute = {}
        self.fusion_build_config = {}
        self.res_map = {}
        self.tik_tensor_dict = {}
        self.cheque_list = []
        self.res_index = 0

    def clear(self):
        """clear"""
        self.__init__()

    def register(self, register_name):
        """Register compute

        Parameters
        ----------
        register_name : string
            register_name to call compute.

        Returns
        -------
        decorator : decorator
            decorator to register compute.
        """

        def decorator(func):
            """Save op function name for compute name

            Parameters
            ----------
            func : string
                op function name

            Returns
            -------
                op function name.
            """
            self._op_compute[register_name] = func
            return func

        return decorator

    def has_op_compute(self, register_name):
        """Whether compute_manager has this compute

        Parameters
        ----------
        register_name : string
            compute_name to call compute.

        Returns
        -------
        has_op_compute : bool
            Whether compute_manager has this compute.
        """
        return register_name in self._op_compute

    def get_op_compute_func(self, register_name):
        """get op compute function
        Parameters
        ----------
        register_name : string
            compute_name to call compute.
        Returns
        -------
            op compute function
        """
        return self._op_compute.get(register_name)

    def get_op_compute(self, register_name, *args, **kwds):
        """Get op compute

        Parameters
        ----------
        register_name : string
            register_name to call func.
        *args, **kwds
            op_params.

        Returns
        -------
        op_compute : compute
            compute corresponding to the compute_name.
        """
        return self._op_compute[register_name](*args, **kwds)

    def set_current_op_name(self, op_name):
        """Set current op_name

        Parameters
        ----------
        op_name : string
            update current op_name to save op_params.
        """
        self._current_op_name = op_name

    def has_op_params(self, op_name):
        """
        check if op params has saved
        """
        return op_name in self._op_args

    def set_op_params(self, *args, **kwds):
        """Set current op_name's op_params

        Parameters
        ----------
        *args, **kwds
            save current op_name's op_params.
        """
        self._op_args[self._current_op_name] = args
        self._op_kwds[self._current_op_name] = kwds

    def set_op_build_type(self, args_type):
        """Get current op_name's build type, it is singlebuild or prebuild

        Parameters
        ----------
        args_type : string
            singlebuild or prebuild
        """
        self._op_build_type[self._current_op_name] = args_type

    def get_op_build_type(self, register_name):
        """Get current op_name's build type

        Parameters
        ----------
        register_name : string
            key to get build type

        Returns
        -------
        args
            current op_name's build type.
        """
        if register_name in self._op_build_type:
            return self._op_build_type[register_name]
        return ""

    def set_op_res(self, res):
        """Get current op_name's build type, it is singlebuild or prebuild

        Parameters
        ----------
        args_type : string
            singlebuild or prebuild
        """
        if get_build_cfg() == "enable":
            res_op = []
            op_outputs = []
            if isinstance(res, list):
                for single_res in res:
                    res_op.append(single_res.op)
                    op_outputs.append(single_res.op.name)
            else:
                res_op.append(res.op)
                op_outputs.append(res.op.name)
            sch = _create_schedule(res_op)
            sch.cce_special = {"op_outputs": op_outputs}
            self.res_map.setdefault(self._current_op_name, []).append(sch)

    def set_tensor_list(self, tensor_list):
        """save tensor_list

        Parameters
        ----------
        args_type : list
            tensor_list
        """
        if get_build_cfg() == "disable":
            return
        if self._current_op_name not in self.res_map:
            return

        sch = self.res_map[self._current_op_name][-1]
        if "tensor_list" in sch.cce_special:
            return

        sch.cce_special["tensor_list"] = tensor_list

    def set_tik_tensor(self, input_tensor, output):
        """Save tik op input&output tensor

        Parameters
        ----------
        input_tensor : tik input tensor
        output : tik output tensor
        """
        self.tik_tensor_dict[self._current_op_name] = [input_tensor, output]

    def set_cheque_list(self, cheque_list):
        """Save RL cheque_list

        Parameters
        ----------
        cheque_list : RL cheque_list
        """
        self.cheque_list.append(cheque_list)

    def clear_res_index(self):
        """clear RL res_index

        Parameters
        ----------
        """
        self.res_index = 0

    def clear_cheque_list(self):
        """clear RL cheque_list

        Parameters
        ----------
        """
        self.cheque_list = []

    def get_op_res(self, key):
        """Get current op_name's build type

        Parameters
        ----------
        register_name : string
            key to get build type

        Returns
        -------
        args
            current op_name's build type.
        """
        return self.res_map.get(key, None)

    def get_tik_tensor(self, key):
        """Get tik op input&output tensor

        Parameters
        ----------

        Returns
        -------
        args
            none
        """
        return self.tik_tensor_dict.get(key, None)

    def get_cheque_list(self, res_index=None):
        """Get RL cheque_list

        Parameters
        ----------
        res_index:
        Returns
        -------
        args
            none
        """
        if res_index is None:
            return self.cheque_list
        if res_index >= len(self.cheque_list):
            return None
        return self.cheque_list[res_index]

    def get_res_index(self):
        """Get RL res_index

        Parameters
        ----------
        Returns
        -------
        args
            none
        """
        curr_res_index = self.res_index
        self.res_index += 1
        return curr_res_index

    def get_op_args(self, op_name):
        """Get current op_name's op_args

        Parameters
        ----------
        op_name : string
            key to get op_args

        Returns
        -------
        args
            save current op_name's op_args.
        """
        return self._op_args[op_name]

    def get_op_kwds(self, op_name):
        """Get current op_name's op_kwds

        Parameters
        ----------
        op_name : string
            key to get op_kwds

        Returns
        -------
        kwds
            save current op_name's op_kwds.
        """
        return self._op_kwds[op_name]

    def init_current_op_pattern(self):
        """Init current op's pattern"""

        self._current_op_pattern = "Opaque"

    def set_current_op_pattern(self, op_pattern):
        """Set current op's pattern

        Parameters
        ----------
        op_pattern : string
            current single op's pattern.
        """
        self._current_op_pattern = op_pattern

    def get_current_op_pattern(self):
        """Get current single op's pattern

        Returns
        ----------
        op_pattern : string
            current single op's pattern.
        """
        if self.has_op_compute(self.get_current_op_func_name()):
            return self._current_op_pattern
        return "Opaque"

    def set_current_op_func_name(self, op_func_name):
        """Set current op's func name

        Parameters
        ----------
        op_func_name : string
            current single op's func name.
        """
        self._current_op_func_name = op_func_name

    def get_current_op_func_name(self):
        """Get current single op's func name

        Returns
        ----------
        op_func_name : string
            current single op's func name.
        """
        return self._current_op_func_name

    def get_fuse_info(self):
        """Check whether this op will be fused

        Returns
        ----------
        True : bool
            this op will be fused
        False : bool
            this op will not be fused
        """
        if self._current_op_name \
                and self._current_op_func_name \
                and (self._current_op_name in self._op_args
                     or self._current_op_name in self._op_kwds):
            return True
        return False

    def set_build_cfg(self, op_build_cfg):
        """Set current op's build switch

        Parameters
        ----------
        op_pattern : string
            current single op's switch.
        """
        self._build_cfg = op_build_cfg

    def get_build_cfg(self):
        """Get current single op's switch

        Returns
        ----------
        op_pattern : string
            current single op's switch.
        """
        return self._build_cfg


# pylint: disable=invalid-name
# Singleton for managing all registered compute
fusion_manager = FusionManager()


def set_current_op_name(op_name):
    """Set current op_name, external interface for C call python

    Parameters
    ----------
    op_name : string
        update current op_name to save op_params.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_current_op_name(op_name)


def set_current_op_func_name(op_func_name):
    """Set current op's func name, external interface for C call python

    Parameters
    ----------
    op_func_name : string
        current single op's func name.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_current_op_func_name(op_func_name)


def set_op_params(*args, **kwds):
    """Set current name's op_params, external interface for C call python

    Parameters
    ----------
    *args, **kwds
        save current op_name's op_params.

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_op_params(*args, **kwds)


def set_op_build_type(args_type):
    """Get current op_name's build type

    Parameters
    ----------
    args_type : string
        build type need to be set
    """
    fusion_manager.set_op_build_type(args_type)


def get_op_build_type(register_name):
    """Get current op_name's op_args

    Parameters
    ----------
    register_name : string
        key to get build type

    Returns
    -------
    args
        return current op_name's build type.
    """
    return fusion_manager.get_op_build_type(register_name)


def set_op_res(res_val):
    """Get current op_name's build type

    Parameters
    ----------
    args_type : string
        build type need to be set
    """
    fusion_manager.set_op_res(res_val)


def set_tensor_list(tensor_list):
    """save tensor_list

    Parameters
    ----------
    args_type : list
        tensor_list need to save
    """
    fusion_manager.set_tensor_list(tensor_list)


def set_tik_tensor(input_tensor, output):
    """save tik_tensor

    Parameters
    ----------
    input_tensor : tik input tensor
    output : tik output tensor
    """
    fusion_manager.set_tik_tensor(input_tensor, output)


def set_cheque_list(cheque_list):
    """Save RL cheque_list

    Parameters
    ----------
    cheque_list : RL cheque_list
    """
    fusion_manager.set_cheque_list(cheque_list)


def clear_cheque_list():
    """clear RL cheque_list

    Parameters
    ----------
    """
    fusion_manager.clear_cheque_list()


def clear_res_index():
    """clear RL res_index

    Parameters
    ----------
    """
    fusion_manager.clear_res_index()


def get_op_res(key):
    """Get current op_name's op_args

    Parameters
    ----------
    register_name : string
        key to get build type

    Returns
    -------
    args
        return current op_name's build type.
    """
    return fusion_manager.get_op_res(key)


def get_tik_tensor(key):
    """Get tik op input&output tensor

    Parameters
    ----------

    Returns
    -------
    args
        none
    """
    return fusion_manager.get_tik_tensor(key)


def get_cheque_list(res_index=None):
    """Get RL cheque_list

    Parameters
    ----------
    res_index:
    Returns
    -------
    args
        none
    """
    return fusion_manager.get_cheque_list(res_index)


def get_res_index():
    """Get RL res_index

    Parameters
    ----------
    Returns
    -------
    args
        none
    """
    return fusion_manager.get_res_index()


def op_build_cfg_en():
    """Set current name's op_params, enable  build .o

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_build_cfg("enable")


def op_build_cfg_dis():
    """Set current name's op_params, disable build .o

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    fusion_manager.set_build_cfg("disable")


def get_build_cfg():
    """Get current name's op_params, build .o or not

    Parameters
    ----------

    Returns
    -------
    succ_flag : boolean
        end of execution
    """
    return fusion_manager.get_build_cfg()


def init_op_pattern():
    """init current name's pattern

    Parameters
    ----------

    Returns
    -------
    op pattern value
        end of execution
    """
    fusion_manager.init_current_op_pattern()


def get_op_pattern():
    """Get current name's pattern

    Parameters
    ----------

    Returns
    -------
    op pattern value
        end of execution
    """
    return fusion_manager.get_current_op_pattern()


def call_op_func(op_module, op_func_name, op_args):
    """invoke function of op
    """
    opm = importlib.import_module(op_module)
    opfunc = getattr(opm, op_func_name)
    inputs, outputs, attrs = op_args
    return opfunc(*inputs, *outputs, *attrs)


def get_dyn_op(op_module, op_type, inputs=None, outputs=None,
               unknown_shape=False):
    """get dynamic version of op
    """
    if dyn_get_operator is None:
        return None

    if not unknown_shape:
        if inputs is None:
            return None

        all_args = []
        for ele in itertools.chain(inputs, outputs):
            if isinstance(ele, (list, tuple)):
                all_args.extend(ele)
            else:
                all_args.append(ele)

        dyn_flag = False
        for item in all_args:
            if not isinstance(item, dict):
                continue
            shape = item.get('shape')
            if shape is None:
                continue
            if [ele for ele in shape if ele == -1]:
                dyn_flag = True
                break

        if dyn_flag is False:
            return None

    dyn_op_module = op_module.split('.')
    dyn_op_module[-1] = 'dynamic'
    dyn_op_module = '.'.join(dyn_op_module)
    importlib.import_module(dyn_op_module)
    return dyn_get_operator(op_type)


def range_padding(inputs, outputs):
    """
    pad range paramters if range and shape not match
    """
    inouts = []
    for ele in itertools.chain(inputs, outputs):
        if isinstance(ele, (list, tuple)):
            inouts.extend(ele)
        else:
            inouts.append(ele)

    for item in inouts:
        if not isinstance(item, dict):
            continue
        shape = item.get('shape', [])
        shape_range = item.get('range', [])
        if len(shape) != len(shape_range):
            tmp_range = []
            for dim in shape:
                dim1 = dim if dim > 0 else 1
                dim2 = dim if dim > 0 else None
                tmp_range.append([dim1, dim2])
            item['range'] = tmp_range


def check_op_impl_mode(op_module, op_func_name, op_type,
                       inputs, outputs, unknown_shape):
    """
    check if op has impl_mode paramter
    """
    dyn_opfunc = get_dyn_op(op_module, op_type, inputs, outputs, unknown_shape)
    if dyn_opfunc:
        opfunc = dyn_opfunc
    else:
        opm = importlib.import_module(op_module)
        opfunc = getattr(opm, op_func_name)

    impl_mode_arg = inspect.signature(opfunc).parameters.get('impl_mode', None)

    if impl_mode_arg is not None and \
       impl_mode_arg.kind in (inspect.Parameter.KEYWORD_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return True

    return False


def build_single_op_from_c(op_module, op_func_name, op_type,
                           build_mode, unknown_shape, op_args, int64_mode):
    """
    build single op from tefsuion c side
    """
    inputs, outputs, attrs = op_args
    return build_single_op(op_module, op_func_name, op_type, build_mode,
                           inputs=inputs, outputs=outputs, attrs=attrs,
                           unknown_shape=unknown_shape, int64_mode=int64_mode)


def build_single_op(op_module, op_func_name, op_type, build_mode, *op_args,
                    inputs=None, outputs=None, attrs=None,
                    unknown_shape=False, int64_mode=False):
    """Prebuild Op

    Parameters
    ----------
    op_module: op module name
    op_args: op args

    Returns
    -------
    op pattern value
        end of execution
    """
    dyn_flag = False
    dyn_opfunc = get_dyn_op(op_module, op_type, inputs, outputs, unknown_shape)
    if dyn_opfunc:
        opfunc = dyn_opfunc
        dyn_flag = True
    else:
        try:
            opm = importlib.import_module(op_module)
            opfunc = getattr(opm, op_func_name)
        except ImportError:
            dict_args = dict()
            dict_args["errCode"] = "E40008"
            dict_args["module_path"] = op_module
            raise RuntimeError(dict_args)

    if build_mode == 'prebuild':
        set_current_op_func_name(op_func_name)  # for pattern
        init_op_pattern()                       # for init pattern to Opaque
        op_build_cfg_dis()                      # for cce build
    else:
        op_build_cfg_en()

    kwargs = cce_policy.OpImplPolicy.get_op_impl_mode(opfunc, op_type)

    def call_op():
        _compile_info = None
        if inputs is not None:
            if dyn_flag:
                range_padding(inputs, outputs)
                with operation.dynamic():
                    opfunc(*inputs, *outputs, *attrs, **kwargs)
                    _compile_info = operation.get_compile_info()
            else:
                opfunc(*inputs, *outputs, *attrs, **kwargs)
        else:
            if dyn_flag:
                with operation.dynamic():
                    opfunc(*op_args, **kwargs)
                    _compile_info = operation.get_compile_info()
            else:
                opfunc(*op_args, **kwargs)
        return _compile_info

    if int64_mode:
        with api_config.bit_width_64():
            compile_info = call_op()
    else:
        with api_config.bit_width_32():
            compile_info = call_op()

    if build_mode == 'prebuild':
        op_build_cfg_en()
        pattern = get_op_pattern()
        return pattern

    if dyn_flag:
        return json.dumps(compile_info)

    return ""


def save_op_params(op_name, op_build_type, op_args):
    """Save op params

    Parameters
    ----------
    op_name: op name
    op_func_name: op function name
    op_args: op args

    Returns
    -------
    """
    set_current_op_name(op_name)
    set_op_build_type(op_build_type)  # for fusion
    if isinstance(op_args, (list, tuple)):
        if len(op_args) == 2:
            outputs, attrs = op_args
            set_op_params(*outputs, *attrs)
        else:
            set_op_params(*op_args)
    else:
        set_op_params(op_args)


def op_params_to_json(op_name):
    """
    transform op params to json
    """
    args_json = {'list_args': [], 'kwds_args': {}}
    try:
        args_json['list_args'].extend(fusion_manager.get_op_args(op_name))
        args_json['kwds_args'] = fusion_manager.get_op_kwds(op_name)
    except Exception:       # 'pylint: disable=bare-except,broad-except
        pass
    return json.dumps(args_json)


def get_fusion_build_cfg():
    """get build_config used by fusion manager

    Returns
    -------
    fusion_manger build_config:
    """
    return fusion_manager.fusion_build_config


def reset_fusion_build_cfg():
    """reset build_config used by fusion manager
    """
    fusion_manager.fusion_build_config = {}


def clear_fusion_params():
    """
    clear fusion op params
    """
    fusion_manager.clear()
