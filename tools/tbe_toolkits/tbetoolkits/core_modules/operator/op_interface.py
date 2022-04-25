#!/usr/bin/env python3
"""
Operator Compilation Interface
"""
# Standard Packages
import json
import time
import copy
import ctypes
import inspect
import logging
from types import ModuleType
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Sequence
from collections import Callable

# Third-Party Packages
from .monkey_patches import build_cfg_monkey_patch
from .monkey_patches import dynamic_build_monkey_patch
from .monkey_patches import rl_bank_monkey_patch
from .monkey_patches import op_pattern_monkey_patch
from .monkey_patches import ir_pass_monkey_patch
from .monkey_patches import soc_spec_monkey_patch
from .special_operator_rules import special_operator_registry
from ...utilities import get
from ...utilities import parse_dtype
from ...utilities import tuple_flatten
from ...utilities import apply_as_list
from ...utilities import get_loaded_so_path
from ...utilities import param_transformation
from ...utilities import get_global_storage
from ...utilities import eliminate_scalar_shapes
from ...utilities import DynamicCompilationResult
from ...utilities import BinaryCompilationResult
from ..testcase_manager import UniversalTestcaseStructure


class OperatorInterface:
    """
    Class Interface for Operator Definition and its Compilation
    """
    special_operator_registry = special_operator_registry

    def __init__(self, ):
        # Load top te tbe modules
        self.te = __import__("te")
        self.tbe = __import__("tbe")
        logging.debug("Using te module from %s" % self.te.__file__)
        logging.debug("Using tbe module from %s" % self.tbe.__file__)
        # noinspection PyProtectedMember
        logging.debug("Using libtvm.so from %s" % self.te._ffi.base._LIB._name)
        ctypes.CDLL("liboptiling.so")
        # noinspection PyBroadException
        try:
            logging.debug("Using liboptiling.so from %s" % get_loaded_so_path(ctypes.CDLL("liboptiling.so")))
        except:
            logging.debug("Get liboptiling.so path failed, apt install lsof or yum install lsof may solve this.")
        # Special
        self.cube_random_buff = __import__("te.platform.cube_random_buff",
                                           fromlist=["cube_random_buff"]).cube_random_buff
        self.vector_random_buff = __import__("te.platform.vector_random_buff",
                                             fromlist=["vector_random_buff"]).vector_random_buff
        self.api_config = __import__("tbe.tvm._api_config",
                                     fromlist=["_api_config"]).api_config
        # Set platform
        logging.debug(f"Setting te version to "
                      f"{get_global_storage().device_platform} for {get_global_storage().core_type}")
        self.te.platform.cce_conf.te_set_version(get_global_storage().device_platform, get_global_storage().core_type)

    @staticmethod
    def get_operator(module: str, op_name: str) -> Optional[Callable]:
        """
        Get operator function
        :param module: operator module name (usually dynamic or impl)
        :param op_name: operator name
        :return: operator function
        """
        # Try to get operator module and operator function in it
        search_list = (f"{module}.{op_name}",
                       f"tbetoolkits.user_defined_modules.fusion_op.{module}.{op_name}",
                       f"tbetoolkits.user_defined_modules.dsl_op.{module}.{op_name}")
        as_module = None
        for s in search_list:
            try:
                parent_module: ModuleType = __import__(s, fromlist=[op_name])
                logging.debug("Trying to get operator %s from %s" % (op_name, parent_module.__file__))
                as_module = getattr(parent_module, op_name)
                break
            except (AttributeError, ModuleNotFoundError):
                continue
            except Exception as e:
                logging.exception("Operator import sequence for %s encountered a fatal error: %s", op_name, e)

        if not isinstance(as_module, Callable):
            result = getattr(as_module, op_name, None)
        else:
            result = as_module
        return result

    @staticmethod
    def prepare_operator_parameters(input_tensor_shapes,
                                    input_tensor_ori_shapes,
                                    input_tensor_dtypes,
                                    input_tensor_formats,
                                    input_tensor_ori_formats,
                                    input_tensor_ranges,
                                    output_tensor_shapes,
                                    output_tensor_ori_shapes,
                                    output_tensor_dtypes,
                                    output_tensor_formats,
                                    output_tensor_ori_formats,
                                    output_tensor_ranges) -> Tuple[Optional[Dict[str, Any]]]:
        """
        This method is intended to construct operator dict inputs
        :return:
        """
        # Construct inputs
        tensors = []
        for ip_n, shape in enumerate(input_tensor_shapes):
            if shape is not None:
                temp_tensor_dict = {"shape": shape,
                                    "ori_shape": get(input_tensor_ori_shapes, ip_n),
                                    "range": get(input_tensor_ranges, ip_n),
                                    "dtype": parse_dtype(get(input_tensor_dtypes, ip_n)),
                                    "format": get(input_tensor_formats, ip_n),
                                    "ori_format": get(input_tensor_ori_formats, ip_n)}
            else:
                temp_tensor_dict = None
            tensors.append(temp_tensor_dict)
        for op_n, shape in enumerate(output_tensor_shapes):
            if shape is not None:
                temp_tensor_dict = {"shape": shape,
                                    "ori_shape": get(output_tensor_ori_shapes, op_n),
                                    "range": get(output_tensor_ranges, op_n),
                                    "dtype": parse_dtype(get(output_tensor_dtypes, op_n)),
                                    "format": get(output_tensor_formats, op_n),
                                    "ori_format": get(output_tensor_ori_formats, op_n)}
            else:
                temp_tensor_dict = None
            tensors.append(temp_tensor_dict)
        return tuple(tensors)

    @staticmethod
    def prepare_operator_parameters_const(testcase: UniversalTestcaseStructure):
        """
        This method is intended to construct operator dict inputs for const
        :return:
        """
        from ...user_defined_modules.op_tiling_transformation import op_tiling_trans_registry
        dyn_input_tensor_shapes = eliminate_scalar_shapes(testcase.dyn_inputs)
        dyn_input_tensor_dtypes = testcase.dyn_input_dtypes
        dyn_input_tensor_formats = testcase.dyn_input_formats
        dyn_input_tensor_ori_formats = testcase.dyn_input_ori_formats
        dyn_input_tensor_ranges = testcase.dyn_input_ranges
        stc_input_tensor_shapes = eliminate_scalar_shapes(testcase.stc_inputs)
        stc_input_tensor_ori_shapes = eliminate_scalar_shapes(testcase.stc_ori_inputs)
        stc_input_tensor_dtypes = testcase.stc_input_dtypes
        stc_input_tensor_formats = testcase.stc_input_formats
        stc_input_tensor_ori_formats = testcase.stc_input_ori_formats
        stc_output_tensor_shapes = testcase.stc_outputs
        stc_output_tensor_ori_shapes = eliminate_scalar_shapes(testcase.stc_ori_outputs)
        stc_output_tensor_dtypes = testcase.output_dtypes
        stc_output_tensor_formats = testcase.output_formats
        stc_output_tensor_ori_formats = testcase.output_ori_formats
        stc_output_tensor_ranges = testcase.dyn_output_ranges
        # Op Tiling Transformation
        possible_const_input_dict = testcase.other_runtime_params
        if testcase.op_name in op_tiling_trans_registry.op_tiling_trans_map:
            trans_func = op_tiling_trans_registry.op_tiling_trans_map[testcase.op_name]
            static_input_shapes, possible_const_input_dict = trans_func(stc_input_tensor_shapes,
                                                                        possible_const_input_dict)
        # Construct inputs
        tensors = []
        skipped_index_count = 0
        for ip_n, _ in enumerate(dyn_input_tensor_shapes):
            if ip_n not in testcase.const_input_indexes:
                stc_index = ip_n - skipped_index_count
                if stc_input_tensor_shapes[stc_index] is not None:
                    temp_tensor_dict = {"shape": stc_input_tensor_shapes[stc_index],
                                        "ori_shape": get(stc_input_tensor_ori_shapes, stc_index),
                                        "range": get(dyn_input_tensor_ranges, ip_n),
                                        "dtype": parse_dtype(get(stc_input_tensor_dtypes, stc_index)),
                                        "format": get(stc_input_tensor_formats, stc_index),
                                        "ori_format": get(stc_input_tensor_ori_formats, stc_index)}
                else:
                    temp_tensor_dict = None
            else:
                skipped_index_count += 1
                dyn_func_params = tuple(inspect.signature(OperatorInterface.get_operator("impl.dynamic",
                                                                                         testcase.op_name)).parameters)
                key = get(dyn_func_params, ip_n)
                if key in possible_const_input_dict:
                    value = possible_const_input_dict[key]
                else:
                    raise KeyError("Unable to find const input %s" % key)
                if not isinstance(value, Sequence):
                    value = (value,)
                my_dtype = parse_dtype(get(dyn_input_tensor_dtypes, ip_n))
                my_value = tuple_flatten(value)
                temp_tensor_dict = {"shape": (len(value),),
                                    "ori_shape": (len(value),),
                                    "range": get(dyn_input_tensor_ranges, ip_n),
                                    "dtype": my_dtype,
                                    "format": get(dyn_input_tensor_formats, ip_n),
                                    "ori_format": get(dyn_input_tensor_ori_formats, ip_n),
                                    "value": my_value,
                                    "const_value": my_value}
            tensors.append(temp_tensor_dict)
        for op_n, shape in enumerate(stc_output_tensor_shapes):
            if shape is not None:
                temp_tensor_dict = {"shape": shape,
                                    "ori_shape": get(stc_output_tensor_ori_shapes, op_n),
                                    "range": get(stc_output_tensor_ranges, op_n),
                                    "dtype": parse_dtype(get(stc_output_tensor_dtypes, op_n)),
                                    "format": get(stc_output_tensor_formats, op_n),
                                    "ori_format": get(stc_output_tensor_ori_formats, op_n)}
            else:
                temp_tensor_dict = None
            tensors.append(temp_tensor_dict)
        return tuple(tensors)

    def compile_dynamic_shape(self, dyn_params: tuple, testcase: UniversalTestcaseStructure,
                              use_static_context: bool = False,
                              mode: str = "Dynamic") -> Union[None, Tuple[str,
                                                              dict,
                                                              str,
                                                              Tuple[str],
                                                              str,
                                                              str,
                                                              str]]:
        """
        Dynamic shape operator compilation
        """
        # Get Operator
        operator_func = self.get_operator("impl.dynamic", testcase.op_name)
        if operator_func is None:
            return None
        # Replace build config
        g_build_cfg = build_cfg_monkey_patch(testcase.manual_dyn_build_config)
        # Turn off parallel fatbin
        g_parallel_fatbin = ir_pass_monkey_patch(self.te.tvm.ir_pass)
        # soc spec injection
        g_soc_spec = None
        if get_global_storage().soc_spec_override:
            g_soc_spec = soc_spec_monkey_patch(get_global_storage().soc_spec_override)
            next(g_soc_spec)
        next(g_build_cfg)
        # Get rl_bank query result
        rl_query_result = ["Unknown"]
        g_rl_bank = rl_bank_monkey_patch(self.tbe.common.rl_bank, rl_query_result)
        next(g_rl_bank)
        if not get_global_storage().parallel_fatbin:
            next(g_parallel_fatbin)
        # Fix kwargs
        dynamic_shape_func_parameters = tuple(inspect.signature(operator_func).parameters)
        op_kwargs = param_transformation(testcase.other_compilation_params, dynamic_shape_func_parameters)
        op_kwargs["kernel_name"] = testcase.dyn_kernel_name
        tensor_list_list = apply_as_list(dyn_params, testcase.dyn_input_as_list_distribution)
        # Call function
        int64_shape_enable = get_global_storage().int64_shape_mode
        with self.api_config.bit_width_64() if int64_shape_enable else self.api_config.bit_width_32():
            with self.tbe.common.context.op_context.OpContext("dynamic" if not use_static_context else "static") as cxt:
                op_info_registered = False
                for line in inspect.getsource(operator_func).split("\n"):
                    if "register_operator" in line:
                        op_type = eval(line[line.index("register_operator") + 18:-1].split(",")[0])
                        op_info = self.tbe.common.context.op_info.OpInfo(op_type, op_type)
                        cxt.add_op_info(op_info)
                        if use_static_context:
                            op_info.inputs = dyn_params[:len(testcase.dyn_inputs)]
                            op_info.outputs = dyn_params[len(testcase.dyn_inputs):]
                        op_info_registered = True
                        break
                if not op_info_registered:
                    logging.warning("OpInfo not registered, add unknown opinfo")
                    op_type = "UNKNOWN"
                    op_info = self.tbe.common.context.op_info.OpInfo(op_type, op_type)
                    cxt.add_op_info(op_info)
                    if use_static_context:
                        op_info.inputs = dyn_params[:len(testcase.dyn_inputs)]
                        op_info.outputs = dyn_params[len(testcase.dyn_inputs):]
                if testcase.random_buff:
                    cxt.add_addition("compile_reset_op", "clear_cube")
                    self.cube_random_buff()
                    cxt.add_addition("compile_reset_op", "clear_vector")
                    self.vector_random_buff()
                    cxt.add_addition("compile_reset_op", "")
                    self.te.platform.cce_build_module.check_reset_op()
                try:
                    sch_count = [0]
                    g_dyn_build = dynamic_build_monkey_patch(self.tbe.dsl, sch_count)
                    next(g_dyn_build)
                    before_compile = time.time()
                    if testcase.dyn_input_as_list_distribution:
                        operator_func(*copy.deepcopy(tensor_list_list),
                                      **copy.deepcopy(op_kwargs))
                    else:
                        operator_func(*copy.deepcopy(dyn_params),
                                      **copy.deepcopy(op_kwargs))
                    after_compile = time.time()
                except:
                    if testcase.dyn_input_as_list_distribution:
                        param_print = self.print_func_params(dynamic_shape_func_parameters, op_kwargs, tensor_list_list)
                    else:
                        param_print = self.print_func_params(dynamic_shape_func_parameters, op_kwargs, dyn_params)
                    logging.error(
                        "%s shape operator func call failure\n" % mode +
                        ("Operator: %s\n" % testcase.op_name) +
                        ("\n".join(param_print)))
                    raise
                finally:
                    next(g_build_cfg)
                    next(g_dyn_build)
                    next(g_rl_bank)
                    if not get_global_storage().parallel_fatbin:
                        next(g_parallel_fatbin)
                    if get_global_storage().soc_spec_override:
                        next(g_soc_spec)
                compute_contexts = self.tbe.dsl.base.operation.get_context().get_computes()
                if compute_contexts:
                    op_pattern = str(set(tuple_flatten(tuple(cc.get_pattern() for cc in compute_contexts))))
                else:
                    op_pattern = "UNKNOWN"
                compile_info = self.tbe.dsl.base.operation.get_compile_info()
                tiling_op_type = self.tbe.dsl.base.operation.get_context().get_op_type()
                logging.debug("Received op_type from operator context: %s" % tiling_op_type)
        return (str(tiling_op_type),
                compile_info,
                str(after_compile - before_compile),
                tuple(dynamic_shape_func_parameters),
                str(sch_count[0]),
                op_pattern,
                rl_query_result[0])

    def compile_static_shape(self,
                             op_params: Tuple[Optional[Dict[str, Any]]],
                             testcase: UniversalTestcaseStructure) -> Union[None, Tuple[float, str, str]]:
        """
        Static shape operator compilation
        """
        # Get operator function from static impl
        logging.debug("Importing operator %s from impl" % testcase.stc_op_name)
        stc_operator_func = self.get_operator("impl", testcase.stc_op_name)
        if stc_operator_func is None:
            stc_operator_func = self.get_operator("impl", testcase.stc_op_name + "_d")
        if stc_operator_func is None:
            return None
        build_config = testcase.manual_stc_build_config
        kernel_name = testcase.stc_kernel_name
        op_kwargs = {**testcase.other_compilation_params, **testcase.other_runtime_params,
                     "kernel_name": kernel_name}
        as_list = testcase.stc_input_as_list_distribution
        # soc spec injection
        g_soc_spec = None
        if get_global_storage().soc_spec_override:
            g_soc_spec = soc_spec_monkey_patch(get_global_storage().soc_spec_override)
            next(g_soc_spec)
        # Update build config
        g_build_cfg = build_cfg_monkey_patch(build_config)
        next(g_build_cfg)
        # Get rl_bank query result
        rl_query_result = ["Unknown"]
        g_rl_bank = rl_bank_monkey_patch(self.tbe.common.rl_bank, rl_query_result)
        next(g_rl_bank)
        # Get static shape operator pattern
        op_pattern = ["Unknown"]
        g_op_pattern = op_pattern_monkey_patch(self.tbe.dsl.static_schedule.cce_schedule, op_pattern)
        next(g_op_pattern)
        # Special logic for operators like add_n
        tensor_list_list = apply_as_list(op_params, as_list)
        # Static shape func parameter fixes
        static_shape_func_parameters = tuple(inspect.signature(stc_operator_func).parameters)
        op_kwargs = param_transformation(op_kwargs, static_shape_func_parameters)
        for param in tuple(op_kwargs.keys()):
            param_index = static_shape_func_parameters.index(param)
            if (as_list and param_index < len(tensor_list_list)) or (not as_list and param_index < len(op_params)):
                del op_kwargs[param]
        try:
            before_compile = time.time()
            with self.tbe.common.context.op_context.OpContext("pre-static") as cxt:
                if testcase.random_buff:
                    self.tbe.common.context.op_context.get_context().add_addition("compile_reset_op", "clear_cube")
                    self.cube_random_buff()
                    self.tbe.common.context.op_context.get_context().add_addition("compile_reset_op", "clear_vector")
                    self.vector_random_buff()
                    self.tbe.common.context.op_context.get_context().add_addition("compile_reset_op", "")
                    self.te.platform.cce_build_module.check_reset_op()
                if testcase.manual_tiling_op_type:
                    op_type = testcase.manual_tiling_op_type
                    op_info = self.tbe.common.context.op_info.OpInfo(op_type, op_type)
                    cxt.add_op_info(op_info)
                if as_list:
                    stc_operator_func(*copy.deepcopy(tensor_list_list),
                                      **copy.deepcopy(op_kwargs))
                else:
                    stc_operator_func(*copy.deepcopy(op_params),
                                      **copy.deepcopy(op_kwargs))
            after_compile = time.time()
        except:
            if as_list:
                param_print = self.print_func_params(static_shape_func_parameters, op_kwargs, tensor_list_list)
            else:
                param_print = self.print_func_params(static_shape_func_parameters, op_kwargs, op_params)
            logging.error(
                ("%s shape operator func call failure\n" % "Static") +
                ("Operator: %s\n" % testcase.stc_op_name) +
                "\n".join(param_print))
            raise
        finally:
            next(g_rl_bank)
            next(g_build_cfg)
            next(g_op_pattern)
            if get_global_storage().soc_spec_override:
                next(g_soc_spec)
        compile_time = after_compile - before_compile
        return compile_time, rl_query_result[0], op_pattern[0]

    def call_const_op_tiling(self,
                             compile_result: Union[DynamicCompilationResult, BinaryCompilationResult],
                             testcase: UniversalTestcaseStructure) -> dict:
        """
        Dynamic shape op_tiling
        """
        from ...user_defined_modules.op_tiling_transformation import op_tiling_trans_registry
        inputs = []
        outputs = []
        static_input_shapes = eliminate_scalar_shapes(testcase.stc_inputs)
        static_input_ori_shapes = eliminate_scalar_shapes(testcase.stc_ori_inputs)
        static_output_shapes = eliminate_scalar_shapes(testcase.stc_outputs)
        static_output_ori_shapes = eliminate_scalar_shapes(testcase.stc_ori_outputs)
        dyn_func_params = compile_result.func_params
        # Op Type initialization
        tiling_op_type = testcase.manual_tiling_op_type \
            if testcase.manual_tiling_op_type else compile_result.tiling_op_type
        # Op Tiling Transformation
        possible_const_input_dict = testcase.other_runtime_params
        if tiling_op_type in op_tiling_trans_registry.op_tiling_trans_map:
            trans_func = op_tiling_trans_registry.op_tiling_trans_map[tiling_op_type]
            static_input_shapes, possible_const_input_dict = trans_func(static_input_shapes,
                                                                        possible_const_input_dict)
        possible_const_input_dict = param_transformation(possible_const_input_dict, dyn_func_params)
        attrs = OperatorInterface.construct_optiling_attrs({**testcase.other_compilation_params,
                                                            **testcase.other_runtime_params})
        if len(attrs) == 0:
            attrs = None
            logging.debug(f"Calling optiling with attrs None")
        else:
            logging.debug(f"Calling optiling with attrs: {attrs}")
        # Construct inputs
        skipped_input_index = 0
        for idx, dyn_shape in enumerate(testcase.dyn_inputs):
            if dyn_shape is not None:
                if idx in testcase.const_input_indexes:
                    skipped_input_index += 1
                    const_input_in_const_index = testcase.const_input_indexes.index(idx)
                    tiling_name = get(testcase.const_input_modes, const_input_in_const_index)
                    if tiling_name is None:
                        # Default mode const input will search its same-name arguments in possible_const_input_dict
                        tiling_name = dyn_func_params[idx]
                    dyn_input_const_name = dyn_func_params[idx]
                    # noinspection PyBroadException
                    if dyn_input_const_name not in possible_const_input_dict \
                            and tiling_name not in possible_const_input_dict:
                        raise KeyError(
                            "Could not find const input %s nor %s for op_tiling %s, existing const inputs are: %s"
                            % (dyn_input_const_name, tiling_name, tiling_op_type, str(possible_const_input_dict)))
                    if dyn_input_const_name in possible_const_input_dict:
                        value = possible_const_input_dict[dyn_input_const_name]
                    else:
                        value = possible_const_input_dict[tiling_name]
                    if not isinstance(value, Sequence):
                        value = (value,)
                    if len(value) not in testcase.dyn_inputs[idx] \
                            and -1 not in testcase.dyn_inputs[idx] and -2 not in testcase.dyn_inputs[idx]:
                        logging.warning("Const input %s shape %s not match with const input tensor shape %s for %s"
                                        % (tiling_name, str(len(value)), str(testcase.dyn_inputs[idx]),
                                           tiling_op_type))
                    my_dtype = get(testcase.dyn_input_dtypes, idx)
                    my_value = tuple_flatten(value)
                    inputs.append({"shape": (len(value),),
                                   "dtype": my_dtype,
                                   "format": "ND",
                                   "ori_format": "ND",
                                   "name": tiling_name,
                                   "const_value": my_value})
                    # const inputs
                else:
                    # inputs
                    idx -= skipped_input_index
                    tensor_representation = {"shape": static_input_shapes[idx],
                                             "dtype": parse_dtype(get(testcase.stc_input_dtypes, idx)),
                                             "ori_shape": static_input_ori_shapes[idx],
                                             "format": get(testcase.stc_input_formats, idx),
                                             "ori_format": get(testcase.stc_input_ori_formats, idx)}
                    if None in tensor_representation.values():
                        logging.warning("Detected None in op_tiling tensor representation, op_tiling may fail!")
                    inputs.append(tensor_representation)
        # Construct outputs
        for idx, shape in enumerate(testcase.dyn_outputs):
            if shape is not None:
                outputs.append({"shape": static_output_shapes[idx],
                                "dtype": parse_dtype(get(testcase.output_dtypes, idx)),
                                "ori_shape": static_output_ori_shapes[idx],
                                "format": get(testcase.output_formats, idx),
                                "ori_format": get(testcase.output_ori_formats, idx)})
        # List construction
        fused_input = apply_as_list(inputs + outputs, testcase.dyn_input_as_list_distribution)
        # Convert fused_input back to inputs and outputs
        final_inputs = []
        final_outputs = []
        if fused_input:
            input_tensor_count = 0
            for i in fused_input:
                if input_tensor_count == len(inputs):
                    final_outputs.append(i)
                elif input_tensor_count > len(inputs):
                    raise RuntimeError("dyn_input_as_list_distribution is invalid, exceeded size of dyn_inputs.")
                else:
                    final_inputs.append(i)
                    if isinstance(i, Sequence):
                        input_tensor_count += len(i)
                    else:
                        input_tensor_count += 1
        else:
            final_inputs = inputs
            final_outputs = outputs
        # Call do_op_tiling
        logging.debug("Calling TbeOptiling with arguments: %s" % str((tiling_op_type,
                                                                      json.dumps(compile_result.compile_info),
                                                                      final_inputs,
                                                                      final_outputs)))
        tiling_time = []
        try:
            tiling_result = self.tbe.common.utils.op_tiling.do_op_tiling(tiling_op_type,
                                                                         compile_result.compile_info,
                                                                         final_inputs,
                                                                         final_outputs,
                                                                         timer=tiling_time,
                                                                         attrs=attrs)
        except:
            logging.error(f"{tiling_op_type} OP Tiling C++ Func Call Failed, "
                          f"check ascend log or ascend logging print for details")
            raise RuntimeError(f"{tiling_op_type} Op Tiling Func Call Failure") from None
        if tiling_time:
            for i in range(get_global_storage().tiling_run_time):
                tiling_time_temp = []
                self.tbe.common.utils.op_tiling.do_op_tiling(tiling_op_type,
                                                             compile_result.compile_info,
                                                             final_inputs,
                                                             final_outputs,
                                                             timer=tiling_time_temp,
                                                             attrs=attrs)
                tiling_time += tiling_time_temp
        tiling_result["tiling_time"] = str(tiling_time)
        return tiling_result

    @staticmethod
    def print_func_params(dynamic_shape_func_parameters, op_kwargs, tensor_list_list):
        """

        :param dynamic_shape_func_parameters:
        :param op_kwargs:
        :param tensor_list_list:
        :return:
        """
        op_param_distribution = {}
        param_idx = 0
        for param in dynamic_shape_func_parameters:
            if param in op_kwargs:
                param_idx += 1
                op_param_distribution[param] = op_kwargs[param]
            else:
                if param_idx < len(tensor_list_list):
                    op_param_distribution[param] = tensor_list_list[param_idx]
                else:
                    op_param_distribution[param] = "UNKNOWN"
                param_idx += 1
        param_idx = 0
        for param in dynamic_shape_func_parameters:
            if param_idx < len(tensor_list_list):
                op_param_distribution[param] = tensor_list_list[param_idx]
            param_idx += 1
        param_print = ["***Params***"]
        for param in op_param_distribution:
            param_print.append("%s %s:\n%s" % (param, str(type(op_param_distribution[param])),
                                               str(op_param_distribution[param])))
        param_print.append("***Params***")
        return param_print

    @staticmethod
    def construct_optiling_attrs(attr_dictionary: dict) -> tuple:
        def detect_type_of_sequence(_sequence: Sequence):
            supported_types = (bool, float, int, str)
            _result = None
            if all(isinstance(_element, Sequence) and not isinstance(_element, str) for _element in _sequence):
                _element_types = set()
                for element in _sequence:
                    _element_types.add(detect_type_of_sequence(element))
                if len(_element_types) == 1:
                    _element_type = _element_types.pop()
                    if _element_type is not None:
                        _result = "list_" + _element_type
            else:
                for _type in supported_types:
                    if all(isinstance(_element, _type) for _element in _sequence):
                        _result = _type.__name__
            return _result

        result = []
        for key in attr_dictionary:
            if isinstance(attr_dictionary[key], Sequence) and not isinstance(attr_dictionary[key], str):
                attr_type = detect_type_of_sequence(attr_dictionary[key])
                if attr_type is None:
                    logging.warning(f"DType detection for attr {key} "
                                    f"value {attr_dictionary[key]} "
                                    f"type {type(attr_dictionary[key])}, ignored")
                    continue
                result.append({"name": key, "dtype": "list_" + attr_type, "value": attr_dictionary[key]})
            else:
                attr_type = detect_type_of_sequence((attr_dictionary[key],))
                if attr_type is None:
                    logging.warning(f"DType detection for attr {key} "
                                    f"value {attr_dictionary[key]} "
                                    f"type {type(attr_dictionary[key])}, ignored")
                    continue
                result.append({"name": key, "dtype": attr_type, "value": attr_dictionary[key]})
        return tuple(result)
