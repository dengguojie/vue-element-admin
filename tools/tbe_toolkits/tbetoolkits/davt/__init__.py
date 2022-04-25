"""
davt test infrastructure aims to provide a clean and neat way of single operator test
"""
import numpy

from enum import Enum
from enum import auto
from typing import Union
from typing import Optional
from typing import Sequence

from tbetoolkits.core_modules import runtime
from tbetoolkits import utilities


class Tensor:
    """
    Representation of Tensor object, can be used as dynamic or static tensor
    """

    def __init__(self,
                 unknownshape=(-1, -1, -1, -1, -1, -1, -1, -1),
                 _range=None,
                 ori_shape=(-1, -1, -1, -1, -1, -1, -1, -1),
                 dtype="float16",
                 _format="ND",
                 ori_format="ND",
                 is_out=False,
                 array=None,
                 real_ori_shape=None):
        self.unknownshape = unknownshape
        self.range = _range if _range is not None else tuple((1, None) for _ in unknownshape)
        self.ori_shape = ori_shape
        self.dtype = dtype
        self.format = _format
        self.ori_format = ori_format
        self.is_out = is_out
        # Real tensor
        self.array = array
        self.real_ori_shape = real_ori_shape if real_ori_shape is not None else \
            self.array.shape if self.array is not None else None

    def __call__(self, real_tensor: "Tensor" = None):
        if real_tensor is None:
            return {"shape": self.unknownshape, "ori_shape": self.ori_shape, "range": self.range,
                    "format": self.format, "ori_format": self.ori_format,
                    "dtype": self.dtype}
        else:
            return {"shape": real_tensor.array.shape, "ori_shape": real_tensor.real_ori_shape,
                    "format": self.format, "ori_format": self.ori_format,
                    "dtype": self.dtype}

    def put(self, array, real_ori_shape=None):
        """
        Change static tensor data
        """
        self.array = array
        self.real_ori_shape = real_ori_shape if real_ori_shape is not None else \
            self.array.shape if self.array is not None else None

    def tonumpy(self):
        """
        Get static tensor data
        """
        return self.array


class ListOfTensor:
    """
    Container of Tensors
    """

    def __init__(self, tensors):
        self.container = tensors

    def __call__(self, real_tensors=None):
        if not len(self.container) == len(real_tensors):
            raise RuntimeError("Required %d tensors but received %d" % (len(self.container), len(real_tensors)))
        if real_tensors is None:
            return [tensor() for tensor in self.container]
        else:
            return [tensor(real_tensors[i]) for i, tensor in enumerate(self.container)]


class DynamicConst(Tensor):
    """
    Const Tensor data for dynamic operator
    """

    def __init__(self,
                 name,
                 unknownshape=(-1, -1, -1, -1, -1, -1, -1, -1),
                 _range=None,
                 ori_shape=(-1, -1, -1, -1, -1, -1, -1, -1),
                 dtype="float16",
                 _format="ND",
                 ori_format="ND",
                 is_out=False):
        super().__init__(unknownshape, _range, ori_shape, dtype, _format, ori_format, is_out)
        self.name = name

    def __call__(self, real_tensor=None):
        if real_tensor is None:
            return super().__call__(real_tensor)
        else:
            super_result = super().__call__(real_tensor)
            super_result.update({"name": self.name, "const_value": tuple(map(int, real_tensor.array))})
            return super_result


class Attr:
    """
    Operator Attribute
    """

    def __init__(self,
                 name,
                 value=None):
        self.name = name
        self.value = value

    def __call__(self):
        return self.value


class RunMode(Enum):
    """
    Operator execution mode enumeration
    """
    LOCAL_ONBOARD_MODE = auto()
    LOCAL_PVMODEL = auto()
    LOCAL_CAMODEL = auto()
    LOCAL_PEMMODEL = auto()


class CompiledDynamicOp:
    """
    Compiled Dynamic Operator
    """

    def __init__(self,
                 op_func,
                 op_params: tuple,
                 op_tiling="AutoTiling",
                 workspace_byte=b"\x01",
                 platform="Ascend910A",
                 core_type="AiCore"):
        # Run properties
        self.run_mode = RunMode.LOCAL_ONBOARD_MODE
        # Main properties
        self.function = op_func
        self.params = list(op_params)
        self.attrs = [param for param in self.params if isinstance(param, Attr)]
        self.tensors = [param for param in self.params if isinstance(param, (Tensor, ListOfTensor))]
        self.op_tiling = op_tiling
        self.core_type = core_type
        self.workspace_byte = workspace_byte
        self.attrs_map = self.__construct_attrs_map__()
        self.__construct_kernel_name__()
        __import__("te").platform.cce_conf.te_set_version(platform, core_type)
        self.__compile__()

    def __construct_attrs_map__(self):
        attrs_map = {}
        for attr in self.attrs:
            if attr.name not in attrs_map:
                attrs_map[attr.name] = attr
            else:
                raise IndexError("Attr %s should be unique" % attr.name)
        return attrs_map

    def __construct_kernel_name__(self):
        if "kernel_name" not in self.attrs_map:
            self.attrs_map["kernel_name"] = Attr("kernel_name", "davt_" + self.function.__name__)
            self.params.append(self.attrs_map["kernel_name"])
        elif self.attrs_map["kernel_name"].value is None:
            _idx = self.params.index(self.attrs_map["kernel_name"])
            self.params.remove(self.attrs_map["kernel_name"])
            self.params.insert(_idx, Attr("kernel_name", "davt_" + self.function.__name__))
            self.attrs_map["kernel_name"] = self.params[_idx]

    def __compile__(self):
        compile_params = [param() if isinstance(param, (Tensor, Attr, ListOfTensor)) else
                          param for param in self.params]
        self.tbe = __import__("tbe")
        with self.tbe.common.context.op_context.OpContext("dynamic"):
            op_info = self.tbe.common.context.op_info.OpInfo(self.op_tiling, self.op_tiling)
            self.tbe.common.context.op_context.get_context().add_op_info(op_info)
            self.function(*compile_params)
            self.complile_info = self.tbe.dsl.base.operation.get_compile_info()

    def setattr(self,
                name: str,
                value):
        """
        Set Attribute
        """
        if name in self.attrs_map:
            self.attrs_map[name].value = value
        else:
            raise IndexError("Attr %s does not exist" % name)

    def __call__(self,
                 *args: Optional[Union[Tensor, ListOfTensor]],
                 manual_block_dim=None,
                 manual_tiling_data=None,
                 manual_tiling_key=None,
                 manual_workspaces=None):
        # Run check
        for idx, arg in enumerate(args):
            if not isinstance(arg, Tensor):
                if not isinstance(arg, type(self.tensors[idx])):
                    raise TypeError("Invalid type of DynamicOp running parameter: %s, Expected: %s"
                                    % (str(arg), type(self.tensors[idx])))
        if None not in [manual_block_dim, manual_tiling_data, manual_tiling_key, manual_workspaces]:
            if not isinstance(manual_block_dim, int):
                raise RuntimeError("At least give me an integer for block_dim, okay? ):")
            if not isinstance(manual_tiling_data, bytes):
                raise RuntimeError("At least give me a byte-array for tiling_data, okay? ):")
            if not isinstance(manual_tiling_key, (str, int)):
                raise RuntimeError("At least give me an int or str for tiling_key, okay? ):")
            if not isinstance(manual_workspaces, Sequence):
                raise RuntimeError("At least give me a Sequence for workspaces, okay? ):")
            for workspace in manual_workspaces:
                if not isinstance(workspace, bytes):
                    raise RuntimeError("At least give me an byte-array for workspace, okay? ):")
            block_dim = manual_block_dim
            tiling_data = manual_tiling_data
            tiling_key = manual_tiling_key
            workspaces = manual_workspaces
        else:
            block_dim, tiling_data, tiling_key, workspaces = self.__do_op_tiling__(args)
        if self.run_mode == RunMode.LOCAL_ONBOARD_MODE:
            # Execute Kernel
            self.__execute__(args, block_dim, tiling_data, tiling_key, workspaces)
            return workspaces

    def __do_op_tiling__(self, args):
        # Construct op_tiling inputs
        inputs = []
        outputs = []
        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor):
                if self.tensors[idx].is_out:
                    outputs.append(self.tensors[idx](arg))
                else:
                    inputs.append(self.tensors[idx](arg))
        tiling_time = []
        # Do op_tiling
        tiling_result = self.tbe.common.utils.op_tiling.do_op_tiling(self.op_tiling,
                                                                     self.complile_info,
                                                                     inputs,
                                                                     outputs,
                                                                     timer=tiling_time)
        block_dim = tiling_result["block_dim"]
        tiling_data = tiling_result["tiling_data"]
        tiling_key = tiling_result["tiling_key"]
        workspaces = tiling_result["workspaces"]
        return block_dim, tiling_data, tiling_key, workspaces

    def __execute__(self, args, block_dim, tiling_data, tiling_key, workspaces):
        device = runtime.RTSInterface()
        device.set_device(0)
        device.create_context("RT_CTX_NORMAL_MODE")
        registered_binary = device.register_device_binary_kernel("kernel_meta/%s.o" %
                                                                 self.attrs_map["kernel_name"].value, self.core_type)
        try:
            stubfunc_p = device.register_function(registered_binary,
                                                  self.attrs_map["kernel_name"].value + f"_{tiling_key}", 0)
        except RuntimeError:
            stubfunc_p = device.register_function(registered_binary,
                                                  self.attrs_map["kernel_name"].value + "__kernel0", 0)
        tensor_arrays = [device.copy_bin_to_hbm(array.tonumpy().tobytes()) for array in args if array is not None]
        tensor_arrays += [device.copy_bin_to_hbm((self.workspace_byte * workspace)) for workspace in workspaces]
        if not len(tiling_data) == 0:
            tensor_arrays.append(device.copy_bin_to_hbm(tiling_data))
        device.launch_kernel(stubfunc_p, block_dim, tuple(tensor_arrays), len(tensor_arrays), None, None)
        device.synchronize_with_stream(None)
        result_arrays = [device.get_data_from_hbm(array_p, args[idx].tonumpy().nbytes)
                         for idx, array_p in enumerate(tensor_arrays) if idx < len(args)]
        idx_count = 0
        for result_array in args:
            if isinstance(result_array, Tensor):
                original_array = result_array.tonumpy()
                result_array.array = numpy.frombuffer(result_arrays[idx_count],
                                                      original_array.dtype).reshape(original_array.shape)
                idx_count += 1
        device.reset()


class StaticOp:
    """
    Compiled Static Operator
    """

    def __init__(self,
                 op_func,
                 op_params: tuple,
                 workspace_byte=b"\x01",
                 core_type="AiCore"):
        # Main properties
        self.function = op_func
        self.params = list(op_params)
        self.attrs = [param for param in self.params if isinstance(param, Attr)]
        self.tensors = [param for param in self.params if isinstance(param, Tensor)]
        self.workspace_byte = workspace_byte
        self.core_type = core_type
        # Construct attr maps
        self.attrs_map = {}
        for attr in self.attrs:
            if attr.name not in self.attrs_map:
                self.attrs_map[attr.name] = attr
            else:
                raise IndexError("Attr %s should be unique" % attr.name)
        # Check if automatic kernel name is needed
        if "kernel_name" not in self.attrs_map:
            self.attrs_map["kernel_name"] = Attr("kernel_name", "davt_" + self.function.__name__)
            self.params.append(self.attrs_map["kernel_name"])
        elif self.attrs_map["kernel_name"].value is None:
            _idx = self.params.index(self.attrs_map["kernel_name"])
            self.params.remove(self.attrs_map["kernel_name"])
            self.params.insert(_idx, Attr("kernel_name", "davt_" + self.function.__name__))
            self.attrs_map["kernel_name"] = self.params[_idx]
        # Compilation
        compile_params = [param(param) if isinstance(param, Tensor) else
                          param() for param in self.params]
        self.function(*compile_params)
        self.block_dim, self.workspaces = utilities.get_stc_json_op_data("./kernel_meta/"
                                                                         + self.attrs_map["kernel_name"].value)

    def setattr(self,
                name: str,
                value):
        """
        Set Attribute
        """
        if name in self.attrs_map:
            self.attrs_map[name].value = value
        else:
            raise IndexError("Attr %s does not exist" % name)

    def __call__(self, *args: Optional[Tensor]):
        # Run check
        for idx, arg in enumerate(args):
            if not isinstance(arg, Tensor):
                if not isinstance(arg, type(self.tensors[idx])):
                    raise TypeError("Invalid type of DynamicOp running parameter: %s" % str(arg))
        # Construct op_tiling inputs
        inputs = []
        outputs = []
        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor):
                if self.tensors[idx].is_out:
                    outputs.append(self.tensors[idx](arg))
                else:
                    inputs.append(self.tensors[idx](arg))
        # Execute Kernel
        device = runtime.RTSInterface()
        device.set_device(0)
        device.create_context("RT_CTX_NORMAL_MODE")
        registered_binary = device.register_device_binary_kernel("kernel_meta/%s.o" %
                                                                 self.attrs_map["kernel_name"].value, self.core_type)
        stubfunc_p = device.register_function(registered_binary,
                                              self.attrs_map["kernel_name"].value + "__kernel0", 0)
        tensor_arrays = [device.copy_bin_to_hbm(array.tonumpy().tobytes()) for array in args if array is not None]
        tensor_arrays += [device.copy_bin_to_hbm((self.workspace_byte * workspace)) for workspace in self.workspaces]
        device.launch_kernel(stubfunc_p, int(self.block_dim), tuple(tensor_arrays), len(tensor_arrays), None, None)
        device.synchronize_with_stream(None)
        result_arrays = [device.get_data_from_hbm(array_p, args[idx].tonumpy().nbytes)
                         for idx, array_p in enumerate(tensor_arrays) if idx < len(args)]
        idx_count = 0
        for result_array in args:
            if isinstance(result_array, Tensor):
                original_array = result_array.tonumpy()
                result_array.array = numpy.frombuffer(result_arrays[idx_count],
                                                      original_array.dtype).reshape(original_array.shape)
                idx_count += 1
        return self.workspaces
