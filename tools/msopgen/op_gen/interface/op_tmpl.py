#!/usr/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves op file content.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020-2021
"""

# ==================1.ini file==================
INI_OP = """[{op_type}]
"""
INI_INPUT = """input{index}.name={name}
input{index}.dtype={dtype}
input{index}.paramType={paramType}
input{index}.format={format}
"""
INI_OUTPUT = """output{index}.name={name}
output{index}.dtype={dtype}
output{index}.paramType={paramType}
output{index}.format={format}
"""
INI_ATTR_LIST = """attr.list={attr_info}
"""
INI_ATTR_TYPE_VALUE = """attr_{name}.type={type}
attr_{name}.value=all
"""
INI_ATTR_PARAM_TYPE = """attr_{name}.paramType={paramType}
"""
INI_ATTR_DEFAULT_VALUE = """attr_{name}.defaultValue={defaultValue}
"""
INI_BIN_FILE = """opFile.value={name}
opInterface.value={name}
"""
# =============================================
# ==================2.IR file==================
IR_H_HEAD = """/**
 * Copyright (C)  2020-2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_{op_type_upper}_H
#define GE_OP_{op_type_upper}_H
#include "graph/operator_reg.h"
namespace ge {left_braces}

REG_OP({op_type})
"""
IR_H_INPUT = """    .INPUT({name}, TensorType({{{type}}}))
"""
IR_H_DYNAMIC_INPUT = """    .DYNAMIC_INPUT({name}, TensorType({{{type}}}))
"""
IR_H_OUTPUT = """    .OUTPUT({name}, TensorType({{{type}}}))
"""
IR_H_DYNAMIC_OUTPUT = """    .DYNAMIC_OUTPUT({name}, TensorType({{{type}}}))
"""
IR_H_ATTR_WITHOUT_VALUE = """    .REQUIRED_ATTR({name}, {type})
"""
IR_H_ATTR_WITH_VALUE = """    .ATTR({name}, {type}, {value})
"""
IR_H_END = """    .OP_END_FACTORY_REG({op_type})
{right_braces}
#endif //GE_OP_{op_type_upper}_H
"""
IR_CPP_HEAD = """#include "{fix_op_type}.h"
namespace ge {left_braces}

IMPLEMT_COMMON_INFERFUNC({op_type}InferShape)
{left_braces}
    return GRAPH_SUCCESS;
{right_braces}

IMPLEMT_VERIFIER({op_type}, {op_type}Verify)
{left_braces}
    return GRAPH_SUCCESS;
{right_braces}

COMMON_INFER_FUNC_REG({op_type}, {op_type}InferShape);
VERIFY_FUNC_REG({op_type}, {op_type}Verify);

{right_braces}  // namespace ge
"""
# =================================================
# ==================3.plugin file==================
TF_PLUGIN_CPP = """/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"

namespace domi {left_braces}
// register op info to GE
REGISTER_CUSTOM_OP("{name}")
    .FrameworkType({fmk_type})   // type: CAFFE, TENSORFLOW
    .OriginOpType("{name}")      // name in tf module
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
{right_braces}  // namespace domi
"""

ONNX_PLUGIN_CPP = """/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"

namespace domi {left_braces}
// Onnx ParseParams
Status ParseParam{name}(const Message* op_src, ge::Operator& op_dest) {left_braces}
    // To do: Implement the operator plugin by referring to the Onnx Operator Development Guide.
    return SUCCESS;
{right_braces}

// register {name} op info to GE
REGISTER_CUSTOM_OP("{name}")     // Set the registration name of operator
    .FrameworkType({fmk_type})   // Operator name with the original framework
    .OriginOpType("")      // Set the original frame type of the operator
    .ParseParamsFn(ParseParam{name}); // Registering the callback function for parsing operator parameters
{right_braces}  // namespace domi
"""

CAFFE_PLUGIN_CPP = """/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"
#include "graph/operator.h"

using namespace ge;

namespace domi
 {left_braces}
// Caffe ParseParams
Status ParseParam{name}(const Operator& op_src, ge::Operator& op_dest)
{left_braces}
    // To do: Implement the operator plug-in by referring to the TBE Operator Development Guide.
    return SUCCESS;
{right_braces}

// register op info to GE
REGISTER_CUSTOM_OP("{name}")
    .FrameworkType({fmk_type})    // type: CAFFE, TENSORFLOW
    .OriginOpType("{name}")       // name in caffe module
    .ParseParamsByOperatorFn(ParseParam{name});
{right_braces} // namespace domi
"""

PLUGIN_CMAKLIST = """# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
aux_source_directory(. SRCS)
message(STATUS "SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${TF_PLUGIN_TARGET}
            COMMAND mkdir -p ${TF_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${TF_PLUGIN_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${TF_PLUGIN_TARGET_OUT_DIR})

add_library(${TF_PLUGIN_TARGET} SHARED ${SRCS})

target_compile_definitions(${TF_PLUGIN_TARGET} PRIVATE
    google=ascend_private
)

target_link_libraries(${TF_PLUGIN_TARGET} ${ASCEND_INC}/../lib64/libgraph.so)
"""

ONNX_PLUGIN_CMAKLIST = """# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
aux_source_directory(. SRCS)
message(STATUS "SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${ONNX_PLUGIN_TARGET}
            COMMAND mkdir -p ${ONNX_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${ONNX_PLUGIN_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${ONNX_PLUGIN_TARGET_OUT_DIR})

add_library(${ONNX_PLUGIN_TARGET} SHARED ${SRCS})

target_compile_definitions(${ONNX_PLUGIN_TARGET} PRIVATE
    google=ascend_private
)

target_link_libraries(${ONNX_PLUGIN_TARGET} ${ASCEND_INC}/../lib64/libgraph.so)
"""

CAFFE_PLUGIN_CMAKLIST = """# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

aux_source_directory(. SRCS)
aux_source_directory(./proto/caffe PROTO_SRCS)
list(APPEND SRCS ${PROTO_SRCS})

message(STATUS "SRCS = ${SRCS}")

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${CAFFE_PLUGIN_TARGET}
            COMMAND mkdir -p ${CAFFE_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${CAFFE_PLUGIN_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${CAFFE_PLUGIN_TARGET_OUT_DIR})

include_directories(./proto/caffe)
add_library(${CAFFE_PLUGIN_TARGET} SHARED ${SRCS})
"""

CAFFE_CUSTOM_PROTO = """
syntax = "proto2";
package domi.caffe;
message NetParameter {
  optional string name = 1; // consider giving the network a name
  // The layers that make up the net.  Each of their configurations, including
  // connectivity and behavior, is specified as a LayerParameter.
  repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.

}
message LayerParameter {
  optional string name = 1;  // the layer name
  optional string type = 2;  // the layer type

  // Add new LayerParameter here.
  optional CustomTestParameter custom_test_param = 1000;
}

// Add the definition of LayerParameter here.
message CustomTestParameter {
    optional bool adj_x1 = 1 [default = false];
    optional bool adj_x2 = 2 [default = false];
}
"""
# =================================================
# ==================4.impl file==================
PY_HEAD = """import tbe.dsl as tbe
from tbe import tvm
from tbe.common.register import register_op_compute

"""
PY_COMPUTE_WITHOUT_ATTR = """
@register_op_compute("{name}")
def {name}_compute({input_name}, {output}, kernel_name="{name}"):
    \"""
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    \"""
"""
PY_COMPUTE_WITH_ATTR = """
@register_op_compute("{name}")
def {name}_compute({input_name}, {output}, {attr}, kernel_name="{name}"):
    \"""
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    \"""
"""
PY_COMPUTE_END = """
    res = tbe.XXX({input_name})
    return res
"""
PY_DEF_WITHOUT_ATTR = """
def {name}({input_name}, {output}, kernel_name="{name}"):
    \"""
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    \"""
"""
PY_DEF_WITH_ATTR = """
def {name}({input_name}, {output}, {attr}, kernel_name="{name}"):
    \"""
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    \"""
"""
PY_PLACEHOLDER = \
    """    data_{name} = tvm.placeholder({name}.get(\"shape\"), dtype={name}.get(\"dtype\"), name=\"data_{name}\")
"""

PY_RES_WITHOUT_ATTR = """
    res = {name}_compute({input_data}, {output_data}, kernel_name)
"""
PY_RES_WIT_ATTR = """
    res = {name}_compute({input_data}, {output_data}, {attr}, kernel_name)
"""
PY_TARGET_CCE = """
    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)
"""
PY_BUILD = """
    # operator build
    config = {left_braces}"name": kernel_name,
              "tensor_list": [{input_data}, res]{right_braces}
    tbe.build(schedule, config)
    """
# ==================4.2 MindSpore python file================
PY_MS_HEAD = """from __future__ import absolute_import
from te import tvm
from topi import generic
import te.lang.cce
from topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

"""
PY_MS_COMPUTE = """def {name}_compute({input_name}, {output}):
    \"""
    The compute function of the {up_name} implementation.
    \"""
    res = te.lang.cce.XXX({input_name})
    return res

"""
PY_MS_ATTR_WITHOUT_VALUE_INFO = \
    """.attr("{attr_name}", "{param_type}", "{attr_type}", "all")\\"""
PY_MS_INPUT_INFO = """.input(0, "{input_name}", False, "required", "all")\\"""
PY_MS_OUTPUT_INFO = """.output(0, "{output_name}", False, "required", "all")\\"""
PY_MS_DATA_TYPE = """DataType.{data_type}"""
PY_MS_DTYPE_FORMAT = """.dtype_format({data_types_join})\\"""
PY_MS_OP_WITHOUT_ATTR_INFO = """
# Define the kernel info of {up_name}.
{name}_op_info = TBERegOp("{up_name}") \\
    .fusion_type("OPAQUE") \\
    .partial_flag(True) \\
    .async_flag(False) \\
    .binfile_name("{name}.so") \\
    .compute_cost(10) \\
    .kernel_name("{name}_impl") \\
    {inputs}
    {outputs}
    {data_types}
    .get_op_info()

"""
PY_MS_OP_WITH_ATTR_INFO = """
# Define the kernel info of {up_name}.
{name}_op_info = TBERegOp("{up_name}") \\
    .fusion_type("OPAQUE") \\
    .partial_flag(True) \\
    .async_flag(False) \\
    .binfile_name("{name}.so") \\
    .compute_cost(10) \\
    .kernel_name("{name}_impl") \\
    {attrs}
    {inputs}
    {outputs}
    {data_types}
    .get_op_info()

"""
PY_MS_OP_INFO_REGISTER_TVM = \
    """data{data_count} = tvm.placeholder(shape, name="data{data_count}", dtype=dtype.lower())"""
PY_MS_OP_INFO_REGISTER = """
# Binding kernel info with the kernel implementation.
@op_info_register({name}_op_info)
def {name}_impl({input_name}, {output}, kernel_name="{name}_impl"):
    \"""
    The entry function of the {up_name} implementation.
    \"""
    shape = {input_x}.get("shape")
    dtype = {input_x}.get("dtype").lower()

    shape = util.shape_refine(shape)
    {tvm_placeholder}

    with tvm.target.cce():
        res = {name}_compute({datas_join}, {output})
        sch = generic.auto_schedule(res)
"""
PY_MS_OP_INFO_REGISTER_CONFIG = """
    config = {{"print_ir": False,
              "name": kernel_name,
              "tensor_list": [{datas_join}, res]}}

    te.lang.cce.cce_build_code(sch, config)
"""
PY_MS_PROTO_HEAD = """from mindspore.ops import prim_attr_register, \
PrimitiveWithInfer
import mindspore.ops as ops
# description
class {up_name}(PrimitiveWithInfer):
    \"""
    The definition of the {up_name} primitive.
    \"""
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['{input_name}'], outputs=['{output}'])
        # Import the entry function of the kernel implementation from relative
        #  path or PYTHONPATH.
        from {name}_impl import {name}_impl

    def infer_shape(self, {data_shapes}):
        return data1_shape

    def infer_dtype(self, {data_dtypes}):
        return data1_dtype"""
# ==================5.AICPU ini file==================
AICPU_INI_STRING = """[{op_type}]
opInfo.engine=DNN_VM_AICPU
opInfo.flagPartial=False
opInfo.computeCost=100
opInfo.flagAsync=False
opInfo.opKernelLib=CUSTAICPUKernel
opInfo.kernelSo=libcust_aicpu_kernels.so
opInfo.functionName=RunCpuKernel
opInfo.workspaceSize=1024
"""
# =============================================
# ==================6.AICPU impl cc file==================
AICPU_IMPL_CPP_STRING = """
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of {op_type}
 */
#include "{fix_op_type}_kernels.h"

namespace  {left_braces}
const char *{op_type_upper} = "{op_type}";
{right_braces}

namespace aicpu  {left_braces}
uint32_t {op_type}CpuKernel::Compute(CpuKernelContext &ctx)
{left_braces}
    return 0;
{right_braces}

REGISTER_CPU_KERNEL({op_type_upper}, {op_type}CpuKernel);
{right_braces} // namespace aicpu
"""
# =============================================
# ==================7.AICPU impl h file==================
AICPU_IMPL_H_STRING = """
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of {op_type}
 */

#ifndef _{op_type_upper}_KERNELS_H_
#define _{op_type_upper}_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {left_braces}
class {op_type}CpuKernel : public CpuKernel {left_braces}
public:
    ~{op_type}CpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
{right_braces};
{right_braces} // namespace aicpu
#endif
"""
# =======================================================
