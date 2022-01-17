/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "folding.h"

#include <vector>
#include <set>
#include "cpu_attr_value.h"
#include "cpu_context.h"
#include "cpu_kernel_register.h"
#include "cpu_kernel_utils.h"
#include "device_cpu_kernel.h"
#include "cpu_node_def.h"
#include "log.h"

namespace {
const char *kVtString = "VT_STRING";
const char *kVtListString = "VT_LIST_STRING";
const char *kVtFloat = "VT_FLOAT";
const char *kVtListFloat = "VT_LIST_FLOAT";
const char *kVtInt = "VT_INT";
const char *kVtListInt = "VT_LIST_INT";
const char *kVtListListInt = "VT_LIST_LIST_INT";
const char *kVtBool = "VT_BOOL";
const char *kVtListBool = "VT_LIST_BOOL";
const char *kVtDataType = "VT_DATA_TYPE";
const char *kVtListDataType = "VT_LIST_DATA_TYPE";
const char *kVtTensor = "VT_TENSOR";
const char *kVtListTensor = "VT_LIST_TENSOR";

void ConvertGeToAicpuTensor(const ge::GeTensorDesc &tensor_desc,
                            const std::string &tensor_name,
                            const ge::Tensor &ge_tensor,
                            std::shared_ptr<aicpu::Tensor> &aicpu_tensor) {
  aicpu::CpuKernelUtils::SetTensorName(tensor_name, aicpu_tensor);
  aicpu_tensor->SetDataType(
      static_cast<aicpu::DataType>(tensor_desc.GetDataType()));
  aicpu_tensor->SetData(const_cast<uint8_t *>(ge_tensor.GetData()));
  aicpu_tensor->SetDataSize(ge_tensor.GetSize());
  auto shape = aicpu_tensor->GetTensorShape();
  if (shape != nullptr) {
    ge::GeShape ge_shape = tensor_desc.GetShape();
    std::vector<int64_t> shape_dims = ge_shape.GetDims();
    shape->SetDimSizes(shape_dims);
    shape->SetFormat(static_cast<aicpu::Format>(tensor_desc.GetFormat()));
  }
  CPU_LOG_INFO("Op set tensor[%s], tensor info[type:%d, data:%p, size:%llu].",
               tensor_name.c_str(), tensor_desc.GetDataType(),
               ge_tensor.GetData(), ge_tensor.GetSize());
}

int32_t AddStringAttrToNodeDef(ge::Operator &op, const std::string &name,
                               aicpu::NodeDef *node_def) {
  std::string s;
  ge::graphStatus ret = op.GetAttr(name, s);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetString(s);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListStringAttrToNodeDef(ge::Operator &op, const std::string &name,
                                   aicpu::NodeDef *node_def) {
  std::vector<std::string> list_s;
  ge::graphStatus ret = op.GetAttr(name, list_s);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetListString(list_s);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddFloatAttrToNodeDef(ge::Operator &op, const std::string &name,
                              aicpu::NodeDef *node_def) {
  float f = 0;
  ge::graphStatus ret = op.GetAttr(name, f);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetFloat(f);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListFloatAttrToNodeDef(ge::Operator &op, const std::string &name,
                                  aicpu::NodeDef *node_def) {
  std::vector<float> list_f;
  ge::graphStatus ret = op.GetAttr(name, list_f);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetListFloat(list_f);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddBoolAttrToNodeDef(ge::Operator &op, const std::string &name,
                             aicpu::NodeDef *node_def) {
  bool b = false;
  ge::graphStatus ret = op.GetAttr(name, b);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetBool(b);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListBoolAttrToNodeDef(ge::Operator &op, const std::string &name,
                                 aicpu::NodeDef *node_def) {
  std::vector<bool> list_b;
  ge::graphStatus ret = op.GetAttr(name, list_b);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetListBool(list_b);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddIntAttrToNodeDef(ge::Operator &op, const std::string &name,
                            aicpu::NodeDef *node_def) {
  int64_t i = 0;
  ge::graphStatus ret = op.GetAttr(name, i);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetInt(i);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListIntAttrToNodeDef(ge::Operator &op, const std::string &name,
                                aicpu::NodeDef *node_def) {
  std::vector<int64_t> list_i;
  ge::graphStatus ret = op.GetAttr(name, list_i);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetListInt(list_i);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListListIntAttrToNodeDef(ge::Operator &op, const std::string &name,
                                aicpu::NodeDef *node_def) {
  std::vector<std::vector<int64_t>> list_i;
  ge::graphStatus ret = op.GetAttr(name, list_i);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetListListInt(list_i);
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddDataTypeAttrToNodeDef(ge::Operator &op, const std::string &name,
                                 aicpu::NodeDef *node_def) {
  ge::DataType data_type = ge::DT_UNDEFINED;
  ge::graphStatus ret = op.GetAttr(name, data_type);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  attr->SetDataType(static_cast<aicpu::DataType>(data_type));
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListDataTypeAttrToNodeDef(ge::Operator &op, const std::string &name,
                                     aicpu::NodeDef *node_def) {
  std::vector<ge::DataType> list_type;
  ge::graphStatus ret = op.GetAttr(name, list_type);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")
  for (const ge::DataType &data_type : list_type) {
    attr->AddListDataType(static_cast<aicpu::DataType>(data_type));
  }
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddTensorAttrToNodeDef(ge::Operator &op, const std::string &name,
                               aicpu::NodeDef *node_def) {
  ge::Tensor ge_tensor;
  ge::graphStatus ret = op.GetAttr(name, ge_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")

  auto aicpu_tensor = aicpu::CpuKernelUtils::CreateTensor();
  CPU_CHECK_NULLPTR_WARN(aicpu_tensor, -1, "Op create tensor failed.")
  ge::TensorDesc ge_tensor_desc = ge_tensor.GetTensorDesc();
  aicpu_tensor->SetDataType(
      static_cast<aicpu::DataType>(ge_tensor_desc.GetDataType()));
  aicpu_tensor->SetData(const_cast<uint8_t *>(ge_tensor.GetData()));
  aicpu_tensor->SetDataSize(ge_tensor.GetSize());

  auto aicpu_tensor_shape = aicpu_tensor->GetTensorShape();
  CPU_CHECK_NULLPTR_WARN(aicpu_tensor_shape, -1,
                         "Op create tensor shape failed.")
  aicpu_tensor_shape->SetDimSizes(ge_tensor_desc.GetShape().GetDims());
  attr->SetTensor(aicpu_tensor.get());
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListTensorAttrToNodeDef(ge::Operator &op, const std::string &name,
                                   aicpu::NodeDef *node_def) {
  std::vector<ge::Tensor> ge_list_tensor;
  ge::graphStatus ret = op.GetAttr(name, ge_list_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    CPU_LOG_WARN("Op get attr[%s] failed.", name.c_str());
    return -1;
  }

  auto attr = aicpu::CpuKernelUtils::CreateAttrValue();
  CPU_CHECK_NULLPTR_WARN(attr, -1, "Op create attr value failed.")

  for (const ge::Tensor &ge_tensor : ge_list_tensor) {
    auto aicpu_tensor = attr->AddListTensor();
    CPU_CHECK_NULLPTR_WARN(aicpu_tensor, -1, "Op attr add tensor failed.")
    ge::TensorDesc ge_tensor_desc = ge_tensor.GetTensorDesc();
    aicpu_tensor->SetDataType(
        static_cast<aicpu::DataType>(ge_tensor_desc.GetDataType()));
    aicpu_tensor->SetData(const_cast<uint8_t *>(ge_tensor.GetData()));
    aicpu_tensor->SetDataSize(ge_tensor.GetSize());

    auto aicpu_tensor_shape = aicpu_tensor->GetTensorShape();
    CPU_CHECK_NULLPTR_WARN(aicpu_tensor_shape, -1,
                           "Op create tensor shape failed.")
    aicpu_tensor_shape->SetDimSizes(ge_tensor_desc.GetShape().GetDims());
  }
  node_def->AddAttrs(name, attr.get());
  return 0;
}

int32_t AddListAttrToNodeDef(ge::Operator &op, const std::string &name,
                             const std::string &type,
                             aicpu::NodeDef *node_def) {
  int32_t ret = 0;
  if (type == kVtListString) {
    ret = AddListStringAttrToNodeDef(op, name, node_def);
  } else if (type == kVtListFloat) {
    ret = AddListFloatAttrToNodeDef(op, name, node_def);
  } else if (type == kVtListInt) {
    ret = AddListIntAttrToNodeDef(op, name, node_def);
  } else if (type == kVtListListInt) {
    ret = AddListListIntAttrToNodeDef(op, name, node_def);
  } else if (type == kVtListBool) {
    ret = AddListBoolAttrToNodeDef(op, name, node_def);
  } else if (type == kVtListDataType) {
    ret = AddListDataTypeAttrToNodeDef(op, name, node_def);
  } else if (type == kVtListTensor) {
    ret = AddListTensorAttrToNodeDef(op, name, node_def);
  } else {
    CPU_LOG_WARN("Attr type is unsuported, name: [%s], type: [%s].",
                 name.c_str(), type.c_str());
  }
  return ret;
}

int32_t AddAttrToNodeDef(ge::Operator &op, const std::string &name,
                         const std::string &type, aicpu::NodeDef *node_def) {
  int32_t ret = 0;
  if (type.empty() || type[0] == '_') {
    return ret;
  }
  if (type == kVtString) {
    ret = AddStringAttrToNodeDef(op, name, node_def);
  } else if (type == kVtFloat) {
    ret = AddFloatAttrToNodeDef(op, name, node_def);
  } else if (type == kVtInt) {
    ret = AddIntAttrToNodeDef(op, name, node_def);
  } else if (type == kVtBool) {
    ret = AddBoolAttrToNodeDef(op, name, node_def);
  } else if (type == kVtDataType) {
    ret = AddDataTypeAttrToNodeDef(op, name, node_def);
  } else if (type == kVtTensor) {
    ret = AddTensorAttrToNodeDef(op, name, node_def);
  } else {
    ret = AddListAttrToNodeDef(op, name, type, node_def);
  }
  return ret;
}
}  // namespace

extern "C" {
__attribute__((visibility("default"))) int32_t InitCpuConstantFolding(
    ge::HostCpuOp *(*create_fn)()) {
  CPU_LOG_INFO("Init cpu constant folding begin.");
  std::set<std::string> black_list = {"Assign", "NoOp"};
  std::vector<std::string> ops =
      aicpu::CpuKernelRegister::Instance().GetAllRegisteredOpTypes();
  for (const std::string &op_type : ops) {
    if (black_list.find(op_type) != black_list.end()) {
      continue;
    }
    CPU_LOG_INFO("Register op[%s].", op_type.c_str());
    ::ge::HostCpuOpRegistrar registrar __attribute__((unused)) =
        ::ge::HostCpuOpRegistrar(op_type.c_str(), create_fn);
  }
  return 0;
}

__attribute__((visibility("default"))) int32_t CpuConstantFoldingCompute(
    ge::Operator &op, const std::map<std::string, const ge::Tensor> &inputs,
    std::map<std::string, ge::Tensor> outputs) {
  const string &op_type = op.GetOpType();
  CPU_LOG_INFO("Enter cpu op[%s].", op_type.c_str());
  auto kernel = aicpu::CpuKernelRegister::Instance().GetCpuKernel(op_type);
  CPU_CHECK_NULLPTR_WARN(kernel, -1, "Op[%s] is not registered in cpu kernels.",
                         op_type.c_str())

  auto node_def = aicpu::CpuKernelUtils::CreateNodeDef();
  CPU_CHECK_NULLPTR_WARN(node_def, -1, "Op[%s] create node def failed.",
                         op_type.c_str())

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CPU_CHECK_NULLPTR_WARN(
      op_desc, -1, "Op[%s] get op desc from operator failed.", op_type.c_str())

  node_def->SetOpType(op_type);
  size_t input_size = op_desc->GetAllInputsSize();
  for (size_t i = 0; i < input_size; ++i) {
    ge::GeTensorDescPtr input_desc = op_desc->MutableInputDesc(i);
    if (input_desc == nullptr) {
      continue;
    }
    std::string input_name = op_desc->GetInputNameByIndex(i);
    auto iter = inputs.find(input_name);
    if (iter == inputs.end()) {
      CPU_LOG_WARN("Op[%s] input tensor[%s] is not found in inputs.",
                   op_type.c_str(), input_name.c_str());
      return -1;
    }

    auto input_tensor = node_def->AddInputs();
    CPU_CHECK_NULLPTR_WARN(input_tensor, -1,
                           "Op[%s] node def add input[%zu] failed.",
                           op_type.c_str(), i)
    ConvertGeToAicpuTensor(*input_desc, input_name, iter->second, input_tensor);
  }

  size_t output_size = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++i) {
    ge::GeTensorDesc output_desc = op_desc->GetOutputDesc(i);
    std::string output_name = op_desc->GetOutputNameByIndex(i);
    auto iter = outputs.find(output_name);
    if (iter == outputs.end()) {
      CPU_LOG_WARN("Op[%s] output tensor[%s] is not found in outputs.",
                   op_type.c_str(), output_name.c_str());
      return -1;
    }

    auto output_tensor = node_def->AddOutputs();
    CPU_CHECK_NULLPTR_WARN(output_tensor, -1,
                           "Op[%s] node def add input[%zu] failed.",
                           op_type.c_str(), i)
    ConvertGeToAicpuTensor(output_desc, output_name, iter->second,
                           output_tensor);
  }

  auto attrs = op.GetAllAttrNamesAndTypes();
  for (const auto &attr : attrs) {
    std::string name = attr.first;
    std::string type = attr.second;
    int32_t ret = AddAttrToNodeDef(op, name, type, node_def.get());
    if (ret != 0) {
      return ret;
    }
  }

  aicpu::CpuKernelContext ctx(aicpu::HOST);
  uint32_t ret = ctx.Init(node_def.get());
  if (ret != 0) {
    return -1;
  }

  ret = aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx);
  if (ret != 0) {
    return -1;
  }

  CPU_LOG_INFO("Finish cpu op[%s].", op_type.c_str());
  return 0;
}

__attribute__((visibility("default"))) uint32_t RunHostCpuKernel(void *param) {
  return RunCpuKernel(param);
}
}
