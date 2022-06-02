/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common_autotiling_util.h"

#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"

#include <iostream>
#include <vector>

#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling_registry.h"

using namespace std;
using namespace ge;

namespace optiling {
namespace {
enum TilingOutputIndex {
  kOutputTilingKey,
  kOutputBlockDim,
  kOutputAtomicCleanFlag,
  kOutputTilingData,
  kOutputWorkspace,

  // add new output definitions here
  kOutputNum
};
}

void AutoTilingTest::GenFaker(const std::vector<std::vector<int64_t>>& ori_input_shapes,
                 const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& ori_output_shapes,
                 const std::vector<std::vector<int64_t>>& output_shapes,
                 const std::vector<ge::DataType>& input_type,
                 const std::vector<ge::DataType>& output_type,
                 const std::vector<ge::Format>& input_ori_format,
                 const std::vector<ge::Format>& input_format,
                 const std::vector<ge::Format>& output_ori_format,
                 const std::vector<ge::Format>& output_format) {
  size_t input_size = input_shapes.size();
  size_t output_size = output_shapes.size();
  tiling_data = gert::TilingData::CreateCap(2048);
  faker = faker.NodeIoNum(input_size, output_size).TilingData(tiling_data.get());
  store_input_shapes.resize(input_size);
  store_output_shapes.resize(output_size);

  std::vector<void*> input_shapes_ref(input_size);
  std::vector<void*> output_shapes_ref(output_size);
  for (size_t i = 0; i < input_size; i++) {
    for (int64_t dim : ori_input_shapes[i]) {
      store_input_shapes[i].MutableOriginShape().AppendDim(dim);
    }
    for (int64_t dim : input_shapes[i]) {
      store_input_shapes[i].MutableStorageShape().AppendDim(dim);
    }
    input_shapes_ref[i] = &store_input_shapes[i];
    faker = faker.NodeInputTd(i, input_type[i], input_ori_format[i], input_format[i]);
  }
  faker = faker.InputShapes(input_shapes_ref);

  std::vector<uint32_t> ir_num(input_size, 1);
  faker = faker.IrInstanceNum(ir_num);

  for (size_t i = 0; i < output_size; i++) {
    for (int64_t dim : ori_output_shapes[i]) {
      store_output_shapes[i].MutableOriginShape().AppendDim(dim);
    }
    for (int64_t dim : output_shapes[i]) {
      store_output_shapes[i].MutableStorageShape().AppendDim(dim);
    }
    // store_output_shapes.push_back(storage_shape);
    output_shapes_ref[i] = &store_output_shapes[i];
    faker = faker.NodeOutputTd(i, output_type[i], output_ori_format[i], output_format[i]);
  }

  faker = faker.OutputShapes(output_shapes_ref);
}

AutoTilingTest::AutoTilingTest(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes,
    ge::DataType input_type,
    ge::DataType output_type) {
  size_t input_size = input_shapes.size();
  size_t output_size = output_shapes.size();
  std::vector<ge::DataType> input_types(input_size, input_type);
  std::vector<ge::DataType> output_types(output_size, output_type);
  std::vector<ge::Format> input_formats(input_size, ge::FORMAT_ND);
  std::vector<ge::Format> output_formats(output_size, ge::FORMAT_ND);
  GenFaker(input_shapes, input_shapes, output_shapes, output_shapes, input_types,
           output_types, input_formats, input_formats, output_formats, output_formats);
}

AutoTilingTest::AutoTilingTest(const std::vector<std::vector<int64_t>>& input_shapes,
                               const std::vector<std::vector<int64_t>>& output_shapes,
                               const std::vector<ge::DataType>& input_types,
                               const std::vector<ge::DataType>& output_types) {
  size_t input_size = input_shapes.size();
  size_t output_size = output_shapes.size();
  std::vector<ge::Format> input_formats(input_size, ge::FORMAT_ND);
  std::vector<ge::Format> output_formats(output_size, ge::FORMAT_ND);
  GenFaker(input_shapes, input_shapes, output_shapes, output_shapes, input_types,
           output_types, input_formats, input_formats, output_formats, output_formats);
}

AutoTilingTest::AutoTilingTest(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes,
    ge::DataType input_type,
    ge::DataType output_type,
    ge::Format input_format,
    ge::Format output_format) {
  size_t input_size = input_shapes.size();
  size_t output_size = output_shapes.size();
  std::vector<ge::DataType> input_types(input_size, input_type);
  std::vector<ge::DataType> output_types(output_size, output_type);
  std::vector<ge::Format> input_formats(input_size, input_format);
  std::vector<ge::Format> output_formats(output_size, output_format);
  GenFaker(input_shapes, input_shapes, output_shapes, output_shapes, input_types,
           output_types, input_formats, input_formats, output_formats, output_formats);
}

AutoTilingTest::AutoTilingTest(
    const std::vector<std::vector<int64_t>>& ori_input_shapes,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int64_t>>& ori_output_shapes,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<ge::DataType>& input_type,
    const std::vector<ge::DataType>& output_type,
    const std::vector<ge::Format>& input_ori_format,
    const std::vector<ge::Format>& input_format,
    const std::vector<ge::Format>& output_ori_format,
    const std::vector<ge::Format>& output_format) {
  GenFaker(ori_input_shapes, input_shapes, ori_output_shapes, output_shapes, input_type,
           output_type, input_ori_format, input_format, output_ori_format, output_format);
}

gert::TilingContext* AutoTilingTest::GetContext() {
  Build();
  tiling_context = context_holder.GetContext<gert::TilingContext>();
  if (tiling_context == nullptr) {
    return nullptr;
  }
  return tiling_context;
}

void AutoTilingTest::Build() {
  if (!const_tensors.empty()) {
    faker.ConstInput(const_tensors);
  }
  context_holder = faker.Build();
}

bool AutoTilingTest::Test() {
  Build();
  auto context = context_holder.GetContext<gert::TilingContext>();
  if (context == nullptr) {
    return false;
  }

  if (gert::OpImplRegistry::GetInstance().GetOpImpl(op_type) == nullptr) {
    return false;
  }
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling;
  if (tiling_func == nullptr) {
    return false;
  }
  ge::Status status = tiling_func(context);
  if (status != ge::GRAPH_SUCCESS) {
    return false;
  }
  return true;
}

bool AutoTilingTest::Test(const OpInfo* ptr_op_info) {
  Build();
  auto context = context_holder.GetContext<gert::TilingContext>();
  if (context == nullptr) {
    return false;
  }
  if (!DoAutoTiling(context, ptr_op_info)) {
    return false;
  }
  return true;
}

AutoTilingTest& AutoTilingTest::SetOpType(std::string& _op_type) {
  op_type = _op_type;
  return *this;
}

AutoTilingTest& AutoTilingTest::SetTilingDataLen(size_t length) {
  tiling_data = gert::TilingData::CreateCap(length);
  faker = faker.TilingData(tiling_data.get());
  return *this;
}

AutoTilingTest& AutoTilingTest::SetWorkspace(size_t workspace_size) {
  workspace_holder = gert::ContinuousVector::Create<size_t>(8);
  auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspace_holder.get());
  faker = faker.Workspace(workspace);
  return *this;
}

AutoTilingTest& AutoTilingTest::SetInt32ConstInput(size_t const_index, int32_t* const_data, size_t data_size) {
  SetConstInput(const_index, ge::DT_INT32, const_data, data_size);
  return *this;
}

AutoTilingTest& AutoTilingTest::SetInt64ConstInput(size_t const_index, int64_t* const_data, size_t data_size) {
  SetConstInput(const_index, ge::DT_INT64, const_data, data_size);
  return *this;
}

std::string AutoTilingTest::GetInt32TilingData() {
  gert::TilingData *tiling_ptr = reinterpret_cast<gert::TilingData *>(tiling_data.get());
  std::string result;
  const int32_t *data = reinterpret_cast<const int32_t *>(tiling_ptr->GetData());
  size_t len = tiling_ptr->GetDataSize() / sizeof(int32_t);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    if (i != len - 1) {
      result += ", ";
    }
  }
  return result;
}

std::string AutoTilingTest::GetInt64TilingData() {
  gert::TilingData *tiling_ptr = reinterpret_cast<gert::TilingData *>(tiling_data.get());
  std::string result;
  const int64_t *data = reinterpret_cast<const int64_t *>(tiling_ptr->GetData());
  size_t len = tiling_ptr->GetDataSize() / sizeof(int64_t);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    if (i != len - 1) {
      result += ", ";
    }
  }
  return result;
}

int64_t AutoTilingTest::GetTilingKey() {
  auto context = context_holder.GetContext<gert::KernelContext>();
  auto p = context->GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return 0;
    }
  return *p;
}

uint32_t AutoTilingTest::GetBlockDims() {
  auto context = context_holder.GetContext<gert::KernelContext>();
  auto p = context->GetOutputPointer<uint32_t>(kOutputBlockDim);
  if (p == nullptr) {
    return 0;
  }
  return *p;
}

bool AutoTilingTest::GetAtomicFlag() {
  auto context = context_holder.GetContext<gert::KernelContext>();
  auto p = context->GetOutputPointer<uint32_t>(kOutputAtomicCleanFlag);
  if (p == nullptr) {
    return false;
  }
  return *p;
}

std::vector<size_t> AutoTilingTest::GetWorkspace() {
  auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspace_holder.get());
  std::vector<size_t> workspaces(workspace->GetSize());
  const size_t* workspace_data = reinterpret_cast<const size_t*>(workspace->GetData());
  for (size_t i = 0; i < workspace->GetSize(); i++) {
    workspaces[i] = workspace_data[i];
  }
  return workspaces;
}
} // optiling
