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

/*!
 * \file common_autotiling_util.h
 */

#ifndef COMMON_AUTOTILING_UTIL_H
#define COMMON_AUTOTILING_UTIL_H

#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"
#include "auto_tiling_rt2.h"

using namespace std;

namespace optiling {
class AutoTilingTest {
public:
 AutoTilingTest() = default;
  AutoTilingTest(const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& output_shapes,
                 ge::DataType input_type,
                 ge::DataType output_type);
  AutoTilingTest(const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& output_shapes,
                 const std::vector<ge::DataType>& input_types,
                 const std::vector<ge::DataType>& output_types);
  AutoTilingTest(const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& output_shapes,
                 ge::DataType input_type,
                 ge::DataType output_type,
                 ge::Format input_format,
                 ge::Format output_format);
  AutoTilingTest(const std::vector<std::vector<int64_t>>& ori_input_shapes,
                 const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& ori_output_shapes,
                 const std::vector<std::vector<int64_t>>& output_shapes,
                 const std::vector<ge::DataType>& input_type,
                 const std::vector<ge::DataType>& output_type,
                 const std::vector<ge::Format>& input_ori_format,
                 const std::vector<ge::Format>& input_format,
                 const std::vector<ge::Format>& output_ori_format,
                 const std::vector<ge::Format>& output_format);
 ~AutoTilingTest() = default;

public:
 bool Test();
 bool Test(const OpInfo* ptr_op_info);
 AutoTilingTest& SetOpType(std::string& _op_type);
 AutoTilingTest& SetTilingDataLen(size_t length);
 AutoTilingTest& SetWorkspace(size_t workspace_size);
 AutoTilingTest& SetInt32ConstInput(size_t const_index, int32_t* const_data, size_t data_size);
 AutoTilingTest& SetInt64ConstInput(size_t const_index, int64_t* const_data, size_t data_size);
 std::string GetInt32TilingData();
 std::string GetInt64TilingData();
 int64_t GetTilingKey();
 uint32_t GetBlockDims();
 bool GetAtomicFlag();
 std::vector<size_t> GetWorkspace();
 gert::TilingContext* GetContext();

public:
 template <typename T>
 bool TestParse(std::string& json_str, T* compile_info) {
   if (gert::OpImplRegistry::GetInstance().GetOpImpl(op_type) == nullptr) {
     return false;
   }
   auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling_parse;
   if (tiling_prepare_func == nullptr) {
     return false;
   }
   char *json = const_cast<char *>(json_str.c_str());
   auto holder = gert::KernelRunContextFaker()
       .KernelIONum(1, 1)
       .Inputs({json})
       .Outputs({compile_info})
       .IrInstanceNum({1})
       .Build();
   if (tiling_prepare_func(holder.template GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
     return false;
   }
   auto cc = holder.template GetContext<gert::KernelContext>()->template GetOutputPointer<T>(0);
   *compile_info = *cc;
   return true;
  }

 template <typename T>
 AutoTilingTest& SetCompileInfo(std::string& json_str, T* compile_info) {
   if (gert::OpImplRegistry::GetInstance().GetOpImpl(op_type) == nullptr) {
     return *this;
   }
   auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type)->tiling_parse;
   if (tiling_prepare_func == nullptr) {
     return *this;
   }
   char *json = const_cast<char *>(json_str.c_str());
   auto holder = gert::KernelRunContextFaker()
       .KernelIONum(1, 1)
       .Inputs({json})
       .Outputs({compile_info})
       .IrInstanceNum({1})
       .Build();
   if (tiling_prepare_func(holder.template GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
     return *this;
   }
   auto cc = holder.template GetContext<gert::KernelContext>()->template GetOutputPointer<T>(0);
   *compile_info = *cc;
   return SetCompileInfo(compile_info);
 }

  template <typename T>
  AutoTilingTest& SetCompileInfo(T* compile_info) {
    if (compile_info != nullptr) {
      faker = faker.CompileInfo(compile_info);
    }
    return *this;
  }

 template <typename T>
 AutoTilingTest& SetAttrs(std::vector<std::pair<std::string, T>>& keys_to_value) {
  std::vector<std::pair<std::string, ge::AnyValue>> local_attrs;
  for (const auto& single_item : keys_to_value) {
    std::pair<std::string, ge::AnyValue> cur_attr = \
      std::make_pair(single_item.first, ge::AnyValue::CreateFrom<T>(single_item.second));
    local_attrs.push_back(cur_attr);
  }
  faker = faker.NodeAttrs(local_attrs);
  return *this;
 }

private:
  void Build();
  void GenFaker(const std::vector<std::vector<int64_t>>& ori_input_shapes,
                 const std::vector<std::vector<int64_t>>& input_shapes,
                 const std::vector<std::vector<int64_t>>& ori_output_shapes,
                 const std::vector<std::vector<int64_t>>& output_shapes,
                 const std::vector<ge::DataType>& input_type,
                 const std::vector<ge::DataType>& output_type,
                 const std::vector<ge::Format>& input_ori_format,
                 const std::vector<ge::Format>& input_format,
                 const std::vector<ge::Format>& output_ori_format,
                 const std::vector<ge::Format>& output_format);

 template <typename T>
 void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size) {
  std::unique_ptr<uint8_t[]> input_tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
  auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
  static int64_t offset = 0;
  *input_tensor = {
      {{data_size}, {data_size}},                          // storage shape
      {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // storage format
      gert::kFollowing,                    // placement
      dtype,                        // data type
      0,                                   // address
  };
  offset += sizeof(T) * data_size;
  auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
  for(int64_t i =0; i < data_size; i++) {
    tensor_data[i] = const_data[i];
  }
  input_tensor->SetData(gert::TensorData{tensor_data});
  auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
  const_tensors.push_back(std::move(pair));
 }

private:
 std::vector<gert::StorageShape> store_input_shapes;
 std::vector<gert::StorageShape> store_output_shapes;
 std::unique_ptr<uint8_t[]> tiling_data;
 std::unique_ptr<uint8_t[]> workspace_holder;
 std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
 gert::TilingContextFaker faker;
 gert::KernelRunContextHolder context_holder;
 gert::TilingContext* tiling_context;
 std::string op_type = "DefaultImpl";
};

} // namespace optiling

#endif  // COMMON_AUTOTILING_UTIL_H_