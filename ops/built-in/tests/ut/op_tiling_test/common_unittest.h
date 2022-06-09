/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file common_unittest.h
 */

#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"
#include "graph/utils/op_desc_utils.h"
using namespace std;
using namespace ge;

typedef std::string (*BufToString)(void*, size_t);
void TestTilingParse(const std::string optype, std::string json_str, void* compile_info, ge::graphStatus result);

template <typename T>
string to_string(void* buf, size_t size) {
  std::string result;
  const T* data = reinterpret_cast<const T*>(buf);
  size_t len = size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

static uint8_t* GetConstTensor(ge::Operator& op, const size_t index) {
  ge::Tensor const_tensor;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_name = op_desc->GetInputNameByIndex(index);
  if (op.GetInputConstData(input_name.c_str(), const_tensor) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  auto size = const_tensor.GetSize();
  auto data = const_tensor.GetData();
  ge::DataType const_dtype = op_desc->MutableInputDesc(index)->GetDataType();
  uint8_t* input_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder);
  std::memcpy(input_tensor + 1, data, size);
  int64_t value_size =
      (const_dtype == ge::DT_INT64 || const_dtype == ge::DT_UINT64) ? size / sizeof(int64_t) : size / sizeof(int32_t);
  gert::Tensor tensor({{value_size}, {value_size}},        // shape
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // format
                      gert::kFollowing,                    // placement
                      const_dtype,                         // dt
                      nullptr);
  std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
  return input_tensor_holder;
}

static const std::map<std::string, ge::AnyValue::ValueType> kAttrTypesMap = {
    {"VT_INT", ge::AnyValue::ValueType::VT_INT},
    {"VT_BOOL", ge::AnyValue::ValueType::VT_BOOL},
    {"VT_FLOAT", ge::AnyValue::ValueType::VT_FLOAT},
    {"VT_STRING", ge::AnyValue::ValueType::VT_STRING},
    {"VT_LIST_INT", ge::AnyValue::ValueType::VT_LIST_INT},
    {"VT_LIST_BOOL", ge::AnyValue::ValueType::VT_LIST_BOOL},
    {"VT_LIST_FLOAT", ge::AnyValue::ValueType::VT_LIST_FLOAT},
    {"VT_LIST_LIST_INT", ge::AnyValue::ValueType::VT_LIST_LIST_INT},
};

#define ATTACH_ATTRS_TO_FAKER(faker, op, attrs_name)                               \
  auto op_attrs_map = op.GetAllAttrNamesAndTypes();                                \
  std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value;                 \
  std::pair<std::string, ge::AnyValue> p;                                          \
  if (attrs_name.size() > 0) {                                                     \
    for (auto item : attrs_name) {                                                 \
      p.first = item;                                                              \
      auto attr_it = op_attrs_map.find(item);                                      \
      if (attr_it != op_attrs_map.end()) {                                         \
        auto type_it = kAttrTypesMap.find(attr_it->second);                        \
        if (type_it != kAttrTypesMap.end()) {                                      \
          switch (type_it->second) {                                               \
            case ge::AnyValue::ValueType::VT_BOOL: {                               \
              bool value;                                                          \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<bool>(value);                    \
            } break;                                                               \
            case ge::AnyValue::ValueType::VT_INT: {                                \
              int64_t value;                                                       \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<int64_t>(value);                 \
            } break;                                                               \
            case ge::AnyValue::ValueType::VT_FLOAT: {                              \
              float32_t value;                                                     \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<float32_t>(value);               \
            } break;                                                               \
            case ge::AnyValue::ValueType::VT_STRING: {                             \
              string value;                                                        \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<string>(value);                  \
            } break;                                                               \
            case ge::AnyValue::ValueType::VT_LIST_INT: {                           \
              vector<int64_t> value;                                               \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<vector<int64_t>>(value);         \
            } break;                                                               \
            case ge::AnyValue::ValueType::VT_LIST_BOOL: {                          \
              vector<bool> value;                                                  \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<vector<bool>>(value);            \
            } break;                                                               \
            case ge::AnyValue::ValueType::VT_LIST_LIST_INT: {                      \
              vector<vector<int64_t>> value;                                       \
              ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);                   \
              p.second = ge::AnyValue::CreateFrom<vector<vector<int64_t>>>(value); \
            } break;                                                               \
          }                                                                        \
        }                                                                          \
      }                                                                            \
      keys_to_value.push_back(p);                                                  \
    }                                                                              \
    faker = faker.NodeAttrs(keys_to_value);                                        \
  }

#define TILING_PARSE_JSON_TO_COMPILEINFO(optype, json, compile_info) \
  TestTilingParse(optype, json, &compile_info, ge::GRAPH_SUCCESS)
#define TILING_PARSE_JSON_TO_COMPILEINFO_ERROR(optype, json, compile_info) \
  TestTilingParse(optype, json, &compile_info, ge::GRAPH_FAILED)

#define ATTACH_OPERATOR_TO_HOLDER(holder, op, tiling_len, compile_info)                               \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                                        \
  ASSERT_NE(operator_info, nullptr);                                                                  \
  size_t input_size = operator_info->GetInputsSize();                                                 \
  size_t output_size = operator_info->GetOutputsSize();                                               \
  auto param = gert::TilingData::CreateCap(tiling_len);                                               \
  auto faker = gert::TilingContextFaker().NodeIoNum(input_size, output_size).TilingData(param.get()); \
  faker = faker.CompileInfo(&compile_info);                                                           \
  std::vector<gert::StorageShape> storage_shapes(input_size);                                         \
  std::vector<void*> input_shapes_ref(input_size);                                                    \
  std::vector<uint8_t*> const_tensors;                                                                \
  if (input_size > 0) {                                                                               \
    for (size_t i = 0; i < input_size; ++i) {                                                         \
      auto input_desc = operator_info->MutableInputDesc(i);                                           \
      ASSERT_NE(input_desc, nullptr);                                                                 \
      ge::Format input_format = input_desc->GetFormat();                                              \
      ge::Format origin_format = input_desc->GetOriginFormat();                                       \
      ge::DataType dtype = input_desc->GetDataType();                                                 \
      faker = faker.NodeInputTd(i, dtype, origin_format, input_format);                               \
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                                    \
        storage_shapes[i].MutableOriginShape().AppendDim(dim);                                        \
      }                                                                                               \
      for (int64_t dim : input_desc->MutableShape().GetDims()) {                                      \
        storage_shapes[i].MutableStorageShape().AppendDim(dim);                                       \
      }                                                                                               \
      input_shapes_ref[i] = &storage_shapes[i];                                                       \
    }                                                                                                 \
    faker = faker.InputShapes(input_shapes_ref);                                                      \
    std::vector<uint32_t> irnum(input_size, 1);                                                       \
    faker = faker.IrInstanceNum(irnum);                                                               \
  }                                                                                                   \
  std::vector<gert::StorageShape> output_shapes(output_size);                                         \
  std::vector<void*> output_shapes_ref(output_size);                                                  \
  if (output_size > 0) {                                                                              \
    for (size_t i = 0; i < output_size; ++i) {                                                        \
      auto output_desc = operator_info->MutableOutputDesc(i);                                         \
      ASSERT_NE(output_desc, nullptr);                                                                \
      ge::Format input_format = output_desc->GetFormat();                                             \
      ge::Format origin_format = output_desc->GetOriginFormat();                                      \
      ge::DataType dtype = output_desc->GetDataType();                                                \
      faker = faker.NodeOutputTd(i, dtype, origin_format, input_format);                              \
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {                                   \
        output_shapes[i].MutableOriginShape().AppendDim(dim);                                         \
      }                                                                                               \
      for (int64_t dim : output_desc->MutableShape().GetDims()) {                                     \
        output_shapes[i].MutableStorageShape().AppendDim(dim);                                        \
      }                                                                                               \
      output_shapes_ref[i] = &output_shapes[i];                                                       \
    }                                                                                                 \
    faker = faker.OutputShapes(output_shapes_ref);                                                    \
  }                                                                                                   \
  auto holder = faker.Build()

#define ATTACH_OPERATOR_TO_HOLDER_ATTRS(holder, op, attrs, tiling_len, compile_info)                  \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                                        \
  ASSERT_NE(operator_info, nullptr);                                                                  \
  size_t input_size = operator_info->GetInputsSize();                                                 \
  size_t output_size = operator_info->GetOutputsSize();                                               \
  auto param = gert::TilingData::CreateCap(tiling_len);                                               \
  auto faker = gert::TilingContextFaker().NodeIoNum(input_size, output_size).TilingData(param.get()); \
  faker = faker.CompileInfo(&compile_info);                                                           \
  std::vector<gert::StorageShape> storage_shapes(input_size);                                         \
  std::vector<void*> input_shapes_ref(input_size);                                                    \
  std::vector<uint8_t*> const_tensors;                                                                \
  if (input_size > 0) {                                                                               \
    for (size_t i = 0; i < input_size; ++i) {                                                         \
      auto input_desc = operator_info->MutableInputDesc(i);                                           \
      ASSERT_NE(input_desc, nullptr);                                                                 \
      ge::Format input_format = input_desc->GetFormat();                                              \
      ge::Format origin_format = input_desc->GetOriginFormat();                                       \
      ge::DataType dtype = input_desc->GetDataType();                                                 \
      faker = faker.NodeInputTd(i, dtype, origin_format, input_format);                               \
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                                    \
        storage_shapes[i].MutableOriginShape().AppendDim(dim);                                        \
      }                                                                                               \
      for (int64_t dim : input_desc->MutableShape().GetDims()) {                                      \
        storage_shapes[i].MutableStorageShape().AppendDim(dim);                                       \
      }                                                                                               \
      input_shapes_ref[i] = &storage_shapes[i];                                                       \
    }                                                                                                 \
    faker = faker.InputShapes(input_shapes_ref);                                                      \
    std::vector<uint32_t> irnum(input_size, 1);                                                       \
    faker = faker.IrInstanceNum(irnum);                                                               \
  }                                                                                                   \
  std::vector<gert::StorageShape> output_shapes(output_size);                                         \
  std::vector<void*> output_shapes_ref(output_size);                                                  \
  if (output_size > 0) {                                                                              \
    for (size_t i = 0; i < output_size; ++i) {                                                        \
      auto output_desc = operator_info->MutableOutputDesc(i);                                         \
      ASSERT_NE(output_desc, nullptr);                                                                \
      ge::Format input_format = output_desc->GetFormat();                                             \
      ge::Format origin_format = output_desc->GetOriginFormat();                                      \
      ge::DataType dtype = output_desc->GetDataType();                                                \
      faker = faker.NodeOutputTd(i, dtype, origin_format, input_format);                              \
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {                                   \
        output_shapes[i].MutableOriginShape().AppendDim(dim);                                         \
      }                                                                                               \
      for (int64_t dim : output_desc->MutableShape().GetDims()) {                                     \
        output_shapes[i].MutableStorageShape().AppendDim(dim);                                        \
      }                                                                                               \
      output_shapes_ref[i] = &output_shapes[i];                                                       \
    }                                                                                                 \
    faker = faker.OutputShapes(output_shapes_ref);                                                    \
  }                                                                                                   \
  ATTACH_ATTRS_TO_FAKER(faker, op, attrs);                                                            \
  auto holder = faker.Build()

#define ATTACH_OPERATOR_TO_HOLDER_IRNUM(holder, op, irnum, attrs, tiling_len, compile_info)           \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                                        \
  ASSERT_NE(operator_info, nullptr);                                                                  \
  size_t input_size = operator_info->GetInputsSize();                                                 \
  size_t output_size = operator_info->GetOutputsSize();                                               \
  auto param = gert::TilingData::CreateCap(tiling_len);                                               \
  auto faker = gert::TilingContextFaker().NodeIoNum(input_size, output_size).TilingData(param.get()); \
  faker = faker.CompileInfo(&compile_info);                                                           \
  std::vector<gert::StorageShape> storage_shapes(input_size);                                         \
  std::vector<void*> input_shapes_ref(input_size);                                                    \
  std::vector<uint8_t*> const_tensors;                                                                \
  size_t count = 0;                                                                                   \
  for (size_t idx = 0; idx < irnum.size(); idx++) {                                                   \
    for (size_t i = 0; i < irnum[idx]; ++i) {                                                         \
      auto input_desc = operator_info->MutableInputDesc(count);                                       \
      ASSERT_NE(input_desc, nullptr);                                                                 \
      ge::Format input_format = input_desc->GetFormat();                                              \
      ge::Format origin_format = input_desc->GetOriginFormat();                                       \
      ge::DataType dtype = input_desc->GetDataType();                                                 \
      faker = faker.NodeInputTd(count, dtype, origin_format, input_format);                           \
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                                    \
        storage_shapes[count].MutableOriginShape().AppendDim(dim);                                    \
      }                                                                                               \
      for (int64_t dim : input_desc->MutableShape().GetDims()) {                                      \
        storage_shapes[count].MutableStorageShape().AppendDim(dim);                                   \
      }                                                                                               \
      input_shapes_ref[count] = &storage_shapes[count];                                               \
      count++;                                                                                        \
    }                                                                                                 \
    faker = faker.InputShapes(input_shapes_ref);                                                      \
    faker = faker.IrInstanceNum(irnum);                                                               \
  }                                                                                                   \
  std::vector<gert::StorageShape> output_shapes(output_size);                                         \
  std::vector<void*> output_shapes_ref(output_size);                                                  \
  if (output_size > 0) {                                                                              \
    for (size_t i = 0; i < output_size; ++i) {                                                        \
      auto output_desc = operator_info->MutableOutputDesc(i);                                         \
      ASSERT_NE(output_desc, nullptr);                                                                \
      ge::Format input_format = output_desc->GetFormat();                                             \
      ge::Format origin_format = output_desc->GetOriginFormat();                                      \
      ge::DataType dtype = output_desc->GetDataType();                                                \
      faker = faker.NodeOutputTd(i, dtype, origin_format, input_format);                              \
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {                                   \
        output_shapes[i].MutableOriginShape().AppendDim(dim);                                         \
      }                                                                                               \
      for (int64_t dim : output_desc->MutableShape().GetDims()) {                                     \
        output_shapes[i].MutableStorageShape().AppendDim(dim);                                        \
      }                                                                                               \
      output_shapes_ref[i] = &output_shapes[i];                                                       \
    }                                                                                                 \
    faker = faker.OutputShapes(output_shapes_ref);                                                    \
  }                                                                                                   \
  ATTACH_ATTRS_TO_FAKER(faker, op, attrs);                                                            \
  auto holder = faker.Build()

#define ATTACH_OPERATOR_TO_HOLDER_CONST(holder, op, input_const, attrs, tiling_len, info)             \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                                        \
  ASSERT_NE(operator_info, nullptr);                                                                  \
  size_t input_size = operator_info->GetInputsSize();                                                 \
  size_t output_size = operator_info->GetOutputsSize();                                               \
  auto param = gert::TilingData::CreateCap(tiling_len);                                               \
  auto faker = gert::TilingContextFaker().NodeIoNum(input_size, output_size).TilingData(param.get()); \
  faker = faker.CompileInfo(&info);                                                                   \
  std::vector<gert::StorageShape> storage_shapes(input_size);                                         \
  std::vector<void*> input_shapes_ref(input_size);                                                    \
  std::vector<uint8_t*> const_tensors;                                                                \
  if (input_size > 0) {                                                                               \
    for (size_t i = 0; i < input_size; ++i) {                                                         \
      if (input_const[i]) {                                                                           \
        uint8_t* input_tensor_holder = GetConstTensor(op, i);                                         \
        if (input_tensor_holder == nullptr)                                                           \
          continue;                                                                                   \
        input_shapes_ref[i] = input_tensor_holder;                                                    \
        const_tensors.push_back(input_tensor_holder);                                                 \
      } else {                                                                                        \
        auto input_desc = operator_info->MutableInputDesc(i);                                         \
        ASSERT_NE(input_desc, nullptr);                                                               \
        ge::Format input_format = input_desc->GetFormat();                                            \
        ge::Format origin_format = input_desc->GetOriginFormat();                                     \
        ge::DataType dtype = input_desc->GetDataType();                                               \
        faker = faker.NodeInputTd(i, dtype, origin_format, input_format);                             \
        for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                                  \
          storage_shapes[i].MutableOriginShape().AppendDim(dim);                                      \
        }                                                                                             \
        for (int64_t dim : input_desc->MutableShape().GetDims()) {                                    \
          storage_shapes[i].MutableStorageShape().AppendDim(dim);                                     \
        }                                                                                             \
        input_shapes_ref[i] = &storage_shapes[i];                                                     \
      }                                                                                               \
    }                                                                                                 \
    faker = faker.InputShapes(input_shapes_ref);                                                      \
    std::vector<uint32_t> irnum(input_size, 1);                                                       \
    faker = faker.IrInstanceNum(irnum);                                                               \
  }                                                                                                   \
  std::vector<gert::StorageShape> output_shapes(output_size);                                         \
  std::vector<void*> output_shapes_ref(output_size);                                                  \
  if (output_size > 0) {                                                                              \
    for (size_t i = 0; i < output_size; ++i) {                                                        \
      auto output_desc = operator_info->MutableOutputDesc(i);                                         \
      ASSERT_NE(output_desc, nullptr);                                                                \
      ge::Format input_format = output_desc->GetFormat();                                             \
      ge::Format origin_format = output_desc->GetOriginFormat();                                      \
      ge::DataType dtype = output_desc->GetDataType();                                                \
      faker = faker.NodeOutputTd(i, dtype, input_format, input_format);                               \
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {                                   \
        output_shapes[i].MutableOriginShape().AppendDim(dim);                                         \
      }                                                                                               \
      for (int64_t dim : output_desc->MutableShape().GetDims()) {                                     \
        output_shapes[i].MutableStorageShape().AppendDim(dim);                                        \
      }                                                                                               \
      output_shapes_ref[i] = &output_shapes[i];                                                       \
    }                                                                                                 \
    faker = faker.OutputShapes(output_shapes_ref);                                                    \
  }                                                                                                   \
  ATTACH_ATTRS_TO_FAKER(faker, op, attrs);                                                            \
  auto holder = faker.Build()

#define HOLDER_DO_TILING(holder, optype, expect)                                    \
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(optype), nullptr);        \
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling; \
  ASSERT_NE(tiling_func, nullptr);                                                  \
  gert::TilingContext* context = holder.GetContext<gert::TilingContext>();          \
  EXPECT_EQ(tiling_func(context), expect);                                          \
  for (uint8_t * tensor : const_tensors) {                                          \
    delete[] tensor;                                                                \
  }

#define TILING_DATA_VERIFY_BYTYPE(holder, datatype, expect_str)                     \
  gert::TilingData* tiling_data = reinterpret_cast<gert::TilingData*>(param.get()); \
  ASSERT_NE(tiling_data, nullptr);                                                  \
  EXPECT_EQ(to_string<datatype>(tiling_data->GetData(), tiling_data->GetDataSize()), expect_str)

#define TILING_DATA_VERIFY_CUSTOM(holder, trans_func, expect_str)                   \
  gert::TilingData* tiling_data = reinterpret_cast<gert::TilingData*>(param.get()); \
  ASSERT_NE(tiling_data, nullptr);                                                  \
  EXPECT_EQ(trans_func(tiling_data->GetData(), tiling_data->GetDataSize()), expect_str)
