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
using namespace std;

typedef std::string (*BufToString)(void *, size_t);
void TestTilingParse(const std::string optype, std::string json_str, void *compile_info);

template <typename T>
string to_string(void *buf, size_t size) {
  std::string result;
  const T *data = reinterpret_cast<const T *>(buf);
  size_t len = size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

#define TILING_PARSE_JSON_TO_COMPILEINFO(optype, json, compile_info) \
  TestTilingParse(optype, json, &compile_info)

#define ATTACH_OPERATOR_TO_HOLDER(holder, op, tiling_len, compile_info)   \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);            \
  ASSERT_NE(operator_info, nullptr);                                      \
  size_t input_size = operator_info->GetInputsSize();                     \
  size_t output_size = operator_info->GetOutputsSize();                   \
  auto param = gert::TilingData::CreateCap(tiling_len);                   \
  auto faker = gert::TilingContextFaker()                                 \
                    .NodeIoNum(input_size, output_size)                   \
                    .TilingData(param.get());                             \
  faker = faker.CompileInfo(&compile_info);                               \
  std::vector<gert::StorageShape> storage_shapes(input_size);             \
  std::vector<void *> input_shapes_ref(input_size);                       \
  if (input_size > 0) {                                                   \
    for (size_t i = 0; i < input_size; ++i) {                             \
      auto input_desc = operator_info->MutableInputDesc(i);               \
      ASSERT_NE(input_desc, nullptr);                                     \
      ge::Format input_format = input_desc->GetFormat();                  \
      ge::Format origin_format = input_desc->GetOriginFormat();           \
      ge::DataType dtype = input_desc->GetDataType();                     \
      faker = faker.NodeInputTd(i, dtype, input_format, input_format);    \
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {        \
        storage_shapes[i].MutableOriginShape().AppendDim(dim);            \
      }                                                                   \
      for (int64_t dim : input_desc->MutableShape().GetDims()) {          \
        storage_shapes[i].MutableStorageShape().AppendDim(dim);           \
      }                                                                   \
      input_shapes_ref[i] = &storage_shapes[i];                           \
    }                                                                     \
    faker = faker.InputShapes(input_shapes_ref);                          \
    std::vector<uint32_t> irnum(input_size, 1);                           \
    faker = faker.IrInstanceNum(irnum);                                   \
  }                                                                       \
  std::vector<gert::StorageShape> output_shapes(output_size);             \
  std::vector<void *> output_shapes_ref(output_size);                     \
  if (output_size > 0) {                                                  \
    for (size_t i = 0; i < output_size; ++i) {                            \
      auto output_desc = operator_info->MutableOutputDesc(i);             \
      ASSERT_NE(output_desc, nullptr);                                    \
      ge::Format input_format = output_desc->GetFormat();                 \
      ge::Format origin_format = output_desc->GetOriginFormat();          \
      ge::DataType dtype = output_desc->GetDataType();                    \
      faker = faker.NodeOutputTd(i, dtype, origin_format, input_format);  \
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {       \
        output_shapes[i].MutableOriginShape().AppendDim(dim);             \
      }                                                                   \
      for (int64_t dim : output_desc->MutableShape().GetDims()) {         \
        output_shapes[i].MutableStorageShape().AppendDim(dim);            \
      }                                                                   \
      output_shapes_ref[i] = &output_shapes[i];                           \
    }                                                                     \
    faker = faker.OutputShapes(output_shapes_ref);                        \
  }                                                                       \
  auto holder = faker.Build()

#define ATTACH_OPERATOR_TO_HOLDER_IRNUM(holder, op, irnum, tiling_len, compile_info)   \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                         \
  ASSERT_NE(operator_info, nullptr);                                                   \
  size_t input_size = operator_info->GetInputsSize();                                  \
  size_t output_size = operator_info->GetOutputsSize();                                \
  auto param = gert::TilingData::CreateCap(tiling_len);                                \
  auto faker = gert::TilingContextFaker()                                              \
                    .NodeIoNum(input_size, output_size)                                \
                    .TilingData(param.get());                                          \
  faker = faker.CompileInfo(&compile_info);                                            \
  std::vector<gert::StorageShape> storage_shapes(input_size);                          \
  std::vector<gert::StorageShape *> input_shapes_ref(input_size);                      \
  if (input_size > 0) {                                                                \
    for (size_t i = 0; i < input_size; ++i) {                                          \
      auto input_desc = operator_info->MutableInputDesc(i);                            \
      ASSERT_NE(input_desc, nullptr);                                                  \
      ge::Format input_format = input_desc->GetFormat();                               \
      ge::Format origin_format = input_desc->GetOriginFormat();                        \
      ge::DataType dtype = input_desc->GetDataType();                                  \
      faker = faker.NodeInputTd(i, dtype, origin_format, input_format);                \
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                     \
        storage_shapes[i].MutableOriginShape().AppendDim(dim);                         \
      }                                                                                \
      for (int64_t dim : input_desc->MutableShape().GetDims()) {                       \
        storage_shapes[i].MutableStorageShape().AppendDim(dim);                        \
      }                                                                                \
      input_shapes_ref[i] = &storage_shapes[i];                                        \
    }                                                                                  \
    faker = faker.InputShapes(input_shapes_ref);                                       \
    faker = faker.IrInstanceNum(irnum);                                                \
  }                                                                                    \
  std::vector<gert::StorageShape> output_shapes(output_size);                          \
  std::vector<gert::StorageShape *> output_shapes_ref(output_size);                    \
  if (output_size > 0) {                                                               \
    for (size_t i = 0; i < output_size; ++i) {                                         \
      auto output_desc = operator_info->MutableOutputDesc(i);                          \
      ASSERT_NE(output_desc, nullptr);                                                 \
      ge::Format input_format = output_desc->GetFormat();                              \
      ge::Format origin_format = output_desc->GetOriginFormat();                       \
      ge::DataType dtype = output_desc->GetDataType();                                 \
      faker = faker.NodeOutputTd(i, dtype, input_format, input_format);                \
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {                    \
        output_shapes[i].MutableOriginShape().AppendDim(dim);                          \
      }                                                                                \
      for (int64_t dim : output_desc->MutableShape().GetDims()) {                      \
        output_shapes[i].MutableStorageShape().AppendDim(dim);                         \
      }                                                                                \
      output_shapes_ref[i] = &output_shapes[i];                                        \
    }                                                                                  \
    faker = faker.OutputShapes(output_shapes_ref);                                     \
  }                                                                                    \
  auto holder = faker.Build()

#define HOLDER_DO_TILING(holder, optype, expect)                                        \
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(optype), nullptr);            \
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling;     \
  ASSERT_NE(tiling_func, nullptr);                                                      \
  gert::TilingContext *context = holder.GetContext<gert::TilingContext>();              \
  EXPECT_EQ(tiling_func(context), expect)

#define TILING_DATA_VERIFY_BYTYPE(holder, datatype, expect_str)                                    \
  gert::TilingData *tiling_data = reinterpret_cast<gert::TilingData *>(param.get());               \
  ASSERT_NE(tiling_data, nullptr);                                                                 \
  EXPECT_EQ(to_string<datatype>(tiling_data->GetData(), tiling_data->GetDataSize()), expect_str)

#define TILING_DATA_VERIFY_CUSTOM(holder, trans_func, expect_str)                                  \
  gert::TilingData *tiling_data = reinterpret_cast<gert::TilingData *>(param.get());               \
  ASSERT_NE(tiling_data, nullptr);                                                                 \
  EXPECT_EQ(trans_func(tiling_data->GetData(), tiling_data->GetDataSize()), expect_str)
