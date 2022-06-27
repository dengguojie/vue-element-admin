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
 * \file ut_op_common.h
 */
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/kernel_context.h"

#include <iostream>
#include <vector>

#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "reduce_infer_util.h"
#include "op_util.h"

using namespace std;
using namespace ge;

struct Runtime2TestParam {
  std::vector<std::string> attrs;
  std::vector<bool> input_const;
  std::vector<uint32_t> irnum;
};

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

#define HOLDER_DO_INFER_SHAPE(holder, optype, expect)                                          \
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(optype), nullptr);                   \
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->infer_shape;  \
  ASSERT_NE(infer_shape_func, nullptr);                                                        \
  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), expect);           \
  for(uint8_t* tensor : const_tensors) { delete []tensor; }

#define VERIFY_OUTPUT_SHAPE(holder, index, expect_dims)                                        \
   ASSERT_NE(holder.GetContext<gert::InferShapeContext>(), nullptr);                           \
   auto out_shape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(index);       \
   ASSERT_NE(out_shape, nullptr);                                                              \
   ASSERT_EQ(ops::ToString(*out_shape), ToString(expect_dims))

#define ADD_ATTR_CASE(vttype, datetype)                              \
  case ge::AnyValue::ValueType::vttype: {                            \
    datetype value;                                                  \
    ASSERT_EQ(op.GetAttr(item, value), GRAPH_SUCCESS);               \
    p.second = ge::AnyValue::CreateFrom<datetype>(value);            \
  }                                                                  \
  break

#define ATTACH_INPUTS_TO_FAKER(faker, op, input_size, input_const)                  \
  std::vector<gert::StorageShape> input_shapes(input_size);                         \
  std::vector<void *> input_shapes_ref(input_size);                                 \
  std::vector<uint32_t> irnum(input_size, 0);                                       \
  uint8_t* input_tensor_holder;                                                     \
  if (input_size > 0) {                                                             \
    auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                    \
    ASSERT_NE(operator_info, nullptr);                                              \
    size_t count = 0;                                                               \
    for (size_t i = 0; i < input_const.size(); ++i) {                               \
      if (input_const[i]) {                                                         \
        input_tensor_holder = GetConstTensor(op, i);                                \
        if (input_tensor_holder == nullptr) continue;                               \
        input_shapes_ref[count] = input_tensor_holder;                              \
        const_tensors.push_back(input_tensor_holder);                               \
      } else {                                                                      \
        auto input_desc = operator_info->MutableInputDesc(i);                       \
        if (input_desc == nullptr) continue;                                        \
        ge::Format input_format = input_desc->GetFormat();                          \
        ge::Format origin_format = input_desc->GetOriginFormat();                   \
        ge::DataType dtype = input_desc->GetDataType();                             \
        faker = faker.NodeInputTd(count, dtype, input_format, input_format);        \
        for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                \
          input_shapes[count].MutableOriginShape().AppendDim(dim);                  \
        }                                                                           \
        for (int64_t dim : input_desc->MutableShape().GetDims()) {                  \
          input_shapes[count].MutableStorageShape().AppendDim(dim);                 \
        }                                                                           \
        input_shapes_ref[count] = &input_shapes[count];                             \
      }                                                                             \
      irnum[i] = 1;                                                                 \
      count++;                                                                      \
    }                                                                               \
    faker = faker.InputShapes(input_shapes_ref);                                    \
    faker = faker.IrInstanceNum(irnum);                                             \
  }

#define ATTACH_INPUTS_TO_FAKER_IRNUM(faker, op, input_size, irnum)                  \
  std::vector<gert::StorageShape> input_shapes(input_size);                         \
  std::vector<void *> input_shapes_ref(input_size);                                 \
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);                      \
  ASSERT_NE(operator_info, nullptr);                                                \
  size_t count = 0;                                                                 \
  for (size_t idx = 0; idx < irnum.size(); idx++) {                                 \
    for (size_t i = 0; i < irnum[idx]; ++i) {                                       \
      auto input_desc = operator_info->MutableInputDesc(i + idx);                   \
      ASSERT_NE(input_desc, nullptr);                                               \
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {                  \
        input_shapes[count].MutableOriginShape().AppendDim(dim);                    \
      }                                                                             \
      for (int64_t dim : input_desc->MutableShape().GetDims()) {                    \
        input_shapes[count].MutableStorageShape().AppendDim(dim);                   \
      }                                                                             \
      input_shapes_ref[count] = &input_shapes[count];                               \
      count++;                                                                      \
    }                                                                               \
    faker = faker.InputShapes(input_shapes_ref);                                    \
    faker = faker.IrInstanceNum(irnum);                                             \
  }

#define ATTACH_OUTPUTS_TO_FAKER(faker, output_size)                                 \
  std::vector<gert::StorageShape> output_shapes(output_size);                       \
  std::vector<void *> output_shapes_ref(output_size);                               \
  if (output_size > 0) {                                                            \
    for (size_t i = 0; i < output_size; ++i) {                                      \
      output_shapes_ref[i] = &output_shapes[i];                                     \
    }                                                                               \
    faker = faker.OutputShapes(output_shapes_ref);                                  \
  }

#define ATTACH_ATTRS_TO_FAKER(faker, op, attrs_name)                                \
  auto op_attrs_map = op.GetAllAttrNamesAndTypes();                                 \
  std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value;                  \
  std::pair<std::string, ge::AnyValue> p;                                           \
  if (attrs_name.size() > 0) {                                                      \
    for (auto item : attrs_name) {                                                  \
      p.first = item;                                                               \
      auto attr_it = op_attrs_map.find(item);                                       \
      if (attr_it != op_attrs_map.end()) {                                          \
        auto type_it = kAttrTypesMap.find(attr_it->second);                         \
        if (type_it != kAttrTypesMap.end()) {                                       \
          switch (type_it->second) {                                                \
            ADD_ATTR_CASE(VT_BOOL, bool);                                           \
            ADD_ATTR_CASE(VT_INT, int64_t);                                         \
            ADD_ATTR_CASE(VT_FLOAT, float32_t);                                     \
            ADD_ATTR_CASE(VT_STRING, std::string);                                  \
            ADD_ATTR_CASE(VT_LIST_INT, std::vector<int64_t>);                       \
            ADD_ATTR_CASE(VT_LIST_BOOL, std::vector<bool>);                         \
            ADD_ATTR_CASE(VT_LIST_LIST_INT, std::vector<std::vector<int64_t>>);     \
          }                                                                         \
        }                                                                           \
      }                                                                             \
      keys_to_value.push_back(p);                                                   \
    }                                                                               \
    faker = faker.NodeAttrs(keys_to_value);                                         \
  }

#define ATTACH_OPERATOR_TO_HOLDER(holder, op, attrs_name)                   \
  size_t input_size = op.GetInputsSize();                                   \
  size_t output_size = op.GetOutputsSize();                                 \
  auto faker = gert::InferShapeContextFaker()                               \
                    .NodeIoNum(input_size, output_size);                    \
  vector<bool> input_const(input_size, false);                              \
  vector<uint8_t*> const_tensors;                                           \
  ATTACH_INPUTS_TO_FAKER(faker, op, input_size, input_const);               \
  ATTACH_OUTPUTS_TO_FAKER(faker, output_size);                              \
  ATTACH_ATTRS_TO_FAKER(faker, op, attrs_name);                             \
  auto holder = faker.Build();

#define ATTACH_OPERATOR_TO_HOLDER_WITH_CONST(holder, op, input_const, attrs_name)     \
  size_t input_size = op.GetInputsSize();                                             \
  size_t output_size = op.GetOutputsSize();                                           \
  auto faker = gert::InferShapeContextFaker()                                         \
                    .NodeIoNum(input_size, output_size);                              \
  vector<uint8_t*> const_tensors;                                                     \
  ATTACH_INPUTS_TO_FAKER(faker, op, input_size, input_const);                         \
  ATTACH_OUTPUTS_TO_FAKER(faker, output_size);                                        \
  ATTACH_ATTRS_TO_FAKER(faker, op, attrs_name);                                       \
  auto holder = faker.Build();

#define ATTACH_OPERATOR_TO_HOLDER_WITH_IRNUM(holder, op, irnum, attrs_name)           \
  size_t input_size = op.GetInputsSize();                                             \
  size_t output_size = op.GetOutputsSize();                                           \
  auto faker = gert::InferShapeContextFaker()                                         \
                    .NodeIoNum(input_size, output_size);                              \
  vector<uint8_t*> const_tensors;                                                     \
  ATTACH_INPUTS_TO_FAKER_IRNUM(faker, op, input_size, irnum);                         \
  ATTACH_OUTPUTS_TO_FAKER(faker, output_size);                                        \
  ATTACH_ATTRS_TO_FAKER(faker, op, attrs_name);                                       \
  auto holder = faker.Build();

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
  std::cout << "const_dtype=" << const_dtype << std::endl;
  uint8_t* input_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
  auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder);
  std::memcpy(input_tensor + 1, data, size);
  int64_t value_size = (const_dtype == ge::DT_INT64 || const_dtype == ge::DT_UINT64) ? 
                      size / sizeof(int64_t) : size / sizeof(int32_t);
  gert::Tensor tensor({{value_size}, {value_size}},              // shape
                      {ge::FORMAT_ND, ge::FORMAT_ND, {}},        // format
                      gert::kFollowing,                          // placement
                      const_dtype,                               //dt
                      nullptr);
  std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
  return input_tensor_holder;
}

void CommonInferShapeOperator(ge::Operator& op, vector<string> attrs, vector<vector<int64_t>> expect_shapes);
void CommonInferShapeOperatorFail(ge::Operator& op, vector<string> attrs);

void CommonInferShapeOperatorWithConst(ge::Operator& op, vector<bool> input_const,
                                       vector<string> attrs, vector<vector<int64_t>> expect_shapes);
void CommonInferShapeOperatorWithConstFail(ge::Operator& op, vector<bool> input_const,
                                           vector<string> attrs);

void CommonInferShapeOperatorWithIrNum(ge::Operator& op, vector<uint32_t> irnum,
                                       vector<string> attrs, vector<vector<int64_t>> expect_shapes);
void CommonInferShapeOperatorWithIrNumFail(ge::Operator& op, vector<uint32_t> irnum,
                                           vector<string> attrs);

ge::graphStatus InferShapeTest(ge::Operator& op, const Runtime2TestParam& param);

ge::graphStatus InferShapeTest(ge::Operator& op);
