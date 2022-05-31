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
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"

#include <iostream>
#include <vector>

#define private public
#include "register/op_tiling_registry.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;

void TestTilingParse(const std::string optype, std::string json_str, void *compile_info) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(optype), nullptr);
  auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling_parse;
  ASSERT_NE(tiling_prepare_func, nullptr);
  char *js_buf = new char[1 + json_str.length()];
  ASSERT_NE(js_buf, nullptr);
  strcpy(js_buf, json_str.c_str());
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({js_buf})
                    .Outputs({compile_info})
                    .Build();
  ASSERT_EQ(tiling_prepare_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  delete[] js_buf;
}
typedef std::string (*BufToString)(void *, size_t);

string int32_to_string(void *buf, size_t size) {
  std::string result;
  const int32_t *data = reinterpret_cast<const int32_t *>(buf);
  size_t len = size / sizeof(int32_t);
  cout << "int32_to_string len=" << len << endl;
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

string int64_to_string(void *buf, size_t size) {
  std::string result;
  const int64_t *data = reinterpret_cast<const int64_t *>(buf);
  size_t len = size / sizeof(int64_t);
  cout << "int64_to_string len=" << len << endl;
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

void CommonTilingOperator(ge::Operator& op, std::string& compile_info, void *info_base,
                          size_t tiling_len, const char *expect_tiling_data) {
  auto operator_info = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(operator_info, nullptr);
  std::string optype = op.GetOpType();
  size_t input_size = op.GetInputsSize();
  size_t output_size = op.GetOutputsSize();

  // tiling data
  auto param = gert::TilingData::CreateCap(tiling_len);
  auto faker = gert::TilingContextFaker()
                    .NodeIoNum(input_size, output_size)
                    .TilingData(param.get());
  if (info_base != nullptr) {
    TestTilingParse(optype, compile_info, info_base);
    faker = faker.CompileInfo(info_base);
  }
  std::vector<gert::StorageShape> input_shapes(input_size);
  std::vector<gert::StorageShape *> input_shapes_ref(input_size);
  if (input_size > 0) {
    for (size_t i = 0; i < input_size; ++i) {
      auto input_desc = operator_info->MutableInputDesc(i);
      ASSERT_NE(input_desc, nullptr);
      // get and check input format and shape
      ge::Format input_format = input_desc->GetFormat();
      ge::Format origin_format = input_desc->GetOriginFormat();
      ge::DataType dtype = input_desc->GetDataType();
      faker = faker.NodeInputTd(i, dtype, input_format, input_format);
      for (int64_t dim : input_desc->GetOriginShape().GetDims()) {
        input_shapes[i].MutableOriginShape().AppendDim(dim);
      }
      for (int64_t dim : input_desc->MutableShape().GetDims()) {
        input_shapes[i].MutableStorageShape().AppendDim(dim);
      }
      input_shapes_ref[i] = &input_shapes[i];
    }
    faker = faker.InputShapes(input_shapes_ref);
    std::vector<uint32_t> irnum(input_size, 1);
    faker = faker.IrInstanceNum(irnum);
  }
  std::vector<gert::StorageShape> output_shapes(output_size);
  std::vector<gert::StorageShape *> output_shapes_ref(output_size);
  if (output_size > 0) {
    for (size_t i = 0; i < output_size; ++i) {
      auto output_desc = operator_info->MutableOutputDesc(i);
      ASSERT_NE(output_desc, nullptr);
      // get and check input format and shape
      ge::Format input_format = output_desc->GetFormat();
      ge::Format origin_format = output_desc->GetOriginFormat();
      ge::DataType dtype = output_desc->GetDataType();
      faker = faker.NodeOutputTd(i, dtype, input_format, input_format);
      for (int64_t dim : output_desc->GetOriginShape().GetDims()) {
        output_shapes[i].MutableOriginShape().AppendDim(dim);
      }
      for (int64_t dim : output_desc->MutableShape().GetDims()) {
        output_shapes[i].MutableStorageShape().AppendDim(dim);
      }
      output_shapes_ref[i] = &output_shapes[i];
    }
    faker = faker.OutputShapes(output_shapes_ref);
  }
  auto holder = faker.Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(optype), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(optype)->tiling;
  ASSERT_NE(tiling_func, nullptr);
  gert::TilingContext *context = holder.GetContext<gert::TilingContext>();
  EXPECT_EQ(tiling_func(context), ge::GRAPH_SUCCESS);
  if (expect_tiling_data == nullptr) return;
  gert::TilingData *tiling_data = reinterpret_cast<gert::TilingData *>(param.get());
  ASSERT_NE(tiling_data, nullptr);
  EXPECT_EQ(int64_to_string(tiling_data->GetData(), tiling_data->GetDataSize()), expect_tiling_data);
}
