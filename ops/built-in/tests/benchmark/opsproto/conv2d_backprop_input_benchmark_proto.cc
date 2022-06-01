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

#include <benchmark/benchmark.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "matrix_calculation_ops.h"
#include "register/op_impl_registry.h"

using namespace std;

namespace gert {
ge::graphStatus InferShapeForConv2DBackpropInput(gert::InferShapeContext *context);
}

static void Conv2DBpInputInferShape(benchmark::State &state) {
  vector<int64_t> strides({1, 3, 1, 1});
  vector<int64_t> pads({0, 0, 0, 0});
  vector<int64_t> dilations({1, 1, 1, 1});
  int64_t groups = 1;
  string data_format("NHWC");

  vector<int64_t> input_size = {28, 96, 96, 2};
  gert::StorageShape input_size_shape = {{28, 96, 96, 2}, {28, 96, 96, 2}};
  gert::StorageShape filter_shape = {{62, 2, 2, 2}, {62, 2, 2, 2}};
  gert::StorageShape out_backprop_shape = {{28, 32, 95, 62}, {28, 32, 95, 62}};
  gert::StorageShape output_shape = {{}, {}};

  size_t total_size = 0;
  auto tensor_holder =
      gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  tensor->MutableStorageShape() = input_size_shape.MutableStorageShape();
  tensor->MutableOriginShape() = input_size_shape.MutableOriginShape();
  tensor->SetOriginFormat(ge::FORMAT_NHWC);
  tensor->SetStorageFormat(ge::FORMAT_NHWC);
  (void)memcpy(tensor->GetData<uint8_t>(), input_size.data(), input_size.size() * sizeof(int64_t));

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({tensor, &filter_shape, &out_backprop_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                {"groups", ge::AnyValue::CreateFrom<int64_t>(groups)},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>(data_format)}})
                    .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                    .Build();

  for (auto _ : state) {
    gert::InferShapeForConv2DBackpropInput(holder.GetContext<gert::InferShapeContext>());
  }
}

BENCHMARK(Conv2DBpInputInferShape);