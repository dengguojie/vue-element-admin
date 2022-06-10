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
#include "register/op_impl_registry.h"

namespace gert {
    ge::graphStatus InferShapeForConv2D(InferShapeContext* context);
}

static void Conv2DInferShapeBenchmarkTest(benchmark::State& state)
{
    gert::StorageShape xShape = {{3, 90, 100, 78}, {}};
    gert::StorageShape wShape = {{66, 30, 5, 5}, {}};
    gert::StorageShape biasShape = {{1, 1, 1, 66}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
        .NodeAttrs({
            {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 4})},
            {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
            {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
            {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
            {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")}
            })
        .InputShapes({&xShape, &wShape, &biasShape})
        .OutputShapes({&yShape})
        .Build();

    for (auto _ : state) {
        gert::InferShapeForConv2D(holder.GetContext<gert::InferShapeContext>());
    }
}

BENCHMARK(Conv2DInferShapeBenchmarkTest);