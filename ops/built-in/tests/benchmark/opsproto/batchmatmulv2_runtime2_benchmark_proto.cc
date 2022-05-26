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

#define protected public

#include <benchmark/benchmark.h>

#include <nlohmann/json.hpp>

#include "cache_tiling.h"
#include "cube_tiling_runtime.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"


using gert::InferShapeContext;
namespace gert {
ge::graphStatus InferShapeForBatchMatMulV2(InferShapeContext *context);
}
static void BatchMatMulV2InferShape_runtime2(benchmark::State &state) {
  gert::StorageShape x1_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{16, 64, 64}, {16, 4, 4, 16, 16}};
  gert::StorageShape bias_shape = {{64}, {64}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", ge::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  for (auto _ : state) {
    gert::InferShapeForBatchMatMulV2(holder.GetContext<InferShapeContext>());
  }

  // auto output = holder.GetContext<InferShapeContext>()->GetOutputShape(0);
  // for (size_t idx = 0; idx < output->GetDimNum(); ++idx) {
  //   std::cout << "BatchMatMulV2InferShape output " << output->GetDim(idx) << std::endl;
  // }

  // std::cout << "end" << std::endl;
}
BENCHMARK(BatchMatMulV2InferShape_runtime2);
