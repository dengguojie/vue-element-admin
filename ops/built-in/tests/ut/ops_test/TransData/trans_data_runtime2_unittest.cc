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
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"

namespace gert_test {
class TransDataImplUT : public testing::Test {};
TEST_F(TransDataImplUT, TransDataInferShapeOk) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("TransData"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("TransData")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape input_shape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 8);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 224);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 224);
}
}
