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
#include "runtime/tensor.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tensor_data.h"

namespace ops {
class ShapeUT : public testing::Test {};

TEST_F(ShapeUT, TensorShape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Shape")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape(), gert::Shape({4}));  
}

TEST_F(ShapeUT, VectorShape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Shape")->infer_shape;
  gert::StorageShape input_shape = {{5}, {5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape(), gert::Shape({1}));  
}

TEST_F(ShapeUT, ScalarShape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Shape")->infer_shape;
  gert::StorageShape input_shape = {{}, {}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape(), gert::Shape({0}));  
}

TEST_F(ShapeUT, EmptyInput) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Shape"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Shape")->infer_shape;
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({nullptr})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_NE(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
}  // namespace ops