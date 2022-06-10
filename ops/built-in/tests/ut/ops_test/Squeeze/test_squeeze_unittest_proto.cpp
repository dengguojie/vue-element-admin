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

class SqueezeUT : public testing::Test {
};

TEST_F(SqueezeUT, Squeeze_negative_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-4})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_one_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_two_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{1, 3, 4, 5, 1}, {1, 3, 4, 5, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({0, 4})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_three_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4, 5})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_unsorted_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({5, 4, 2})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_empty_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_out_of_range_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1}, {3, 4, 5, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({5})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(SqueezeUT, Squeeze_repetive_axis) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
  gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({3, 3, 4})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_not_dim1_axis) {
ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze"), nullptr);
auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Squeeze")->infer_shape;
gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
gert::StorageShape output_shape = {{}, {}};

auto holder = gert::InferShapeContextFaker()
    .NodeIoNum(1, 1)
    .IrInputNum(1)
    .InputShapes({&input_shape})
    .OutputShapes({&output_shape})
    .NodeAttrs({{"axis", ge::AnyValue::CreateFrom<std::vector<int64_t>>({3, 1})}})
    .Build();

EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
