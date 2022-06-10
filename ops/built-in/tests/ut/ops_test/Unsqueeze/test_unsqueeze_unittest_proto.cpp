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

class UnsqueezeUT : public testing::Test {
};

TEST_F(UnsqueezeUT, Unsqueeze_negative_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({-5})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 5);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 2);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(4), 5);
}

TEST_F(UnsqueezeUT, Unsqueeze_one_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 5);
}

TEST_F(UnsqueezeUT, Unsqueeze_two_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({0, 4})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 5);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 5);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(4), 1);
}

TEST_F(UnsqueezeUT, Unsqueeze_three_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4, 5})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 6);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 5);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(4), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(5), 1);
}

TEST_F(UnsqueezeUT, Unsqueeze_unsorted_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({5, 4, 2})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 6);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 5);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(4), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(5), 1);
}

TEST_F(UnsqueezeUT, Unsqueeze_empty_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 1);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(2), 4);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(3), 5);
}

TEST_F(UnsqueezeUT, Unsqueeze_out_of_range_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({4})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeUT, Unsqueeze_repetive_axes) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

  gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .InputShapes({&input_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeUT, Unsqueeze_larger_than_kMaxDimNums_axes) {
ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze"), nullptr);
auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Unsqueeze")->infer_shape;

gert::StorageShape input_shape = {{3, 4, 5, 1, 1, 1, 1}, {3, 4, 5, 1, 1, 1, 1}};
gert::StorageShape output_shape = {{}, {}};

auto holder = gert::InferShapeContextFaker()
    .NodeIoNum(1, 1)
    .IrInputNum(1)
    .InputShapes({&input_shape})
    .OutputShapes({&output_shape})
    .NodeAttrs({{"axes", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3, 4, 5})}})
    .Build();

EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
