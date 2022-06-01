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

namespace gert_test {
class MaxPoolV3UT : public testing::Test {};
TEST_F(MaxPoolV3UT, InferShapeOk) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};

  std::vector<gert::StorageShape> output_shapes(1);
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&x_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"ksize", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 3})},
                                {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
                                {"padding_mode", ge::AnyValue::CreateFrom<std::string>("CALCULATED")},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"global_pooling", ge::AnyValue::CreateFrom<bool>(false)},
                                {"ceil_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV3UT, InferShapeFail) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};

  std::vector<gert::StorageShape> output_shapes(1);
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&x_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"ksize", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 3})},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"ceil_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3UT, InferShapeAttrs) {
using namespace gert;
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};
  StorageShape o1;
  auto holder = InferShapeContextFaker()
                    .IrInputNum(1)
                    .InputShapes({&x_shape})
                    .OutputShapes({&o1})
                    .NodeIoNum(1, 1)
                    .NodeAttrs({{"ksize", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 3})},
                                {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
                                {"padding_mode", ge::AnyValue::CreateFrom<std::string>("CALCULATED")},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"global_pooling", ge::AnyValue::CreateFrom<bool>(false)},
                                {"ceil_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  auto context = holder.GetContext<InferShapeContext>();

  auto attrs = context->GetAttrs();
  ASSERT_NE(attrs, nullptr);
  ASSERT_EQ(attrs->GetAttrNum(), 7);

  auto attr2 = attrs->GetAttrPointer<char>(2);
  ASSERT_NE(attr2, nullptr);
  EXPECT_STREQ(attr2, "CALCULATED");

  auto attr4 = attrs->GetAttrPointer<char>(4);
  ASSERT_NE(attr4, nullptr);
  EXPECT_STREQ(attr4, "NCHW");

  auto attr5 = attrs->GetAttrPointer<bool>(5);
  ASSERT_NE(attr5, nullptr);
  EXPECT_FALSE(*attr5);

  auto attr6 = attrs->GetAttrPointer<float>(6);
  ASSERT_NE(attr6, nullptr);
  EXPECT_FALSE(*attr6);

  EXPECT_EQ(attrs->GetAttrPointer<bool>(7), nullptr);
  EXPECT_EQ(infer_shape_func(holder.GetContext<InferShapeContext>()), ge::GRAPH_SUCCESS);
}
}  // namespace gert_test
