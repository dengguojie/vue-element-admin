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
#include "op_proto_test_util.h"

namespace ops {
class RangeUT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "range SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "range TearDown" << std::endl;
  }

  template <typename T>
  gert::Tensor * ConstructInputConstTensor(const T &const_value, ge::DataType const_dtype) {
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T)]);
    auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
    gert::Tensor tensor({{1}, {1}},                                 // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}},        // format
                        gert::kFollowing,                          // placement
                        const_dtype,                               //dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
    *tensor_data = const_value;
    std::cout<<" const_value:" << *tensor_data<< std::endl;
    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor;
  }
};

TEST_F(RangeUT, RangeCaseint32) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Range"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Range")->infer_shape;
  int32_t start = 1;
  int32_t limit = 10;
  int32_t delta = 1;
  auto start_tensor = ConstructInputConstTensor(start, ge::DT_INT32);
  auto limit_tensor = ConstructInputConstTensor(limit, ge::DT_INT32);
  auto delta_tensor = ConstructInputConstTensor(delta, ge::DT_INT32);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, limit_tensor, delta_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 1);
}

TEST_F(RangeUT, RangeCaseint64) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Range"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Range")->infer_shape;
  // input1:start
  int64_t start = 2;
  int64_t limit = 20;
  int64_t delta = 1;
  auto start_tensor = ConstructInputConstTensor(start, ge::DT_INT64);
  auto limit_tensor = ConstructInputConstTensor(limit, ge::DT_INT64);
  auto delta_tensor = ConstructInputConstTensor(delta, ge::DT_INT64);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, limit_tensor, delta_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 1);
}

TEST_F(RangeUT, RangeCasedouble) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Range"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Range")->infer_shape;
  // input1:start
  double start = 1.0;
  double limit = 10.0;
  double delta = 1.0;
  auto start_tensor = ConstructInputConstTensor(start, ge::DT_DOUBLE);
  auto limit_tensor = ConstructInputConstTensor(limit, ge::DT_DOUBLE);
  auto delta_tensor = ConstructInputConstTensor(delta, ge::DT_DOUBLE);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, limit_tensor, delta_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 1);
}

TEST_F(RangeUT, RangeCaseDeltaZreo) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Range"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Range")->infer_shape;
  int32_t start = 1;
  int32_t limit = 10;
  int32_t delta = 0;
  auto start_tensor = ConstructInputConstTensor(start, ge::DT_INT32);
  auto limit_tensor = ConstructInputConstTensor(limit, ge::DT_INT32);
  auto delta_tensor = ConstructInputConstTensor(delta, ge::DT_INT32);

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .InputShapes({start_tensor, limit_tensor, delta_tensor})
                    .OutputShapes({&output_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
}  // namespace ops