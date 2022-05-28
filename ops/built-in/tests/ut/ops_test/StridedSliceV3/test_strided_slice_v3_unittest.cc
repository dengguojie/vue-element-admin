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

#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"

namespace gert_test{
class StridedSliceV3UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice_v3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice_v3 TearDown" << std::endl;
  }

  template <typename T>
  std::unique_ptr<uint8_t[]> ConstructConstTensor(const std::vector<T> &const_value) {
    auto const_dtype = ge::DT_INT32;
    if (sizeof(T) == sizeof(int64_t)) {
      const_dtype = ge::DT_INT64;
    }

    int64_t value_size = const_value.size();
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(T) * value_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
    *input_tensor = {
        {{value_size}, {value_size}},        // storage shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}},  // storage format
        gert::kFollowing,                    // placement
        const_dtype,                         // data type
        0,                                   // address
    };
    auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
    for (size_t j = 0; j < value_size; j++) {
      tensor_data[j] = const_value[j];
    }
    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor_holder;
  }

  template <typename T>
  gert::InferShapeContextFaker ConstructContextFaker(gert::StorageShape &x_shape, gert::StorageShape &y_shape,
                                                     const std::vector<T> &begin_value,
                                                     const std::vector<T> &end_value, const std::vector<T> &axes_value,
                                                     const std::vector<T> &stride_value) {
    std::unique_ptr<uint8_t[]> begin_tensor_holder = ConstructConstTensor(begin_value);
    std::unique_ptr<uint8_t[]> end_tensor_holder = ConstructConstTensor(end_value);
    std::unique_ptr<uint8_t[]> axes_tensor_holder = ConstructConstTensor(axes_value);
    std::unique_ptr<uint8_t[]> stride_tensor_holder = ConstructConstTensor(stride_value);

    auto begin_tensor = reinterpret_cast<gert::Tensor *>(begin_tensor_holder.get());
    auto end_tensor = reinterpret_cast<gert::Tensor *>(end_tensor_holder.get());
    auto axes_tensor = reinterpret_cast<gert::Tensor *>(axes_tensor_holder.get());
    auto stride_tensor = reinterpret_cast<gert::Tensor *>(stride_tensor_holder.get());

    std::vector<uint32_t> instance_num = {1, 1, 1, 1, 1};
    if (axes_value.empty()) {
      instance_num[3] = 0;
    }
    if (stride_value.empty()) {
      instance_num[4] = 0;
    }
    auto faker = gert::InferShapeContextFaker()
                     .NodeIoNum(5, 1)
                     .IrInstanceNum(instance_num)
                     .InputShapes({&x_shape, begin_tensor, end_tensor, axes_tensor, stride_tensor})
                     .OutputShapes({&y_shape});
    return faker;
  }
};

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_normal) {
  gert::StorageShape x_shape = {{9, 10, 11, 12}, {9, 10, 11, 12}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 0};
  std::vector<int32_t> end_value = {8, 7};
  std::vector<int32_t> axes_value = {2, 3};
  std::vector<int32_t> stride_value = {1, 1};
  std::vector<int64_t> expected_output_shape = {9, 10, 8, 7};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_neg_axes) {
  gert::StorageShape x_shape = {{9, 10, 11, 12}, {9, 10, 11, 12}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 0};
  std::vector<int32_t> end_value = {8, 7};
  std::vector<int32_t> axes_value = {-2, -1};
  std::vector<int32_t> stride_value = {1, 1};
  std::vector<int64_t> expected_output_shape = {9, 10, 8, 7};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_neg_ends) {
  gert::StorageShape x_shape = {{9, 10, 11, 12}, {9, 10, 11, 12}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 0};
  std::vector<int32_t> end_value = {-2, -1};
  std::vector<int32_t> axes_value = {-2, -1};
  std::vector<int32_t> stride_value = {2, 3};
  std::vector<int64_t> expected_output_shape = {9, 10, 5, 4};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_ends_out_of_range) {
  gert::StorageShape x_shape = {{20, 10, 5}, {20, 10, 5}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 1};
  std::vector<int32_t> end_value = {1000, 1000};
  std::vector<int32_t> axes_value = {0, 1};
  std::vector<int32_t> stride_value = {1, 1};
  std::vector<int64_t> expected_output_shape = {20, 9, 5};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_empty_strides) {
  gert::StorageShape x_shape = {{20, 10, 5}, {20, 10, 5}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 0, 3};
  std::vector<int32_t> end_value = {20, 10, 4};
  std::vector<int32_t> axes_value = {0, 1, 2};
  std::vector<int32_t> stride_value = {};
  std::vector<int64_t> expected_output_shape = {20, 10, 1};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_empty_axes) {
  gert::StorageShape x_shape = {{20, 10, 5}, {20, 10, 5}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 0, 3};
  std::vector<int32_t> end_value = {20, 10, 4};
  std::vector<int32_t> axes_value = {};
  std::vector<int32_t> stride_value = {2, 3, 4};
  std::vector<int64_t> expected_output_shape = {10, 4, 1};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_empty_axes_and_strides) {
  gert::StorageShape x_shape = {{20, 10, 5}, {20, 10, 5}};
  gert::StorageShape y_shape = {{}, {}};
  std::vector<int32_t> begin_value = {0, 0, 0};
  std::vector<int32_t> end_value = {10,10,10};
  std::vector<int32_t> axes_value = {};
  std::vector<int32_t> stride_value = {};
  std::vector<int64_t> expected_output_shape = {10, 10, 5};

  auto faker = ConstructContextFaker<int32_t>(x_shape, y_shape, begin_value, end_value, axes_value, stride_value);
  auto holder = faker.Build();
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}
}  // namespace gert_test