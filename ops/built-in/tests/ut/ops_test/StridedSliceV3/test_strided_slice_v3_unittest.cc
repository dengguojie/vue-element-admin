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
    gert::Tensor tensor({{value_size}, {value_size}},              // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}},        // format
                        gert::kFollowing,                          // placement
                        const_dtype,                               //dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T *>(input_tensor + 1);
    for (size_t j = 0; j < value_size; j++) {
      tensor_data[j] = const_value[j];
    }
    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor_holder;
  }

  template <typename T>
  void ConstructContextFaker(gert::StorageShape &x_shape, gert::StorageShape &y_shape,
                             std::map<std::string, std::vector<T>> &value_dict) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("StridedSliceV3")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    std::vector<T>& axes_value = value_dict["axis"];
    std::vector<T>& stride_value = value_dict["stride"];
    std::unique_ptr<uint8_t[]> begin_tensor_holder = ConstructConstTensor(value_dict["begin"]);
    std::unique_ptr<uint8_t[]> end_tensor_holder = ConstructConstTensor(value_dict["end"]);
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
    auto holder = gert::InferShapeContextFaker()
                     .NodeIoNum(5, 1)
                     .IrInstanceNum(instance_num)
                     .InputShapes({&x_shape, begin_tensor, end_tensor, axes_tensor, stride_tensor})
                     .OutputShapes({&y_shape})
                     .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  }
};

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_normal) {
  gert::StorageShape x_shape = {{9, 10, 11, 12}, {9, 10, 11, 12}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {2, 3}},
      {"end", {8, 7}},
      {"axis", {2, 3}},
      {"stride", {1, 1}}};
  std::vector<int64_t> expected_output_shape = {9, 10, 6, 4};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_neg_axes) {
  gert::StorageShape x_shape = {{9, 10, 11, 12}, {9, 10, 11, 12}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0}},
      {"end", {8, 7}},
      {"axis", {-2, -1}},
      {"stride", {1, 1}}};
  std::vector<int64_t> expected_output_shape = {9, 10, 8, 7};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_neg_ends) {
  gert::StorageShape x_shape = {{9, 10, 11, 12}, {9, 10, 11, 12}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0}},
      {"end", {-2, -1}},
      {"axis", {-2, -1}},
      {"stride", {2, 3}}};
  std::vector<int64_t> expected_output_shape = {9, 10, 5, 4};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_ends_out_of_range) {
  gert::StorageShape x_shape = {{20, 10, 5}, {20, 10, 5}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 1}},
      {"end", {1000, 1000}},
      {"axis", {0, 1}},
      {"stride", {1, 1}}};
  std::vector<int64_t> expected_output_shape = {20, 9, 5};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_empty_strides) {
  gert::StorageShape x_shape = {{20, 10, 5, 100}, {20, 10, 5, 100}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0, 3}},
      {"end", {20, 10, 4}},
      {"axis", {0, 1, 2}},
      {"stride", {}}};
  std::vector<int64_t> expected_output_shape = {20, 10, 1, 100};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_empty_axes) {
  gert::StorageShape x_shape = {{20, 10, 5, 100}, {20, 10, 5, 100}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0, 3}},
      {"end", {20, 10, 4}},
      {"axis", {}},
      {"stride", {2, 3, 4}}};
  std::vector<int64_t> expected_output_shape = {10, 4, 1, 100};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}

TEST_F(StridedSliceV3UT, strided_slice_v3_infer_shape_empty_axes_and_strides) {
  gert::StorageShape x_shape = {{20, 10, 5}, {20, 10, 5}};
  gert::StorageShape y_shape = {{}, {}};
  std::map<std::string, std::vector<int32_t>> value_dict = {
      {"begin", {0, 0, 0}},
      {"end", {10,10,10}},
      {"axis", {}},
      {"stride", {}}};
  std::vector<int64_t> expected_output_shape = {10, 10, 5};

  ConstructContextFaker<int32_t>(x_shape, y_shape, value_dict);
  EXPECT_EQ(y_shape.GetOriginShape().GetDimNum(), expected_output_shape.size());
  for (int i = 0; i < expected_output_shape.size(); i++) {
    EXPECT_EQ(y_shape.GetOriginShape().GetDim(i), expected_output_shape[i]);
  }
}
}  // namespace gert_test