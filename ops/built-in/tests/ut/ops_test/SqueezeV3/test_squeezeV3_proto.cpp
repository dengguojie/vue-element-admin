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

#include <gtest/gtest.h>

#include <climits>
#include <iostream>

#include "array_ops.h"
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "op_proto_test_util.h"
#include "util.h"

class SQUEEZEV3_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SQUEEZEV3_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SQUEEZEV3_UT TearDown" << std::endl;
  }
};

TEST_F(SQUEEZEV3_UT, CheckInputInvalid) {
  {
    ge::op::SqueezeV3 op("SqueezeV3");
    ge::graphStatus ret;

    // check axes invalid while there is no 1 in selected dims
    op.UpdateInputDesc("x", create_desc({5, 5, 5, 5, 1}, ge::DT_INT32));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(3 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[3] = {1, 2, 3};
    const_tensor.SetData((uint8_t*)const_data, 3 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }

  {
    // check axes out of range
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({5, 5, 5, 5, 1}, ge::DT_INT32));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(1 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[1] = {-8};
    const_tensor.SetData((uint8_t*)const_data, 1 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }
}

TEST_F(SQUEEZEV3_UT, InfershapeTestWithoutAxes) {
  ge::op::SqueezeV3 op("SqueezeV3");
  op.UpdateInputDesc("x", create_desc({1, 3, 2, 1, 5}, ge::DT_INT32));
  ge::graphStatus ret;

  ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // check datatype
  const auto output = op.GetOutputDesc("y");
  EXPECT_EQ(output.GetDataType(), ge::DT_INT32);

  // check squeeze without axes
  std::vector<int64_t> output_shape = {3, 2, 5};
  EXPECT_EQ(output.GetShape().GetDims(), output_shape);
}

TEST_F(SQUEEZEV3_UT, InfershapeTestWithAxes) {
  {
    // check squeeze case 1 with positive axes
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({1, 3, 2, 1, 5}, ge::DT_INT32));
    ge::graphStatus ret;
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(2 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[2] = {0, 3};
    const_tensor.SetData((uint8_t*)const_data, 2 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // check datatype
    const auto output = op.GetOutputDesc("y");
    EXPECT_EQ(output.GetDataType(), ge::DT_INT32);
    std::vector<int64_t> output_shape_1 = {3, 2, 5};
    EXPECT_EQ(output.GetShape().GetDims(), output_shape_1);
  }

  {
    // check squeeze case 2 eith negative axes
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({1, 3, 2, 1, 5}, ge::DT_INT32));
    ge::graphStatus ret;
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(1 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[1] = {-2};
    const_tensor.SetData((uint8_t*)const_data, sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    const auto output = op.GetOutputDesc("y");
    std::vector<int64_t> output_shape_2 = {1, 3, 2, 5};
    EXPECT_EQ(output.GetShape().GetDims(), output_shape_2);
  }

  {
    // check input x shape is {-2}
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_INT32));
    ge::graphStatus ret;
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(1 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[1] = {-2};
    const_tensor.SetData((uint8_t*)const_data, sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    const auto output = op.GetOutputDesc("y");
    std::vector<int64_t> output_shape = {-2};
    EXPECT_EQ(output.GetShape().GetDims(), output_shape);
  }
}

TEST_F(SQUEEZEV3_UT, InfershapeTestWithUnknownShape) {
  {
    // check x shape size not equal to range size
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({-1, 3, -1, 1, 5}, ge::DT_INT32));
    std::vector<std::pair<int64_t, int64_t>> x_range = {{1, 5}, {3, 3}, {1, 1}, {5, 5}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    op_desc->MutableInputDesc(0)->SetShapeRange(x_range);
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(1 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[3] = {0, 2, 3};
    const_tensor.SetData((uint8_t*)const_data, 3 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }

  {
    // check infershape with positive axes 
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({-1, 3, -1, 1, 5}, ge::DT_INT32));
    std::vector<std::pair<int64_t, int64_t>> x_range = {{1, 5}, {3, 3}, {1, 6}, {1, 1}, {5, 5}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    op_desc->MutableInputDesc(0)->SetShapeRange(x_range);
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(1 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[3] = {0, 2, 3};
    const_tensor.SetData((uint8_t*)const_data, 3 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    const auto output = op.GetOutputDesc("y");
    std::vector<int64_t> output_shape = {3, 5};
    EXPECT_EQ(output.GetShape().GetDims(), output_shape);

    std::vector<std::pair<int64_t, int64_t>> output_range = {{3, 3}, {5, 5}};
    std::vector<std::pair<int64_t, int64_t>> actul_output_range;
    (void)output.GetShapeRange(actul_output_range);
    EXPECT_EQ(actul_output_range.size(), output_range.size());

    for (size_t i = 0UL; i < actul_output_range.size(); ++i) {
      EXPECT_EQ(actul_output_range[i], output_range[i]);
    }
  }

  {
    // check infershape without axes 
    ge::op::SqueezeV3 op("SqueezeV3");
    op.UpdateInputDesc("x", create_desc({-1, 3, -1, 1, 5}, ge::DT_INT32));
    std::vector<std::pair<int64_t, int64_t>> x_range = {{1, 5}, {3, 3}, {1, 6}, {1, 1}, {5, 5}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    op_desc->MutableInputDesc(0)->SetShapeRange(x_range);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    const auto output = op.GetOutputDesc("y");
    std::vector<int64_t> output_shape = {-1, 3, -1, 5};
    EXPECT_EQ(output.GetShape().GetDims(), output_shape);

    std::vector<std::pair<int64_t, int64_t>> output_range = {{1, 5}, {3, 3}, {1, 6}, {5, 5}};
    std::vector<std::pair<int64_t, int64_t>> actul_output_range;
    (void)output.GetShapeRange(actul_output_range);
    EXPECT_EQ(actul_output_range.size(), output_range.size());

    for (size_t i = 0UL; i < actul_output_range.size(); ++i) {
      EXPECT_EQ(actul_output_range[i], output_range[i]);
    }
  }
}