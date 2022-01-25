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
#include <iostream>
#include <climits>
#include "op_proto_test_util.h"
#include "util.h"
#include "graph/utils/op_desc_utils.h"
#include "array_ops.h"
#include "graph/ge_tensor.h"

class UNSQUEEZEV3_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UNSQUEEZEV3_UT SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UNSQUEEZEV3_UT TearDown" << std::endl;
  }
};


TEST_F(UNSQUEEZEV3_UT, CheckInputInvalid) {
  {
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    ge::graphStatus ret;

    // check axes invalid while there is duplicate data
    op.UpdateInputDesc("x", create_desc({5, 5, 5, 5, 1}, ge::DT_INT32));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(3 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[3] = {1, 2, 2};
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
    // check axes invalid while out of range
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    ge::graphStatus ret;
    op.UpdateInputDesc("x", create_desc({5, 5, 5, 5, 1}, ge::DT_INT32));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(3 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[3] = {-2, 9, 2};
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
    // check axes invalid while out of range. boundary test
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    ge::graphStatus ret;
    op.UpdateInputDesc("x", create_desc({5, 5, 5}, ge::DT_INT32));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(1 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[1] = {4};
    const_tensor.SetData((uint8_t*)const_data, 1 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
  }
}


TEST_F(UNSQUEEZEV3_UT, InfershapeTest) {
  {
    // check valid infershape with positive axes
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    ge::graphStatus ret;
    op.UpdateInputDesc("x", create_desc({5, 5, 5, 5}, ge::DT_INT32));
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
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    const auto output = op.GetOutputDesc("y");
    EXPECT_EQ(output.GetDataType(), ge::DT_INT32);

    std::vector<int64_t> expect_output_shape = {5, 1, 1, 1, 5, 5, 5};
    EXPECT_EQ(output.GetShape().GetDims(), expect_output_shape);
  }

  {
    // check valid infershape with nagative axes
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    ge::graphStatus ret;
    op.UpdateInputDesc("x", create_desc({5, 5, 5, 5}, ge::DT_INT64));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(2 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[2] = {-5, 2};
    const_tensor.SetData((uint8_t*)const_data, 2 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    const auto output = op.GetOutputDesc("y");
    EXPECT_EQ(output.GetDataType(), ge::DT_INT64);

    std::vector<int64_t> expect_output_shape = {5, 1, 1, 5, 5, 5};
    EXPECT_EQ(output.GetShape().GetDims(), expect_output_shape);
  }

  {
    // check valid infershape with unknown rank x
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    ge::graphStatus ret;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_INT64));
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(2 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[2] = {-5, 2};
    const_tensor.SetData((uint8_t*)const_data, 2 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    const auto output = op.GetOutputDesc("y");
    EXPECT_EQ(output.GetDataType(), ge::DT_INT64);

    std::vector<int64_t> expect_output_shape = {-2};
    EXPECT_EQ(output.GetShape().GetDims(), expect_output_shape);
  }

  {
    // check x shape size not equal to range size
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
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
    // check shape range of unknown shape after infershape
    ge::op::UnsqueezeV3 op("UnsqueezeV3");
    op.UpdateInputDesc("x", create_desc({-1, 3, -1, 1, 5}, ge::DT_INT32));
    std::vector<std::pair<int64_t, int64_t>> x_range = {{1, 5}, {3, 3}, {1, 6}, {1, 1}, {5, 5}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    op_desc->MutableInputDesc(0)->SetShapeRange(x_range);
    ge::Tensor const_tensor;
    ge::TensorDesc const_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
    const_desc.SetSize(2 * sizeof(int64_t));
    const_tensor.SetTensorDesc(const_desc);
    int64_t const_data[2] = {0, 6};
    const_tensor.SetData((uint8_t*)const_data, 2 * sizeof(int64_t));
    auto axes = ge::op::Constant().set_attr_value(const_tensor);
    op.set_input_axes(axes);
    auto desc = op.GetInputDesc("axes");
    desc.SetDataType(ge::DT_INT64);
    op.UpdateInputDesc("axes", desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<std::pair<int64_t, int64_t>> output_range = {{1, 1}, {1, 5}, {3, 3}, {1, 6}, {1, 1}, {5, 5}, {1, 1}};
    const auto output = op.GetOutputDesc("y");
    std::vector<std::pair<int64_t, int64_t>> actul_output_range;

    (void)output.GetShapeRange(actul_output_range);
    EXPECT_EQ(actul_output_range.size(), output_range.size());

    for (size_t i = 0UL; i < actul_output_range.size(); ++i) {
      EXPECT_EQ(actul_output_range[i], output_range[i]);
    }
  }
}