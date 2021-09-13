/**
* Copyright 2021 Huawei Technologies Co., Ltd
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

/*!
* \file test_log_sigmoid_grad_proto.cpp
* \brief
*/
#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class LogSigmoidGradTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "log_sigmoid_grad test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "log_sigmoid_grad test TearDown" << std::endl;
    }
};

TEST_F(LogSigmoidGradTest, log_sigmoid_grad_test_case_1) {
    ge::op::LogSigmoidGrad log_sigmoid_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({2, 3, 4});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    log_sigmoid_grad_op.UpdateInputDesc("grads", tensor_desc);
    log_sigmoid_grad_op.UpdateInputDesc("features", tensor_desc);

    auto ret = log_sigmoid_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = log_sigmoid_grad_op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector < int64_t > expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSigmoidGradTest, log_sigmoid_grad_test_case_2) {
    ge::op::LogSigmoidGrad log_sigmoid_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({12, 23, 34, 5});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    log_sigmoid_grad_op.UpdateInputDesc("grads", tensor_desc);
    log_sigmoid_grad_op.UpdateInputDesc("features", tensor_desc);

    auto ret = log_sigmoid_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = log_sigmoid_grad_op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector < int64_t > expected_output_shape = {12, 23, 34, 5};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSigmoidGradTest, log_sigmoid_grad_test_case_3) {
    ge::op::LogSigmoidGrad log_sigmoid_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({2, 3, 4, 5, 1, 5, 9});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    log_sigmoid_grad_op.UpdateInputDesc("grads", tensor_desc);
    log_sigmoid_grad_op.UpdateInputDesc("features", tensor_desc);

    auto ret = log_sigmoid_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = log_sigmoid_grad_op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector < int64_t > expected_output_shape = {2, 3, 4, 5, 1, 5, 9};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSigmoidGradTest, log_sigmoid_grad_test_case_4) {
    ge::op::LogSigmoidGrad log_sigmoid_grad_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({11});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    log_sigmoid_grad_op.UpdateInputDesc("grads", tensor_desc);
    log_sigmoid_grad_op.UpdateInputDesc("features", tensor_desc);

    auto ret = log_sigmoid_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = log_sigmoid_grad_op.GetOutputDesc("backprops");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector < int64_t > expected_output_shape = {11};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}