/**
* Copyright 2020 Huawei Technologies Co., Ltd
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
* \file test_log_sigmoid_proto.cpp
* \brief
*/
#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "nonlinear_fuc_ops.h"

class LogSigmoidTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "log_sigmoid test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "log_sigmoid test TearDown" << std::endl;
    }
};

TEST_F(LogSigmoidTest, log_sigmoid_test_case_1) {
    // [TODO] define your op here
    ge::op::LogSigmoid log_sigmoid_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({2, 3, 4});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    // [TODO] update op input here
    log_sigmoid_op.UpdateInputDesc("x", tensor_desc);

    // [TODO] call InferShapeAndType function here
    auto ret = log_sigmoid_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = log_sigmoid_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector <int64_t> expected_output_shape = {2, 3, 4};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSigmoidTest, log_sigmoid_test_case_2) {
    // [TODO] define your op here
    ge::op::LogSigmoid log_sigmoid_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({12, 3, 4, 6});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    // [TODO] update op input here
    log_sigmoid_op.UpdateInputDesc("x", tensor_desc);

    // [TODO] call InferShapeAndType function here
    auto ret = log_sigmoid_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = log_sigmoid_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector < int64_t > expected_output_shape = {12, 3, 4, 6};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSigmoidTest, log_sigmoid_test_case_3) {
    // [TODO] define your op here
    ge::op::LogSigmoid log_sigmoid_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({2, 3, 4, 1, 2, 1});
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    // [TODO] update op input here
    log_sigmoid_op.UpdateInputDesc("x", tensor_desc);

    // [TODO] call InferShapeAndType function here
    auto ret = log_sigmoid_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = log_sigmoid_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector < int64_t > expected_output_shape = {2, 3, 4, 1, 2, 1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LogSigmoidTest, log_sigmoid_test_case_4) {
    // [TODO] define your op here
    ge::op::LogSigmoid log_sigmoid_op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({10});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);

    // [TODO] update op input here
    log_sigmoid_op.UpdateInputDesc("x", tensor_desc);

    // [TODO] call InferShapeAndType function here
    auto ret = log_sigmoid_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = log_sigmoid_op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector < int64_t > expected_output_shape = {10};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}