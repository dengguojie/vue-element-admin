#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "linalg_ops.h"

class SlogdetTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SlogdetTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SlogdetTest TearDown" << std::endl;
  }
};

TEST_F(SlogdetTest, slogdet_test_case_1) {
    ge::op::Slogdet op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({3,3});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);
    op.UpdateInputDesc("x", tensor_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc1 = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc1.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape1 = {1};
    EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape1);
    
    auto output_desc2 = op.GetOutputDesc("sign");
    EXPECT_EQ(output_desc2.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape2 = {1};
    EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape2);
}

TEST_F(SlogdetTest, slogdet_test_case_2) {
    ge::op::Slogdet op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({2,-1,2,3,3});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    tensor_desc.SetOriginShape(shape);
    op.UpdateInputDesc("x", tensor_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc1 = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc1.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape1 = {2,-1,2};
    EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape1);
    
    auto output_desc2 = op.GetOutputDesc("sign");
    EXPECT_EQ(output_desc2.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape2 = {2,-1,2};
    EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape2);
}

TEST_F(SlogdetTest, slogdet_test_case_3) {
    ge::op::Slogdet op;
    ge::TensorDesc tensor_desc;
    ge::Shape shape({3,4});
    tensor_desc.SetDataType(ge::DT_FLOAT);
    tensor_desc.SetShape(shape);
    op.UpdateInputDesc("x", tensor_desc);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}