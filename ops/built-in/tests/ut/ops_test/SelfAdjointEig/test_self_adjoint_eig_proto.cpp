#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "linalg_ops.h"

class SelfAdjointEig : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SelfAdjointEig SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SelfAdjointEig TearDown" << std::endl;
  }
};

TEST_F(SelfAdjointEig, SelfAdjointEig_infer_shape_0) {
  ge::op::SelfAdjointEig op;
  ge::TensorDesc tensor_x_handle(ge::Shape({2,2,2}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("x", tensor_x_handle);
  op.SetAttr("compute_v", true);
  auto eigen_value_desc = op.GetOutputDesc("eigen_value");
  EXPECT_EQ(eigen_value_desc.GetDataType(), ge::DT_FLOAT);

  auto eigen_vector_desc = op.GetOutputDesc("eigen_vector");
  EXPECT_EQ(eigen_vector_desc.GetDataType(), ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);  
}

TEST_F(SelfAdjointEig, SelfAdjointEig_infer_shape_1) {
  ge::op::SelfAdjointEig op;
  ge::TensorDesc tensor_x_handle(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
  op.UpdateInputDesc("x", tensor_x_handle);
  op.SetAttr("compute_v", true);
  auto eigen_value_desc = op.GetOutputDesc("eigen_value");
  EXPECT_EQ(eigen_value_desc.GetDataType(), ge::DT_FLOAT);

  auto eigen_vector_desc = op.GetOutputDesc("eigen_vector");
  EXPECT_EQ(eigen_vector_desc.GetDataType(), ge::DT_FLOAT);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);  
}
