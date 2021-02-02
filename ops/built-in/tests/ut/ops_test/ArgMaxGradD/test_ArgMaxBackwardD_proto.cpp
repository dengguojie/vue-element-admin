#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class ArgMaxGradDTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "arg_max_grad_d Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "arg_max_grad_d Proto Test TearDown" << std::endl;
  }
};

TEST_F(ArgMaxGradDTest, arg_max_grad_d_infershape_test){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({4, 3, 4}, ge::DT_INT32));
  
  int dimension = 1;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_test_0){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));
  
  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_test_1){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = -4;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_test_2){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({6, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({6, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = -3;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_0){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = -5;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_1){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({5, 4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_2){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}


TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_3){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  //int dimension = 0;
  //op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_4){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = 5;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_5){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_6){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6,}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({2,1}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({2,1}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6,}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_7){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_8){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 1, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_9){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 2, 3}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTest, arg_max_grad_d_verify_invalid_test_10){
  ge::op::ArgMaxGradD op;
  op.UpdateInputDesc("var", create_desc({6, 4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({4, 3, 4}, ge::DT_INT32));
  op.UpdateInputDesc("assist", create_desc({6, 4, 3, 4}, ge::DT_INT32));

  int dimension = 0;
  op.SetAttr("dimension", dimension);
  
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}