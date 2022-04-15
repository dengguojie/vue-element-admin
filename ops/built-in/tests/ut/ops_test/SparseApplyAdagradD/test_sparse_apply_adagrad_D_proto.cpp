#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"


// ----------------SparseApplyAdagradDProtoTest Begin-------------------
class SparseApplyAdagradDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseApplyAdagradD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseApplyAdagradD Proto Test TearDown" << std::endl;
  }
};


TEST_F(SparseApplyAdagradDProtoTest, sparse_apply_adagrad_D_infershape_test){
  ge::op::SparseApplyAdagradD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));


  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("var");
  auto out_accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(out_accum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {2048, 16, 1024};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
  EXPECT_EQ(out_accum_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(SparseApplyAdagradDProtoTest, sparse_apply_adagrad_D_verify_success_test){
  ge::op::SparseApplyAdagradD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(SparseApplyAdagradDProtoTest, sparse_apply_adagrad_D_verify_failed_01_test){
  ge::op::SparseApplyAdagradD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_INT32));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}