#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_training_ops.h"


// ----------------SparseApplyAdadeltaProtoTest Begin-------------------
class SparseApplyAdadeltaProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseApplyAdadelta Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseApplyAdadelta Proto Test TearDown" << std::endl;
  }
};


TEST_F(SparseApplyAdadeltaProtoTest, sparse_apply_adadelta_infershape_test){
  ge::op::SparseApplyAdadelta op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);


  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {2048, 16, 1024};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(SparseApplyAdadeltaProtoTest, sparse_apply_adadelta_verify_success_test){
  ge::op::SparseApplyAdadelta op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(SparseApplyAdadeltaProtoTest, sparse_apply_adadelta_verify_failed_01_test){
  ge::op::SparseApplyAdadelta op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdadeltaProtoTest, sparse_apply_adadelta_verify_failed_02_test){
  ge::op::SparseApplyAdadelta op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 2048}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdadeltaProtoTest, sparse_apply_adadelta_verify_failed_03_test){
  ge::op::SparseApplyAdadelta op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16, 12}, ge::DT_FLOAT));
  op.SetAttr("use_locking", false);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
// ----------------SparseApplyAdadeltaProtoTest End-------------------

// ----------------SparseApplyAdadeltaDProtoTest Begin-------------------
class SparseApplyAdadeltaDProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SparseApplyAdadeltaD Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SparseApplyAdadeltaD Proto Test TearDown" << std::endl;
  }
};


TEST_F(SparseApplyAdadeltaDProtoTest, sparse_apply_adadelta_d_infershape_test){
  ge::op::SparseApplyAdadeltaD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("epsilon", (float)0.00001);
  op.SetAttr("use_locking", false);


  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("var");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {2048, 16, 1024};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);

  auto out_accum_desc = op.GetOutputDesc("accum");
  EXPECT_EQ(out_accum_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_accum_output_shape = {2048, 16, 1024};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_accum_output_shape);

  auto out_accum_update_desc = op.GetOutputDesc("accum_update");
  EXPECT_EQ(out_accum_update_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_accum_update_output_shape = {2048, 16, 1024};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_accum_update_output_shape);
}

TEST_F(SparseApplyAdadeltaDProtoTest, sparse_apply_adadelta_d_verify_success_test){
  ge::op::SparseApplyAdadeltaD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("epsilon", (float)0.00001);
  op.SetAttr("use_locking", false);


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(SparseApplyAdadeltaDProtoTest, sparse_apply_adadelta_d_verify_failed_01_test){
  ge::op::SparseApplyAdadeltaD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({2, 16}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16,}, ge::DT_FLOAT));
  op.SetAttr("epsilon", (float)0.00001);
  op.SetAttr("use_locking", false);


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdadeltaDProtoTest, sparse_apply_adadelta_d_verify_failed_02_test){
  ge::op::SparseApplyAdadeltaD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16, 12}, ge::DT_FLOAT));
  op.SetAttr("epsilon", (float)0.00001);
  op.SetAttr("use_locking", false);


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdadeltaDProtoTest, sparse_apply_adadelta_d_verify_failed_03_test){
  ge::op::SparseApplyAdadeltaD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 2048}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16, }, ge::DT_FLOAT));
  op.SetAttr("epsilon", (float)0.00001);
  op.SetAttr("use_locking", false);


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(SparseApplyAdadeltaDProtoTest, sparse_apply_adadelta_d_verify_failed_04_test){
  ge::op::SparseApplyAdadeltaD op;
  op.UpdateInputDesc("var", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("accum_update", create_desc({2048, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("lr", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("rho", create_desc({1,}, ge::DT_FLOAT));
  op.UpdateInputDesc("epsilon", create_desc({1, }, ge::DT_FLOAT));
  op.UpdateInputDesc("grad", create_desc({16, 16, 1024}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({16, }, ge::DT_FLOAT));
  op.SetAttr("eps", (float)0.00001);
  op.SetAttr("use_locking", false);


  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
// ----------------SparseApplyAdadeltaDProtoTest End-------------------


